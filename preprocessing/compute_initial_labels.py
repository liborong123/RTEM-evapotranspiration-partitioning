# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Tuple

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_paths import DATA,RESULTS
import numpy as np
from osgeo import gdal


gdal.UseExceptions()

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
NODATA = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="\u7528 FLUXSAT GPP \u8ba1\u7b97 T \u548c WUEt")
    parser.add_argument("--root", default=str(DATA/'01_processed_inputs/daily_rasters'))
    parser.add_argument("--gpp-dir", default="GPP_FluxSat")
    parser.add_argument("--et-dir", default="ET")
    parser.add_argument("--vpd-dir", default="VPD")
    parser.add_argument("--prec-dir", default="prec")
    parser.add_argument("--t-out-dir", default=str(RESULTS/'initial_labels/T_initial'))
    parser.add_argument("--wuet-out-dir", default=str(RESULTS/'initial_labels/WUEt'))
    parser.add_argument("--slope-name", default=str(RESULTS/'initial_labels/uWUEp.tif'))
    parser.add_argument("--sample-count-name", default=str(RESULTS/'initial_labels/sample_count.tif'))
    parser.add_argument("--summary-name", default=str(RESULTS/'initial_labels/processing_summary.json'))
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--rain-threshold", type=float, default=0.1)
    parser.add_argument("--gpp-active-fraction", type=float, default=0.1)
    parser.add_argument("--min-train-samples", type=int, default=30)
    parser.add_argument("--block-rows", type=int, default=12)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def date_map(folder: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in folder.glob("*.tif"):
        match = DATE_RE.search(path.name)
        if match:
            result[match.group(1)] = path
    if not result:
        raise RuntimeError(f"\u672a\u5728\u76ee\u5f55\u4e2d\u627e\u5230\u5e26\u65e5\u671f\u7684 TIFF: {folder}")
    return result


def raster_signature(path: Path) -> Tuple[int, int, Tuple[float, ...], str]:
    ds = gdal.Open(str(path))
    return ds.RasterXSize, ds.RasterYSize, tuple(ds.GetGeoTransform()), ds.GetProjection()


def same_grid(reference: Tuple[int, int, Tuple[float, ...], str], path: Path) -> bool:
    width, height, transform, projection = raster_signature(path)
    return (
        width == reference[0]
        and height == reference[1]
        and np.allclose(transform, reference[2], rtol=0.0, atol=1e-10)
        and projection == reference[3]
    )


def read_array(path: Path, yoff: int = 0, rows: int | None = None) -> np.ndarray:
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    if rows is None:
        array = band.ReadAsArray()
    else:
        array = band.ReadAsArray(0, yoff, ds.RasterXSize, rows)
    array = np.asarray(array, dtype=np.float32)
    nodata = band.GetNoDataValue()
    bad = ~np.isfinite(array) | (array <= -9000)
    if nodata is not None and np.isfinite(nodata):
        bad |= np.isclose(array, nodata)
    array[bad] = np.nan
    return array


def create_tif(path: Path, reference_path: Path, array: np.ndarray, description: str) -> None:
    ref = gdal.Open(str(reference_path))
    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        str(path), ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    out.SetGeoTransform(ref.GetGeoTransform())
    out.SetProjection(ref.GetProjection())
    band = out.GetRasterBand(1)
    band.SetNoDataValue(NODATA)
    band.SetDescription(description)
    band.WriteArray(np.where(np.isfinite(array), array, NODATA).astype(np.float32))
    band.FlushCache()
    out.FlushCache()
    out = None


def weighted_quantile_slope(
    gpp: np.ndarray,
    et: np.ndarray,
    vpd: np.ndarray,
    dry: np.ndarray,
    quantile: float,
    active_fraction: float,
    min_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        p95 = np.nanpercentile(gpp, 95, axis=0)
    active = gpp > (p95[np.newaxis, :, :] * active_fraction)
    valid = (
        np.isfinite(gpp) & np.isfinite(et) & np.isfinite(vpd)
        & (gpp > 0) & (et > 0) & (vpd > 0) & dry & active
    )
    sample_count = valid.sum(axis=0).astype(np.int32)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = gpp * np.sqrt(vpd) / et
    ratio[~valid] = np.inf
    weights = np.where(valid, et, 0.0).astype(np.float32)

    order = np.argsort(ratio, axis=0)
    sorted_ratio = np.take_along_axis(ratio, order, axis=0)
    sorted_weights = np.take_along_axis(weights, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0, dtype=np.float64)
    total = cumulative[-1]
    target = quantile * total
    index = np.argmax(cumulative >= target[np.newaxis, :, :], axis=0)
    slope = np.take_along_axis(sorted_ratio, index[np.newaxis, :, :], axis=0)[0]
    slope[(sample_count < min_samples) | ~(total > 0) | ~np.isfinite(slope) | (slope <= 0)] = np.nan
    return slope.astype(np.float32), sample_count


def main() -> None:
    args = parse_args()
    if not 0 < args.quantile < 1:
        raise ValueError("quantile \u5fc5\u987b\u4ecb\u4e8e 0 \u548c 1 \u4e4b\u95f4")
    if args.block_rows < 1 or args.min_train_samples < 1:
        raise ValueError("block_rows \u548c min_train_samples \u5fc5\u987b\u4e3a\u6b63\u6574\u6570")

    started = time.time()
    root = Path(args.root)
    folders = {
        "GPP": root / args.gpp_dir,
        "ET": root / args.et_dir,
        "VPD": root / args.vpd_dir,
        "PREC": root / args.prec_dir,
    }
    maps = {name: date_map(folder) for name, folder in folders.items()}
    dates = sorted(set(maps["GPP"]) & set(maps["ET"]) & set(maps["VPD"]) & set(maps["PREC"]))
    if not dates:
        raise RuntimeError("GPP\u3001ET\u3001VPD \u548c\u964d\u6c34\u6ca1\u6709\u5171\u540c\u65e5\u671f")

    reference_path = maps["GPP"][dates[0]]
    reference = raster_signature(reference_path)
    width, height = reference[:2]
    for name in ("ET", "VPD", "PREC"):
        if not same_grid(reference, maps[name][dates[0]]):
            raise RuntimeError(f"{name} \u4e0e FLUXSAT GPP \u7f51\u683c\u4e0d\u4e00\u81f4: {maps[name][dates[0]]}")

    t_out = root / args.t_out_dir
    wuet_out = root / args.wuet_out_dir
    t_out.mkdir(parents=True, exist_ok=True)
    wuet_out.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(__file__).resolve().parent / "_fluxsat_wuet_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ntime = len(dates)
    shape = (ntime, height, width)
    gpp_mm = np.memmap(temp_dir / "gpp.float32.dat", mode="w+", dtype=np.float32, shape=shape)
    et_mm = np.memmap(temp_dir / "et.float32.dat", mode="w+", dtype=np.float32, shape=shape)
    vpd_mm = np.memmap(temp_dir / "vpd.float32.dat", mode="w+", dtype=np.float32, shape=shape)
    dry_mm = np.memmap(temp_dir / "dry.uint8.dat", mode="w+", dtype=np.uint8, shape=shape)

    print(f"[1/3] \u8bfb\u53d6 {ntime} \u4e2a\u5171\u540c\u65e5\u671f\uff1a{dates[0]} \u81f3 {dates[-1]}", flush=True)
    for index, date_text in enumerate(dates):
        for name, target in (("GPP", gpp_mm), ("ET", et_mm), ("VPD", vpd_mm)):
            array = read_array(maps[name][date_text])
            if array.shape != (height, width):
                raise RuntimeError(f"\u5c3a\u5bf8\u4e0d\u4e00\u81f4: {maps[name][date_text]} {array.shape}")
            target[index] = array

        current_prec = read_array(maps["PREC"][date_text])
        current_dry = np.isfinite(current_prec) & (current_prec < args.rain_threshold)
        previous_text = (datetime.strptime(date_text, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        if previous_text in maps["PREC"]:
            previous_prec = read_array(maps["PREC"][previous_text])
            current_dry &= np.isfinite(previous_prec) & (previous_prec < args.rain_threshold)
        dry_mm[index] = current_dry.astype(np.uint8)
        if (index + 1) % 100 == 0 or index + 1 == ntime:
            print(f"  \u5df2\u8bfb\u53d6 {index + 1}/{ntime}", flush=True)

    gpp_mm.flush(); et_mm.flush(); vpd_mm.flush(); dry_mm.flush()

    print("[2/3] \u6309\u50cf\u5143\u4f30\u8ba1 uWUEp\uff080.95 \u5206\u4f4d\u56de\u5f52\u8fc7\u539f\u70b9\uff09", flush=True)
    slopes = np.full((height, width), np.nan, dtype=np.float32)
    sample_counts = np.zeros((height, width), dtype=np.int32)
    for yoff in range(0, height, args.block_rows):
        rows = min(args.block_rows, height - yoff)
        slope, count = weighted_quantile_slope(
            np.asarray(gpp_mm[:, yoff:yoff + rows, :]),
            np.asarray(et_mm[:, yoff:yoff + rows, :]),
            np.asarray(vpd_mm[:, yoff:yoff + rows, :]),
            np.asarray(dry_mm[:, yoff:yoff + rows, :], dtype=bool),
            args.quantile,
            args.gpp_active_fraction,
            args.min_train_samples,
        )
        slopes[yoff:yoff + rows] = slope
        sample_counts[yoff:yoff + rows] = count
        print(f"  \u5df2\u5904\u7406\u884c {yoff + rows}/{height}", flush=True)

    create_tif(root / args.slope_name, reference_path, slopes, "FLUXSAT-derived uWUEp slope")
    create_tif(root / args.sample_count_name, reference_path, sample_counts.astype(np.float32), "Training sample count")

    print("[3/3] \u8f93\u51fa\u65e5\u5c3a\u5ea6 T \u548c WUEt", flush=True)
    valid_pixel = np.isfinite(slopes) & (slopes > 0)
    written = 0
    for index, date_text in enumerate(dates):
        gpp = np.asarray(gpp_mm[index])
        vpd = np.asarray(vpd_mm[index])
        base_valid = valid_pixel & np.isfinite(gpp) & np.isfinite(vpd) & (gpp >= 0) & (vpd > 0)
        transpiration = np.full((height, width), np.nan, dtype=np.float32)
        transpiration[base_valid] = (
            gpp[base_valid] * np.sqrt(vpd[base_valid]) / slopes[base_valid]
        ).astype(np.float32)
        wuet = np.zeros((height, width), dtype=np.float32)
        wuet[~base_valid] = np.nan
        positive_t = base_valid & (transpiration > 0)
        wuet[positive_t] = gpp[positive_t] / transpiration[positive_t]

        create_tif(
            t_out / f"T_FluxSat_{date_text}.tif", reference_path, transpiration,
            "FLUXSAT-derived transpiration (mm d-1)",
        )
        create_tif(
            wuet_out / f"v_aWUEt_FluxSat_{date_text}.tif", reference_path, wuet,
            "FLUXSAT-derived WUEt (g C kg-1 H2O)",
        )
        written += 1
        if written % 100 == 0 or written == ntime:
            print(f"  \u5df2\u8f93\u51fa {written}/{ntime}", flush=True)

    valid_slopes = slopes[np.isfinite(slopes)]
    summary = {
        "method": "FLUXSAT GPP-based uWUEp 0.95 quantile regression through origin",
        "formula_uWUEa": "GPP * sqrt(VPD) / ET",
        "formula_T": "GPP * sqrt(VPD) / uWUEp",
        "formula_WUEt": "GPP / T",
        "input_directories": {key: str(value) for key, value in folders.items()},
        "output_T_directory": str(t_out),
        "output_WUEt_directory": str(wuet_out),
        "output_uWUEp": str(root / args.slope_name),
        "date_start": dates[0],
        "date_end": dates[-1],
        "common_date_count": ntime,
        "missing_GPP_calendar_days_within_range": (
            (datetime.strptime(dates[-1], "%Y-%m-%d") - datetime.strptime(dates[0], "%Y-%m-%d")).days + 1 - ntime
        ),
        "grid_width": width,
        "grid_height": height,
        "quantile": args.quantile,
        "rain_threshold": args.rain_threshold,
        "exclude_day_after_rain": True,
        "gpp_active_fraction_of_pixel_p95": args.gpp_active_fraction,
        "min_train_samples": args.min_train_samples,
        "valid_uWUEp_pixel_count": int(valid_slopes.size),
        "uWUEp_min": float(np.min(valid_slopes)) if valid_slopes.size else None,
        "uWUEp_median": float(np.median(valid_slopes)) if valid_slopes.size else None,
        "uWUEp_mean": float(np.mean(valid_slopes)) if valid_slopes.size else None,
        "uWUEp_max": float(np.max(valid_slopes)) if valid_slopes.size else None,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (root / args.summary_name).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)



    del gpp, vpd, gpp_mm, et_mm, vpd_mm, dry_mm
    gc.collect()
    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\u5904\u7406\u5931\u8d25: {exc}", file=sys.stderr, flush=True)
        raise
