"""Load the aligned arrays and screening flags used by the 11-predictor model."""

from pathlib import Path
import numpy as np
import xarray as xr

from rtem_algorithm import DEFAULT_RF_FEATURES

SCREENING_FIELDS = [
    "ET",
    "GPP",
    "WUEt",
    "qualityFlag",
    "vegFlag",
    "NDVIFlag",
    "posFlag",
    "WUEtFlag",
]


def load_model_inputs(path, cswi_limit=-0.5):
    """Read existing labels and flags without recomputing their screening rules.

    The source experiment reused these arrays to keep training samples unchanged
    when replacing the former predictor set with the eleven published inputs.
    Returns the model inputs, an optional reference RF mask and source metadata.
    """
    path = Path(path)
    required = list(dict.fromkeys(DEFAULT_RF_FEATURES + SCREENING_FIELDS))
    with xr.open_dataset(path, engine="netcdf4") as source:
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError("Missing input variables: " + ", ".join(missing))
        if any(name not in source.sizes for name in ("lat", "lon", "timestamp")):
            raise ValueError("Input dimensions must include lat, lon and timestamp")
        data = source[required].load()
        for name in required:
            if data[name].dims != ("lat", "lon", "timestamp"):
                raise ValueError(f"{name} must use dimensions (lat, lon, timestamp)")
        metadata = dict(source.attrs)
        reference_mask = None
        if "RFflag" in source:
            flag = source["RFflag"]
            if "CSWIlims" in flag.dims:
                values = np.asarray(source["CSWIlims"].values, dtype=float)
                matches = np.flatnonzero(np.isclose(values, cswi_limit))
                if len(matches) != 1:
                    raise ValueError(
                        "No unique reference RFflag for the CSWI threshold"
                    )
                flag = flag.isel(CSWIlims=int(matches[0]))
            if flag.dims != ("lat", "lon", "timestamp"):
                raise ValueError("Reference RFflag dimensions do not match the inputs")
            reference_mask = flag.values.astype(bool)
    return data, reference_mask, metadata
