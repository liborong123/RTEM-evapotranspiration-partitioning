from pathlib import Path
import json,sys,os
import numpy as np,pandas as pd,xarray as xr
from osgeo import gdal
sys.stdout.reconfigure(encoding='utf-8');gdal.UseExceptions()
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy
from project_paths import final_result
SOURCE=DATA/'01_processed_inputs/GAM_environment_cache';O=working_copy('05_GAM');A=O/'03_复现资料';T=O/'02_计算底表与指标/统计底表';M=O/'02_计算底表与指标/平均栅格'
for p in [A,T,M,O/'01_最终图件']:p.mkdir(parents=True,exist_ok=True)
with np.load(SOURCE/'environment_means_template.npz') as z:layers={k:z[k] for k in z.files}
expected=layers.pop('expected_T_valid_mask');ds=xr.open_dataset(final_result(),engine='netcdf4');a=ds.RTEM_T.transpose('timestamp','lat','lon').values.astype('float32');a[(a<0)|~np.isfinite(a)]=np.nan
good=np.isfinite(a);n=good.sum(0);total=np.sum(np.where(good,a,0),axis=0,dtype='float64');mean=np.where(n>0,total/np.maximum(n,1),np.nan).astype('float32');gt=layers['GT'];rawgt=(89.4,.05,0,37.2,0,-.05);rr=int(round((gt[3]-rawgt[3])/rawgt[5]));cc=int(round((gt[0]-rawgt[0])/rawgt[1]));hh,ww=expected.shape;new=mean[rr:rr+hh,cc:cc+ww].copy();new=np.where(layers['study'],new,np.nan);assert new.shape==expected.shape and np.array_equal(expected,np.isfinite(new));layers['T']=new
np.savez_compressed(A/'mean_layers.npz',**layers);df=pd.read_csv(SOURCE/'pixel_environment_template.csv');df.insert(5,'T',new.ravel());df.to_csv(T/'pixel_means_and_environment.csv',index=False,encoding='utf-8-sig')
manifest=pd.read_csv(SOURCE/'environment_daily_manifest.csv');manifest['T']=[str(next((DATA/'02_final_RF_outputs/RTEM_T').glob('*'+d+'.tif'))) for d in manifest.date];manifest.to_csv(T/'daily_input_manifest.csv',index=False,encoding='utf-8-sig')
ref=gdal.Open(str(next((DATA/'02_final_RF_outputs/RTEM_T').glob('*.tif'))));out=gdal.GetDriverByName('GTiff').Create(str(M/f"T_mean_common_{ds.sizes['timestamp']}days.tif"),ww,hh,1,gdal.GDT_Float32,options=['COMPRESS=LZW']);out.SetGeoTransform(tuple(gt));out.SetProjection(ref.GetProjection());band=out.GetRasterBand(1);band.SetNoDataValue(-9999);band.WriteArray(np.where(np.isfinite(new),new,-9999));out=None
meta=json.loads((SOURCE/'data_sources_and_methods.json').read_text(encoding='utf-8'));meta['input_folders']['T']=str(DATA/'02_final_RF_outputs/RTEM_T');meta.update({'version':'RTEM with SFE ET and FluxSat GPP; prediction percentile 85','status':'prepared','unchanged_environment_means_reused':True,'T_recomputed_on_all_common_dates':True,'growing_season_if_used':'April–September','environment_template':str(SOURCE/'environment_means_template.npz'),'common_daily_dates':int(ds.sizes['timestamp'])});(A/'data_sources_and_methods.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
if A.resolve()!=SOURCE.resolve():(A/'map_lines.json').write_bytes((SOURCE/'map_lines.json').read_bytes())
print('PREPARED_FIG67',float(np.nanmax(new)),int(np.isfinite(new).sum()),flush=True)

