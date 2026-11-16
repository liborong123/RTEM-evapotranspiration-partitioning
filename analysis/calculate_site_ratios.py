import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy

from project_paths import final_result,prepared_input
import numpy as np,pandas as pd,xarray as xr
prepared=prepared_input()
with xr.open_dataset(prepared,engine='netcdf4') as base,xr.open_dataset(final_result(),engine='netcdf4') as model:
    dt=pd.to_datetime(base.timestamp.values);et=base.ET.isel(lat=57,lon=221).values;t=model.RTEM_T.isel(lat=57,lon=221).values
months=[(2017,m) for m in [6,7,8,9]]+[(2018,m) for m in [7,8,9]];rows=[]
for year,month in months:
    ok=(dt.year==year)&(dt.month==month)&np.isfinite(et)&np.isfinite(t)&(et>=0)&(t>=0)
    ratio=float(t[ok].astype('float64').sum()/et[ok].astype('float64').sum())
    rows.append({'year':year,'month':month,'valid_days':int(ok.sum()),'TET':ratio})
rows.append({'year':'seven_month_mean','month':0,'valid_days':None,'TET':float(np.mean([r['TET'] for r in rows]))})
dest=RESULTS/'evaluation';dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(dest/'site_monthly_ratios.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
