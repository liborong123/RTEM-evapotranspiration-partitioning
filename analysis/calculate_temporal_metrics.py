import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy

import numpy as np,pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
table=DATA/'06_final_tables_figures/03_Fig5/02_统计底表/plotting_series_with_gaps.csv'
df=pd.read_csv(table);inventory=pd.read_csv(table.parent/'window_inventory.csv').set_index('anchor');days=df.anchor.map(inventory.expected_days);dates=pd.to_datetime(df.anchor)+pd.to_timedelta((days-1)/2,unit='D');rows=[]
for field in ['T_mm_day','TET_ratio_of_area_sums']:
    for reference in ['PML','CTE']:
        y=df[f'{reference}_{field}'].values;p=df[f'RTEM_{field}'].values
        for season,mask in [('all_dates',np.ones(len(df),bool)),('Apr_Sep',dates.dt.month.isin([4,5,6,7,8,9]).values)]:
            ok=mask&np.isfinite(y)&np.isfinite(p);a=y[ok];b=p[ok]
            rows.append({'variable':field,'reference':reference,'season':season,'n':int(ok.sum()),'R2':float(r2_score(a,b)),'r':float(pearsonr(a,b).statistic),'MAE':float(mean_absolute_error(a,b)),'RMSE':float(np.sqrt(mean_squared_error(a,b))),'bias':float(np.mean(b-a))})
dest=RESULTS/'evaluation';dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(dest/'temporal_metrics.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
