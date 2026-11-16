import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy

import numpy as np,pandas as pd
src=DATA/'08_supplementary/NDVI90_area/01_统计结果/pixel_area_and_masks_audit.csv'
df=pd.read_csv(src);total=float(df.area_km2.sum())
low=(df.NDVI90>0)&(df.NDVI90<.55)&(df.GPP_mean_gC_m2_day>0)
high=(df.NDVI90>=.55)&(df.GPP_mean_gC_m2_day>0);other=~(low|high)
rows=[{'cover_class':name,'area_km2':float(df.loc[mask,'area_km2'].sum()),'boundary_fraction_percent':100*float(df.loc[mask,'area_km2'].sum())/total,'pixels':int(mask.sum())} for name,mask in [('low_cover',low),('higher_cover',high),('other_or_unavailable',other)]]
out=RESULTS/'supplementary';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'cover_area.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False));print('Full boundary area:',total)
