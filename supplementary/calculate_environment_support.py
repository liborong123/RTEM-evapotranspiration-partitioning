import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy
import numpy as np,pandas as pd
def wmean(x,w):return float(np.average(x,weights=w))
def wvar(x,w):return float(np.average((x-wmean(x,w))**2,weights=w))
def wquant(x,w,p):
    ix=np.argsort(x);return np.interp(p,(np.cumsum(w[ix])-.5*w[ix])/np.sum(w),x[ix])
def wks(x,w,y,v):
    ix=np.argsort(x);iy=np.argsort(y);xx=x[ix];yy=y[iy];points=np.unique(np.r_[xx,yy]);cx=np.r_[0,np.cumsum(w[ix])/sum(w)];cy=np.r_[0,np.cumsum(v[iy])/sum(v)]
    return float(np.max(abs(cx[np.searchsorted(xx,points,side='right')]-cy[np.searchsorted(yy,points,side='right')])))

z=np.load(DATA/'08_supplementary/transfer_diagnostics/03_复现资料/environment_backgrounds.npz');names=z['features'].tolist();shape=z['region'].shape;gt=z['GT']
e2=.0066943799901413165;e=np.sqrt(e2);a=6378137.
def q(phi):
    s=np.sin(np.deg2rad(phi));return (1-e2)*(s/(1-e2*s*s)+np.arctanh(e*s)/e)
top=gt[3]+np.arange(shape[0])*gt[5];rowarea=a*a*abs(np.deg2rad(gt[1])*(q(top)-q(top+gt[5])))/2/1e6
area=np.repeat(rowarea[:,None],shape[1],axis=1)
base=z['region']&z['stable_grassland']&(z['NDVI90']>0)&np.isfinite(z['GPP_mean'])&(z['GPP_mean']>0)
high=base&(z['NDVI90']>.55)&(z['traincount']>0);low=base&(z['NDVI90']<=.55)
cube=np.stack([z['all_dates_'+n] for n in names],axis=2);finite=np.isfinite(cube).all(2);h=high&finite;l=low&finite;hw=area[h];lw=area[l];Xh=cube[h];Xl=cube[l];rows=[]
for j,name in enumerate(names):
    x=Xh[:,j];y=Xl[:,j];outside=(y<x.min())|(y>x.max());sx=np.sqrt((wvar(x,hw)+wvar(y,lw))/2)
    rows.append({'feature':name,'high_mean':wmean(x,hw),'low_mean':wmean(y,lw),'SMD_low_minus_high':(wmean(y,lw)-wmean(x,hw))/sx if sx else np.nan,'weighted_KS_distance':wks(x,hw,y,lw),'area_inside_individual_range_pct':100-100*wmean(outside.astype(float),lw)})
out=RESULTS/'supplementary';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'environment_support.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
