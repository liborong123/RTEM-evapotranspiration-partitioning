import numpy as np
import warnings
def mad_flags(a,group=15):

    h,w,t=a.shape;n=(t+group-1)//group;p=np.full((h,w,n*group),np.nan,dtype=a.dtype);p[:,:,:t]=a;b=p.reshape(h,w,n,group)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(b,axis=-1,keepdims=True);dev=np.abs(b-med);mad=np.nanmedian(dev,axis=-1,keepdims=True)*1.4826;th=3.0*mad
    flags=np.isfinite(b)&((~np.isfinite(mad))|(mad==0)|((b>=med-th)&(b<=med+th)))
    return flags.reshape(h,w,n*group)[:,:,:t].copy()

def valid_wuet(values):
    return mad_flags(values)&(values<100)
