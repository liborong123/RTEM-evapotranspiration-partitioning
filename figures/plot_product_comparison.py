import sys,re,json,warnings,hashlib,shutil,argparse,platform,os
from pathlib import Path
from datetime import date,timedelta
import numpy as np
from osgeo import gdal,ogr,osr
import pandas as pd
from scipy.stats import pearsonr,linregress
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import sklearn, scipy, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator,FuncFormatter
from matplotlib.text import Text
from PIL import Image
gdal.UseExceptions();sys.stdout.reconfigure(encoding='utf-8')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy

RAW=DATA/'01_processed_inputs/daily_rasters';MODEL=DATA/'02_final_RF_outputs'
INPUTS={'RTEM':MODEL/'RTEM_T','uWUE':DATA/'01_processed_inputs/initial_uWUE_labels/T_initial','PML':RAW/'T_PML','CTE':DATA/'03_reference_products/processed_reference_cache/CTE_native_aligned_reference','NDVI':RAW/'NDVI'}
BOUNDARY=DATA/'01_processed_inputs/boundaries/Sanjiangyuan_Boundary.shp'
VEG=DATA/'01_processed_inputs/vegetation';COUNTY=None
OUT=working_copy('06_final_tables_figures/01_Fig4')
FIGDIR=OUT/'01_最终图件'
TABLEDIR=OUT/'02_计算底表与指标/统计底表'
RASTERDIR=OUT/'02_计算底表与指标/指标栅格'
AUXDIR=OUT/'03_复现资料'
FIGURE_STEM='Fig4_RTEM_FluxSat'
GT=None;WKT=None;SHAPE=None
def log(*a):print(*a,flush=True)
def mapping(folder):
 m={}
 for p in folder.glob('*.tif'):
  d=re.search(r'\d{4}-\d{2}-\d{2}',p.name)
  if d:
   key=date.fromisoformat(d[0])
   if key in m:raise ValueError(f'Duplicate {key}: {folder}')
   m[key]=p
 if not m:raise ValueError(f'No dated TIFFs: {folder}')
 return m
def read(p,kind='T',allow_align=False):
 ds=gdal.Open(str(p));b=ds.GetRasterBand(1)
 sr=osr.SpatialReference(wkt=ds.GetProjection());tr=osr.SpatialReference(wkt=WKT)
 aligned=(ds.RasterYSize,ds.RasterXSize)==SHAPE and np.allclose(ds.GetGeoTransform(),GT,rtol=0,atol=1e-7) and sr.IsSame(tr)
 if not aligned:
  if not allow_align:raise ValueError(f'Grid mismatch: {p}: {ds.GetGeoTransform()}')
  ds=gdal.Warp('',ds,format='MEM',dstSRS=WKT,outputBounds=[GT[0],GT[3]+SHAPE[0]*GT[5],GT[0]+SHAPE[1]*GT[1],GT[3]],width=SHAPE[1],height=SHAPE[0],resampleAlg='near',dstNodata=-9999)
  b=ds.GetRasterBand(1)
 a=b.ReadAsArray().astype('float32');nd=b.GetNoDataValue()
 if nd is not None:a[a==nd]=np.nan
 a[~np.isfinite(a)]=np.nan
 if kind=='T':a[a<0]=np.nan
 elif kind=='NDVI':a[(a < -1)|(a>1)]=np.nan
 return a
def save(name,a):
 ds=gdal.GetDriverByName('GTiff').Create(str(RASTERDIR/f'{name}.tif'),SHAPE[1],SHAPE[0],1,gdal.GDT_Float32,options=['COMPRESS=LZW'])
 ds.SetGeoTransform(GT);ds.SetProjection(WKT);b=ds.GetRasterBand(1);b.SetNoDataValue(-9999);b.WriteArray(np.where(np.isfinite(a),a,-9999).astype('float32'));ds=None
def load(m,dates,label,kind='T'):
 from concurrent.futures import ThreadPoolExecutor
 a=np.empty((len(dates),*SHAPE),dtype='float32')
 with ThreadPoolExecutor(max_workers=4) as pool:
  for i,arr in enumerate(pool.map(lambda d:read(m[d],kind),dates)):
   a[i]=arr
   if (i+1)%500==0:log(label,i+1,'/',len(dates))
 return a
def stats(ref,pred):
 ok=np.isfinite(ref)&np.isfinite(pred);y=ref[ok].astype('float64');x=pred[ok].astype('float64')
 if len(y)<3:return {'n':len(y)}
 rr=pearsonr(x,y);lr=linregress(x,y)
 return dict(n=int(len(y)),R2=float(r2_score(y,x)),r=float(rr.statistic),p=float(rr.pvalue),MAE=float(mean_absolute_error(y,x)),RMSE=float(np.sqrt(mean_squared_error(y,x))),bias=float(np.mean(x-y)),slope=float(lr.slope),intercept=float(lr.intercept))
def grid_metrics(ref,pred,valid,min_n):
 n=valid.sum(axis=0);den=np.maximum(n,1);y=np.where(valid,ref,0).astype('float64');x=np.where(valid,pred,0).astype('float64')
 sy=y.sum(0);sx=x.sum(0);ssy=(y*y).sum(0)-sy*sy/den;ssx=(x*x).sum(0)-sx*sx/den;cov=(x*y).sum(0)-sx*sy/den;dif=x-y
 ok=(n>=min_n)&(ssy>1e-12);v=(n>=min_n)&(ssy>1e-12)&(ssx>1e-12)
 def fill(vals,mask):return np.where(mask,vals,np.nan).astype('float32')
 with np.errstate(divide='ignore',invalid='ignore'):
  result={'R2_raw':fill(1-(dif*dif).sum(0)/ssy,ok),'r':fill(cov/np.sqrt(ssx*ssy),v),'MAE':fill(np.abs(dif).sum(0)/den,n>=min_n),'RMSE':fill(np.sqrt((dif*dif).sum(0)/den),n>=min_n),'bias':fill(dif.sum(0)/den,n>=min_n),'n':n.astype('float32')}
 result['R2_display']=np.maximum(result['R2_raw'],0)

 ix=np.flatnonzero(ok.ravel());rng=np.random.default_rng(42)
 for p in rng.choice(ix,min(25,len(ix)),replace=False):
  i,j=np.unravel_index(p,SHAPE);k=valid[:,i,j]
  np.testing.assert_allclose(result['R2_raw'][i,j],r2_score(ref[k,i,j].astype('float64'),pred[k,i,j].astype('float64')),rtol=1e-5,atol=1e-5)
  np.testing.assert_allclose(result['MAE'][i,j],mean_absolute_error(ref[k,i,j].astype('float64'),pred[k,i,j].astype('float64')),rtol=1e-5,atol=1e-5)
 return result
def geometry_lines(path):
 ds=ogr.Open(str(path));layer=ds.GetLayer();sr=layer.GetSpatialRef();dst=osr.SpatialReference();dst.ImportFromEPSG(4326);dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
 if sr:sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
 transform=osr.CoordinateTransformation(sr,dst) if sr and not sr.IsSame(dst) else None
 lines=[]
 def add(g):
  if g.GetGeometryCount():
   for i in range(g.GetGeometryCount()):add(g.GetGeometryRef(i))
  else:
   pts=np.array(g.GetPoints())[:,:2]
   if len(pts)>1:lines.append(pts.tolist())
 for f in layer:
  g=f.GetGeometryRef().Clone()
  if transform:g.Transform(transform)
  e=g.GetEnvelope()
  if e[1]>=85.9 and e[0]<=105.5 and e[3]>=29.8 and e[2]<=38.6:add(g)
 return lines
def compute():
 global GT,WKT,SHAPE
 maps={k:mapping(v) for k,v in INPUTS.items()};dates=sorted(set.intersection(*[set(maps[k]) for k in ['RTEM','uWUE','PML','NDVI']]))
 ds=gdal.Open(str(maps['RTEM'][dates[0]]));GT=ds.GetGeoTransform();WKT=ds.GetProjection();SHAPE=(ds.RasterYSize,ds.RasterXSize)
 log('COMMON',len(dates),dates[0],dates[-1],GT,SHAPE)

 maskds=gdal.GetDriverByName('MEM').Create('',SHAPE[1],SHAPE[0],1,gdal.GDT_Byte);maskds.SetGeoTransform(GT);maskds.SetProjection(WKT)
 shp=ogr.Open(str(BOUNDARY));gdal.RasterizeLayer(maskds,[1],shp.GetLayer(),burn_values=[1]);region=maskds.ReadAsArray().astype(bool)
 stacks={k:load(maps[k],dates,k) for k in ['RTEM','uWUE','PML']}
 ndvi=load(maps['NDVI'],dates,'NDVI','NDVI')
 with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);p90=np.nanpercentile(ndvi,90,axis=0).astype('float32')
 del ndvi
 stable=np.ones(SHAPE,bool)
 for year in range(2016,2021):stable &= read(VEG/f'veg_{year}001.tif','veg',allow_align=True)==10
 cache={'NDVI90':p90,'region':region,'stable_grassland':stable};summary=[];dailyvalid=region & np.isfinite(stacks['RTEM'])&np.isfinite(stacks['uWUE'])&np.isfinite(stacks['PML'])
 regional=[]
 for cl,lo,hi in [('low',0,.55),('high',.55,1)]:
  spatial=region&stable&(p90>lo)&((p90<=hi) if hi==.55 else (p90<hi))
  v=dailyvalid&spatial;nn=v.sum((1,2))
  for i,d in enumerate(dates):
   if nn[i]<5:continue
   regional.append(dict(NDVI_class=cl,date=str(d),reference='PML',scale='daily',pixels=int(nn[i]),RTEM_T=float(stacks['RTEM'][i][v[i]].mean()),reference_T=float(stacks['PML'][i][v[i]].mean())))
 for model in ['uWUE','RTEM']:
  result=grid_metrics(stacks['PML'],stacks[model],dailyvalid,30)
  for metric,a in result.items():save(f'{model}_vs_PML_daily_{metric}',np.where(region,a,np.nan))
  cache[f'{model}_PML']=result['R2_display'];a=result['R2_raw'];v=a[np.isfinite(a)]
  summary.append(dict(model=model,reference='PML',scale='daily',pixels=len(v),median_R2=float(np.median(v)),negative_R2_pixels=int((v<0).sum())))

 index={d:i for i,d in enumerate(dates)};records=[];comp={k:[] for k in ['RTEM','uWUE','PML','CTE']};weights=[]
 for anchor,p in sorted(maps['CTE'].items()):
  if anchor>dates[-1] or anchor<dates[0]:continue
  end=min(anchor+timedelta(days=7),date(anchor.year,12,31))
  support=[anchor+timedelta(days=i) for i in range((end-anchor).days+1)]
  avail=[index[d] for d in support if d in index];include=len(avail)==len(support)
  records.append(dict(anchor=str(anchor),end=str(end),expected_days=len(support),available_days=len(avail),included=include))
  if not include:continue
  for k in ['RTEM','uWUE','PML']:
   block=stacks[k][avail];valid=np.isfinite(block).all(0)
   comp[k].append(np.where(valid,block.mean(0),np.nan))
  comp['CTE'].append(read(p));weights.append(len(support))
 for k in comp:comp[k]=np.stack(comp[k])
 valid8=region.copy()[None]&np.logical_and.reduce([np.isfinite(v) for v in comp.values()]);weights=np.array(weights)[:,None,None]
 included=[r for r in records if r['included']]
 for cl,lo,hi in [('low',0,.55),('high',.55,1)]:
  spatial=region&stable&(p90>lo)&((p90<=hi) if hi==.55 else (p90<hi))
  v=valid8&spatial;nn=v.sum((1,2))
  for i,rec in enumerate(included):
   if nn[i]<5:continue
   regional.append(dict(NDVI_class=cl,date=rec['anchor'],reference='CTE',scale='8-day',pixels=int(nn[i]),RTEM_T=float(comp['RTEM'][i][v[i]].mean()),reference_T=float(comp['CTE'][i][v[i]].mean())))
 pd.DataFrame(regional).to_csv(TABLEDIR/'scatter_regional_timeseries.csv',index=False,encoding='utf-8-sig')
 for model in ['uWUE','RTEM']:
  result=grid_metrics(comp['CTE'],comp[model],valid8,10)
  for metric,a in result.items():save(f'{model}_vs_CTE_8day_{metric}',np.where(region,a,np.nan))
  cache[f'{model}_CTE']=result['R2_display'];v=result['R2_raw'][np.isfinite(result['R2_raw'])]
  summary.append(dict(model=model,reference='CTE',scale='8-day',pixels=len(v),median_R2=float(np.median(v)),negative_R2_pixels=int((v<0).sum())))

 count=np.sum(valid8*weights,axis=0);cache['paired_days']=count
 for k in comp:
  cache[f'mean_{k}']=np.where(count>0,np.sum(np.where(valid8,comp[k],0)*weights,axis=0)/np.maximum(count,1),np.nan).astype('float32')
  save(f'mean_{k}_T_common_CTE_support',cache[f'mean_{k}'])
 save('NDVI90_common_daily_dates',p90);save('stable_grassland_2016_2020',np.where(region,stable.astype('float32'),np.nan))
 cache['GT']=np.array(GT);np.savez_compressed(AUXDIR/'plot_data.npz',**cache)
 pd.DataFrame(records).to_csv(TABLEDIR/'CTE_window_inventory.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(summary).to_csv(TABLEDIR/'grid_R2_summary.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame({'date':[str(d) for d in dates],**{k:[str(maps[k][d]) for d in dates] for k in ['RTEM','uWUE','PML','NDVI']}}).to_csv(TABLEDIR/'daily_input_manifest.csv',index=False,encoding='utf-8-sig')
 missing=[str(dates[0]+timedelta(days=i)) for i in range((dates[-1]-dates[0]).days+1) if dates[0]+timedelta(days=i) not in index]
 metadata=dict(inputs={k:str(v) for k,v in INPUTS.items()},boundary=str(BOUNDARY),county=str(COUNTY),vegetation=str(VEG),grid=GT,shape=SHAPE,crs=WKT,daily_count=len(dates),start=str(dates[0]),end=str(dates[-1]),missing_dates=missing,CTE_included_windows=int(len(weights)),CTE_excluded_windows=[r for r in records if not r['included']],version='RTEM with SFE ET and FluxSat GPP; prediction percentile 85',reference_direction='sklearn r2_score(y_true=reference_T, y_pred=estimated_T)',software=dict(python=platform.python_version(),numpy=np.__version__,GDAL=gdal.VersionInfo(),sklearn=sklearn.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),manuscript_figure=4)
 (AUXDIR/'data_and_methods.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
 (AUXDIR/'map_lines.json').write_text(json.dumps({'boundary':geometry_lines(BOUNDARY),'county':json.loads((DATA/'06_final_tables_figures/01_Fig4/03_复现资料/map_lines.json').read_text(encoding='utf-8'))['county']}),encoding='utf-8')
 log('COMPUTE DONE',summary,'CTE windows',len(weights))
def panel_label(ax,t,x=-.085,y=1.02):ax.text(x,y,t+'.',transform=ax.transAxes,fontsize=28,va='bottom',ha='left',clip_on=False)
def draw():
 global GT,SHAPE
 z=np.load(AUXDIR/'plot_data.npz');GT=z['GT'];SHAPE=z['NDVI90'].shape;lines=json.loads((AUXDIR/'map_lines.json').read_text(encoding='utf-8'));meta=json.loads((AUXDIR/'data_and_methods.json').read_text(encoding='utf-8'))
 plt.rcParams.update({'font.family':'Arial','font.size':10,'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','axes.linewidth':1.15,'mathtext.fontset':'custom','mathtext.rm':'Arial:bold','mathtext.it':'Arial:bold:italic','mathtext.bf':'Arial:bold','mathtext.bfit':'Arial:bold:italic','mathtext.sf':'Arial:bold','pdf.fonttype':42,'svg.fonttype':'none','xtick.direction':'in','ytick.direction':'in'})
 fig=plt.figure(figsize=(16,13),facecolor='white')
 positions=[[.047,.696,.445,.249],[.547,.696,.445,.249],[.047,.408,.445,.249],[.547,.408,.445,.249]]
 names=['uWUE_PML','uWUE_CTE','RTEM_PML','RTEM_CTE'];cm=plt.get_cmap('GnBu').copy();cm.set_bad('white')
 for i,(pos,key) in enumerate(zip(positions,names)):
  ax=fig.add_axes(pos);ax.set_xlim(85.9,105.5);ax.set_ylim(29.8,38.6)
  ax.add_collection(LineCollection(lines['county'],colors='#b9b9b9',linewidths=.28,zorder=0))
  arr=np.where(z['region'],z[key],np.nan)
  ax.imshow(arr,extent=(GT[0],GT[0]+SHAPE[1]*GT[1],GT[3]+SHAPE[0]*GT[5],GT[3]),vmin=0,vmax=1,cmap=cm,interpolation='nearest',aspect='auto',zorder=1)
  ax.add_collection(LineCollection(lines['boundary'],colors='#505050',linewidths=1.3,zorder=2))
  ax.set_xticks([87,90,93,96,99,102,105]);ax.set_yticks([30,32,34,36,38]);ax.set_xticklabels([f'{n}°E' for n in [87,90,93,96,99,102,105]]);ax.set_yticklabels([f'{n}°N' for n in [30,32,34,36,38]])
  ax.xaxis.set_minor_locator(MultipleLocator(1.5));ax.tick_params(top=True,right=True,length=5,width=1.15,labelsize=11);ax.tick_params(which='minor',top=True,length=3)

  ax.text(87.8,37.55,'N',ha='center',va='center',fontsize=17)
  ax.add_patch(Polygon([[87.8,37.05],[87.2,35.1],[87.8,35.75]],closed=True,color='black',zorder=5));ax.add_patch(Polygon([[87.8,37.05],[88.4,35.1],[87.8,35.75]],closed=True,facecolor='white',edgecolor='black',lw=1.2,zorder=5))
  cax=ax.inset_axes([.065,.10,.025,.255]);cax.imshow(np.linspace(1,0,256)[:,None],cmap=cm,aspect='auto',vmin=0,vmax=1);cax.axis('off')
  ax.text(.115,.305,'High : 1',transform=ax.transAxes,fontsize=12,fontstyle='italic',fontweight='bold');ax.text(.115,.115,'Low : 0',transform=ax.transAxes,fontsize=12,fontstyle='italic',fontweight='bold')

  x0=96.0;y0=30.55;dl=60*1609.344/(111320*np.cos(np.deg2rad(31)))
  for j in range(8):ax.add_patch(Rectangle((x0+j*dl,y0),dl,.18,facecolor='black' if j%2==0 else 'white',edgecolor='black',lw=.5,zorder=6))
  for j in [0,1,2,4,6,8]:ax.text(x0+j*dl,y0+.26,str(j*60),ha='center',fontsize=8)
  ax.text(x0+8*dl+.13,y0,'Miles',fontsize=8,va='bottom');panel_label(ax,'abcd'[i],-.085,1.015)

 p90=z['NDVI90'];series=pd.read_csv(TABLEDIR/'scatter_regional_timeseries.csv');metrics=[]
 for j,(pos,low,high) in enumerate([([.055,.068,.237,.257],0,.55),([.355,.068,.237,.257],.55,1)]):
  ax=fig.add_axes(pos);group=series[series.NDVI_class==('low' if j==0 else 'high')]
  maxv=float(group[['RTEM_T','reference_T']].max().max());lim=max([1.2,3.6][j],np.ceil(maxv/.2)*.2)
  cols={'PML':'#4d7fb5','CTE':'#e4772a'}
  for k,name in [('PML','PML-V2'),('CTE','CTE')]:
   subset=group[group.reference==k];x=subset.RTEM_T.values;y=subset.reference_T.values;s=stats(y,x);s.update(panel='ef'[j],NDVI_class=f'{low}<NDVI90'+('<=.55' if j==0 else '<1'),reference=name);metrics.append(s)
   ax.scatter(x,y,s=9,alpha=.65,color=cols[k],edgecolors='none',label=name,rasterized=True)
   xx=np.linspace(0,lim,100);ax.plot(xx,s['slope']*xx+s['intercept'],color=cols[k],lw=1.2)
   iy=0 if k=='PML' else 1;pv='<0.001' if s['p']<.001 else f"= {s['p']:.3f}"
   txt=f"{name}: $R^2$ = {s['R2']:.3f}, p {pv}\nCorr = {s['r']:.3f}, RMSE = {s['RMSE']:.3f} $mm/day$"
   ax.text(.05,.95-.17*iy,txt,transform=ax.transAxes,va='top',color='#145361',fontsize=8.4,fontweight='bold')
  ax.plot([0,lim],[0,lim],color='#9a9a9a',ls='--',lw=1);ax.set(xlim=(0,lim),ylim=(0,lim));ax.grid(color='#dedede',lw=.45);ax.set_axisbelow(True)
  ax.set_xlabel(r'RTEM $T$ ($mm/day$)',fontweight='bold');ax.set_ylabel(r'Reference $T$ ($mm/day$)',fontweight='bold');ax.tick_params(labelsize=9)
  ax.xaxis.set_major_locator(MultipleLocator(.2 if j==0 else .4));ax.yaxis.set_major_locator(MultipleLocator(.2 if j==0 else .4))
  ax.text(.25,.05,rf'${low:g} < NDVI_{{90\%}} '+(r'\leq' if high==.55 else '<')+rf' {high:g}$',transform=ax.transAxes,color='#145361',fontsize=10,fontweight='bold')
  ax.legend(loc='lower right',fontsize=8,frameon=True,handletextpad=.3,borderpad=.3);panel_label(ax,'ef'[j],-.20,1.07)

 mask=z['region']&(p90>=0)&(p90<=1)&(z['paired_days']>=30)
 for k in ['RTEM','uWUE','PML','CTE']:mask &= np.isfinite(z[f'mean_{k}'])
 bins=[]
 for left in np.arange(0,1,.0005):
  b=mask&(p90>=left)&(p90<left+.0005);n=int(b.sum())
  if n>=5:bins.append(dict(NDVI90=left+.00025,n=n,**{k:float(z[f'mean_{k}'][b].mean()) for k in ['RTEM','uWUE','PML','CTE']}))
 df=pd.DataFrame(bins)

 shown=df[df.NDVI90>=.1].copy()
 ax=fig.add_axes([.665,.068,.323,.257]);colors={'RTEM':'#3b7fb1','uWUE':'#4a9b68','PML':'#c9c9c9','CTE':'#e77e36'}
 for k in ['RTEM','uWUE','PML','CTE']:

  xx=[];yy=[]
  for n,(xv,yv) in enumerate(zip(shown.NDVI90,shown[k])):
   if n and xv-shown.NDVI90.iloc[n-1]>.010000001:xx.append(np.nan);yy.append(np.nan)
   xx.append(xv);yy.append(yv)
  ax.plot(xx,yy,color=colors[k],lw=1.05,marker='.',markersize=.65,label=('PML-V2' if k=='PML' else k)+r' $T$')
 ax.set_xlim(0,1);ax.set_ylim(0,max(1,float(shown[['RTEM','uWUE','PML','CTE']].max().max())*1.03));ax.set_xlabel(r'$NDVI_{90\%}$',fontweight='bold');ax.set_ylabel(r'$T$ ($mm/day$)',fontweight='bold');ax.xaxis.set_major_locator(MultipleLocator(.1));ax.legend(loc='upper left',bbox_to_anchor=(.06,.96),ncol=2,fontsize=10,frameon=True,columnspacing=.8,handlelength=2,handletextpad=.3);panel_label(ax,'g',-.20,1.07)
 legend=ax.get_legend()
 for handle in legend.get_lines():
  handle.set_linewidth(3.5);handle.set_markersize(0)
 assert all(handle.get_linewidth()==3.5 for handle in legend.get_lines())

 ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'0' if abs(value)<1e-9 else f'{value:.1f}'))
 ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'' if abs(value)<1e-9 else f'{value:.1f}'))
 fig.canvas.draw()
 for label in fig.findobj(Text):label.set_fontweight('bold')
 fig.canvas.draw()
 assert all(t.get_fontweight()=='bold' for t in fig.findobj(Text))
 assert ax.get_xlim()==(0.,1.) and shown.NDVI90.min()>=.1
 assert ax.xaxis.get_major_formatter()(0)=='0' and ax.yaxis.get_major_formatter()(0)==''


 for extension,kwargs in [('png',{'dpi':220}),('tiff',{'dpi':600,'pil_kwargs':{'compression':'tiff_lzw'}}),('pdf',{'dpi':600})]:
  temporary=FIGDIR/f'.{FIGURE_STEM}.writing.{extension}'
  fig.savefig(temporary,**kwargs)
  os.replace(temporary,FIGDIR/f'{FIGURE_STEM}.{extension}')
 plt.close(fig)
 pd.DataFrame(metrics).to_csv(TABLEDIR/'scatter_metrics.csv',index=False,encoding='utf-8-sig');df.to_csv(TABLEDIR/'NDVI90_binned_T.csv',index=False,encoding='utf-8-sig')
 log('DRAW DONE',metrics)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--plot-only',action='store_true');args=parser.parse_args()
 for d in [OUT,FIGDIR,TABLEDIR,RASTERDIR,AUXDIR]:d.mkdir(parents=True,exist_ok=True)
 if not args.plot_only:compute()
 draw()
