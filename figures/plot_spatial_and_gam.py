import sys,json,os,argparse
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr,linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon,Rectangle,Patch
from matplotlib.colors import LinearSegmentedColormap,ListedColormap,BoundaryNorm
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator,FuncFormatter
sys.stdout.reconfigure(encoding='utf-8')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy
O=working_copy('05_GAM');F=O/'01_最终图件';T=O/'02_计算底表与指标/统计底表';A=O/'03_复现资料'
def style():
 plt.rcParams.update({'font.family':'Arial','font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':11,'axes.linewidth':1.2,'mathtext.fontset':'custom','mathtext.rm':'Arial:bold','mathtext.it':'Arial:bold:italic','mathtext.bf':'Arial:bold','mathtext.bfit':'Arial:bold:italic','mathtext.sf':'Arial:bold','pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in'})
def label(ax,letter,x=-.09,y=1.025):ax.text(x,y,letter+'.',transform=ax.transAxes,fontsize=30,va='bottom',ha='left',clip_on=False)
def finish(fig,stem):
 fig.canvas.draw()
 for t in fig.findobj(Text):t.set_fontweight('bold')
 fig.canvas.draw();assert all(t.get_fontweight()=='bold' for t in fig.findobj(Text))
 for ext,kw in [('png',{'dpi':220}),('tiff',{'dpi':600,'pil_kwargs':{'compression':'tiff_lzw'}}),('pdf',{'dpi':600})]:
  p=F/f'.{stem}.writing.{ext}';fig.savefig(p,**kw);os.replace(p,F/f'{stem}.{ext}')
 plt.close(fig)
def map_panel(ax,z,lines):
 gt=z['GT'];a=z['T'];height,width=a.shape;top=float(np.nanmax(a));cmap=plt.get_cmap('OrRd').copy();cmap.set_bad('white')
 ax.set(xlim=(85.9,105.5),ylim=(29.8,38.6))
 ax.add_collection(LineCollection(lines['county'],colors='#c2c2c2',linewidths=.3,zorder=0));ax.imshow(a,extent=(gt[0],gt[0]+width*gt[1],gt[3]+height*gt[5],gt[3]),vmin=0,vmax=top,cmap=cmap,aspect='auto',interpolation='nearest',zorder=1);ax.add_collection(LineCollection(lines['boundary'],colors='#505050',linewidths=1.2,zorder=2))
 ax.set_xticks([87,90,93,96,99,102,105]);ax.set_xticklabels([str(v)+'°E' for v in [87,90,93,96,99,102,105]]);ax.set_yticks([30,32,34,36,38]);ax.set_yticklabels([str(v)+'°N' for v in [30,32,34,36,38]]);ax.tick_params(top=True,right=True,labelsize=10,length=5,width=1.2);ax.xaxis.set_minor_locator(MultipleLocator(1.5));ax.tick_params(which='minor',top=True,length=3)
 ax.text(87.5,37.65,'N',ha='center',fontsize=17);ax.add_patch(Polygon([[87.5,37.3],[86.9,35.3],[87.5,35.95]],color='black',zorder=5));ax.add_patch(Polygon([[87.5,37.3],[88.1,35.3],[87.5,35.95]],facecolor='white',edgecolor='black',lw=1.3,zorder=5))
 cax=ax.inset_axes([.065,.11,.026,.27]);cax.imshow(np.linspace(1,0,256)[:,None],cmap=cmap,vmin=0,vmax=1,aspect='auto');cax.axis('off');ax.text(.005,.43,r'$T$ ($mm/day$)',transform=ax.transAxes,fontsize=12);ax.text(.105,.31,f'High : {top:.2f}',transform=ax.transAxes,fontstyle='italic',fontsize=11);ax.text(.105,.12,'Low : 0',transform=ax.transAxes,fontstyle='italic',fontsize=11)
 x0=96.;y0=30.4;dl=60*1609.344/(111320*np.cos(np.deg2rad(31)))
 for j in range(8):ax.add_patch(Rectangle((x0+j*dl,y0),dl,.18,facecolor='black' if j%2==0 else 'white',edgecolor='black',lw=.5,zorder=6))
 for j in [0,1,2,4,6,8]:ax.text(x0+j*dl,y0+.25,str(j*60),ha='center',fontsize=7)
 ax.text(x0+8*dl+.1,y0,'Miles',fontsize=7,fontstyle='italic')
 return top
def plot_spatial_patterns():
 style();z=np.load(A/'mean_layers.npz');lines=json.loads((A/'map_lines.json').read_text(encoding='utf-8'));df=pd.read_csv(T/'pixel_means_and_environment.csv');fig=plt.figure(figsize=(16,7.8),facecolor='white')
 ax=fig.add_axes([.045,.555,.405,.375]);top=map_panel(ax,z,lines);label(ax,'a',-.09,1.025)

 a=z['T'];dem=z['DEM'];valid=np.isfinite(a)&np.isfinite(dem);min_dem=np.floor(dem[valid].min()/200)*200;max_dem=np.ceil(dem[valid].max()/200)*200;edges=np.arange(min_dem,max_dem+200,200);bdata=[];centers=[];ranges=[];summary=[]
 for lo,hi in zip(edges[:-1],edges[1:]):
  b=a[valid&(dem>=lo)&(dem<hi)]
  if len(b)<200:continue
  bdata.append(b);centers.append((lo+hi)/2);ranges.append(f'{int(lo)}–{int(hi)}');q=np.percentile(b,[25,50,75]);summary.append({'elevation_low':lo,'elevation_high':hi,'n':len(b),'mean':float(b.mean()),'q25':q[0],'median':q[1],'q75':q[2]})
 ax=fig.add_axes([.045,.155,.405,.335]);ax.boxplot(bdata,positions=centers,widths=160,patch_artist=True,showfliers=False,boxprops={'facecolor':'#A6C8E0','edgecolor':'#4C72B0','alpha':.65,'linewidth':1.3},whiskerprops={'color':'#4C72B0','linewidth':1.2},capprops={'color':'#4C72B0','linewidth':1.2},medianprops={'color':'#DD8452','linewidth':2})
 ax.plot(centers,[r['median'] for r in summary],color='#DD8452',lw=2.3,label=r'Median $T$');ax.set_xlim(min(centers)-200,max(centers)+200);ax.set_xticks(centers);ax.set_xticklabels(ranges,rotation=45,ha='right',fontsize=9);ax.tick_params(top=True,right=True);ax.set_xlabel(r'Elevation ($m$)',fontsize=10);ax.set_ylabel(r'$T$ ($mm/day$)',fontsize=10);ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right',fontsize=9,frameon=False);label(ax,'b',-.09,1.025)
 pd.DataFrame(summary).to_csv(T/'Fig6_elevation_boxplot_summary.csv',index=False,encoding='utf-8-sig')

 order=['T','ET','GPP','LAI','DEM','prec','Rn','shum','srad','temp','VPD','SWC','bd','thickness','CEC','SOC','pH','clay','silt','sand','TN','TP']
 labels=[r'$T$','ET','GPP','LAI','DEM',r'$P$',r'$R_n$',r'$q_a$','SW',r'$T_a$','VPD','SWC',r'$\rho_b$',r'$H_{soil}$','CEC','SOC','pH','Clay','Silt','Sand','TN','TP']
 good=np.logical_and.reduce([np.isfinite(df[k])&(df[k]>=0) for k in order]);values=df.loc[good,order].values;rr=spearmanr(values,axis=0);corr=rr.statistic;p=rr.pvalue;np.fill_diagonal(p,0)
 pd.DataFrame(corr,index=order,columns=order).to_csv(T/'Fig6_Spearman_correlation.csv',encoding='utf-8-sig');pd.DataFrame(p,index=order,columns=order).to_csv(T/'Fig6_Spearman_pvalues.csv',encoding='utf-8-sig');df['Fig6_heatmap_mask']=good;df.to_csv(T/'pixel_means_and_environment.csv',index=False,encoding='utf-8-sig')
 ax=fig.add_axes([.535,.155,.395,.81]);cmap=LinearSegmentedColormap.from_list('original_diverging',['#417da0','#eeeeee','#c93648']);im=ax.imshow(corr,vmin=-1,vmax=1,cmap=cmap,aspect='equal',interpolation='nearest');n=len(order)
 for i in range(n):
  for j in range(n):ax.text(j,i,f'{corr[i,j]:.2f}'+('*' if p[i,j]<.01 else ''),ha='center',va='center',fontsize=4.9,color='white' if abs(corr[i,j])>.53 else '#444444')
 ax.set_xticks(np.arange(n));ax.set_xticklabels(labels,rotation=45,ha='right',fontsize=10);ax.set_yticks(np.arange(n));ax.set_yticklabels(labels,fontsize=10);ax.tick_params(length=0)
 for sp in ax.spines.values():sp.set_visible(False)
 cax=fig.add_axes([.953,.168,.017,.765]);cb=fig.colorbar(im,cax=cax,ticks=np.arange(-1,1.01,.25));cb.ax.tick_params(labelsize=9);fig.text(.48,.985,'c.',fontsize=30,va='top',ha='left')
 (A/'Fig6_summary.json').write_text(json.dumps({'T_max':top,'map_valid_pixels':int(np.isfinite(a).sum()),'boxplot_bins':len(summary),'boxplot_pixels':sum(v['n'] for v in summary),'heatmap_n':int(good.sum()),'correlation_method':'Spearman','star_threshold':.01,'shape':[16,7.8]},indent=2),encoding='utf-8')
 finish(fig,'Fig7_spatial_patterns');print('FIG6 DONE',top,'heatmapN',good.sum(),flush=True)
def plot_gam_results():
 style();scores=json.loads((A/'Fig7_model_metrics.json').read_text(encoding='utf-8'));df=pd.read_csv(T/'Fig7_test_predictions.csv');df=df[df.included_in_metrics];imp=pd.read_csv(T/'Fig7_drop_group_importance.csv').sort_values('delta_pseudo_r2',ascending=True)
 fig=plt.figure(figsize=(16,6.5),facecolor='white');ax=fig.add_axes([.065,.16,.405,.73]);mx=max(float(df.RTEM_T.max()),float(df.GAM_T.max()));lim=max(1.,np.ceil(mx/.1)*.1);boundaries=np.linspace(np.floor(df.DEM.min()/10)*10,np.ceil(df.DEM.max()/10)*10,9);colors=['#bed6e0','#83b3d7','#b5d28b','#54ac5b','#e7b1ae','#eb5257','#f4ca84','#eda843'];cmap=ListedColormap(colors);norm=BoundaryNorm(boundaries,len(colors),clip=True)
 sc=ax.scatter(df.RTEM_T,df.GAM_T,c=df.DEM,cmap=cmap,norm=norm,s=9,alpha=.75,edgecolors='none',rasterized=True);ax.plot([0,lim],[0,lim],ls='--',color='black',lw=1.1,dashes=(5,5));reg=linregress(df.RTEM_T,df.GAM_T);ax.plot([0,lim],reg.intercept+reg.slope*np.array([0,lim]),color='#3592b8',lw=1.2)
 ax.set(xlim=(0,lim),ylim=(0,lim));ax.set_xlabel(r'RTEM $T$ ($mm/day$)',fontsize=13,labelpad=12);ax.set_ylabel(r'GAM $T$ ($mm/day$)',fontsize=13,labelpad=10);ax.xaxis.set_major_locator(MultipleLocator(.2));ax.yaxis.set_major_locator(MultipleLocator(.2));ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'0' if abs(value)<1e-9 else f'{value:.1f}'));ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'' if abs(value)<1e-9 else f'{value:.1f}'));ax.tick_params(labelsize=12)
 ax.text(.065,.86,rf'$R^2$={scores["R2"]:.3f}  RMSE={scores["RMSE"]:.3f} $mm/day$  N={scores["n"]}',transform=ax.transAxes,fontsize=12)
 cax=ax.inset_axes([.51,.15,.46,.021]);cb=fig.colorbar(sc,cax=cax,orientation='horizontal',ticks=boundaries);cb.ax.set_xticklabels([str(int(v)) for v in boundaries],rotation=45,ha='right');cb.ax.tick_params(labelsize=7);cax.set_title(r'DEM ($m$)',fontsize=11,pad=4);label(ax,'a',-.13,1.035)
 ax=fig.add_axes([.625,.16,.35,.69]);colors=['#EF8D7A' if d=='Soil' else '#57A5B1' for d in imp.domain];bars=ax.barh(imp.name,imp.delta_pseudo_r2,color=colors,edgecolor='#555555',lw=.6);maxv=float(imp.delta_pseudo_r2.max());minv=min(0,float(imp.delta_pseudo_r2.min()));ax.set_xlim(minv*1.25,maxv*1.24)
 for bar,pct in zip(bars,imp['relative_importance_%']):ax.text(bar.get_width()+maxv*.01,bar.get_y()+bar.get_height()/2,f'{pct:.1f}%',va='center',fontsize=17)
 ax.set_xlabel(r'Drop in pseudo-$R^2$ (Group importance)',fontsize=18,labelpad=12);ax.set_title('Relative importance of predictor groups',fontsize=20,pad=16);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=18);ax.tick_params(axis='x',labelsize=16);ax.legend(handles=[Patch(facecolor='#57A5B1',edgecolor='black',label='Climate'),Patch(facecolor='#EF8D7A',edgecolor='black',label='Soil')],loc='lower right',frameon=False,fontsize=15);fig.text(.525,.985,'b.',fontsize=30,va='top',ha='left')
 finish(fig,'Fig8_GAM');print('FIG7 DONE',scores,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--spatial-only',action='store_true');parser.add_argument('--gam-only',action='store_true');args=parser.parse_args()
 if not args.gam_only:plot_spatial_patterns()
 if not args.spatial_only:plot_gam_results()
