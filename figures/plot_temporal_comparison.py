import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy

import os,json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,matplotlib.dates as mdates
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator,FixedLocator
from matplotlib.patches import Patch
OUT=working_copy('06_final_tables_figures/03_Fig5');TAB=OUT/'02_统计底表';AUX=OUT/'03_复现资料';FIG=OUT/'01_最终图件'
FIG.mkdir(parents=True,exist_ok=True)
log=print
def draw(product_keys=('RTEM','uWUE','PML','CTE'),scope_label=None):
 meta=json.loads((AUX/'data_sources_and_methods.json').read_text(encoding='utf-8'));df=pd.read_csv(TAB/'plotting_series_with_gaps.csv');inventory=pd.read_csv(TAB/'window_inventory.csv').set_index('anchor');days=df.anchor.map(inventory.expected_days);x=pd.to_datetime(df.anchor)+pd.to_timedelta((days-1)/2,unit='D')
 plt.rcParams.update({'font.family':'Arial','font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':11,'axes.linewidth':1.3,'mathtext.fontset':'custom','mathtext.rm':'Arial:bold','mathtext.it':'Arial:bold:italic','mathtext.bf':'Arial:bold','mathtext.bfit':'Arial:bold:italic','pdf.fonttype':42})
 fig,axes=plt.subplots(2,1,figsize=(14,8.2),sharex=True);fig.subplots_adjust(left=.085,right=.975,bottom=.11,top=.94,hspace=.37)
 colors={'RTEM':'#3b7fb1','uWUE':'#4a9b68','PML':'#b8b8b8','CTE':'#e77e36'};names={'RTEM':'RTEM','uWUE':'uWUE','PML':'PML-V2','CTE':'CTE'}
 start=pd.Timestamp(meta['daily_start']);end=pd.Timestamp(meta['daily_end']);hidden={};plotlines=[];season_color='#f1f0e5'
 for j,(ax,field,ylabel) in enumerate(zip(axes,['T_mm_day','TET_ratio_of_area_sums'],[r'$T$ ($mm/day$)',r'$T/ET$'])):
  for year in range(start.year,end.year+1):
   ax.axvspan(pd.Timestamp(year,4,1),pd.Timestamp(year,10,1),facecolor=season_color,alpha=.75,zorder=0)
   if year>start.year:ax.axvline(pd.Timestamp(year,1,1),color='#bdbdbd',lw=.6,ls=(0,(4,3)),zorder=.5)
   left=max(start,pd.Timestamp(year,1,1));right=min(end,pd.Timestamp(year+1,1,1));mid=left+(right-left)/2
   ax.text(mdates.date2num(mid),-.22,str(year),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=11,fontweight='bold')
  for k in product_keys:
   y=df[f'{k}_{field}'].copy()
   if j==1:
    invalid=y>1;hidden[k]={'count':int(invalid.sum()),'anchors':df.loc[invalid,'anchor'].tolist(),'values':y[invalid].tolist()};y=y.mask(invalid)
   line,=ax.plot(x,y,color=colors[k],lw=1.8,label=names[k]+(r' $T$' if j==0 else r' $T/ET$'),zorder=3)
   if j==1:plotlines.append(line)
  ax.set_ylabel(ylabel,fontsize=14,labelpad=10);ax.tick_params(which='both',direction='in',top=True,right=True,labeltop=False,labelright=False,width=1.2);ax.tick_params(which='major',length=5);ax.tick_params(which='minor',length=2.5);ax.yaxis.set_minor_locator(AutoMinorLocator(2));ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'));ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=16));ax.tick_params(axis='x',labelbottom=True,labelsize=8,pad=5,labelrotation=45);ax.grid(axis='y',color='#dedede',lw=.5,alpha=.65)
  for label in ax.get_xticklabels():label.set_ha('right');label.set_rotation_mode('anchor')
  for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(1.3)
  ax.text(-.075,1.02,'ab'[j]+'.',transform=ax.transAxes,fontsize=27,va='bottom',fontweight='bold')
  handles,labels=ax.get_legend_handles_labels();handles.append(Patch(facecolor=season_color,edgecolor='#d0cfbf'));labels.append('Apr–Sep')
  legend=ax.legend(handles,labels,loc='upper left',ncol=5,fontsize=10.5,frameon=True,framealpha=1,facecolor='white',edgecolor='#aaaaaa',borderpad=.55,handlelength=2.6,columnspacing=1.5)
  for handle in legend.get_lines():handle.set_linewidth(3.5)
  assert all(handle.get_linewidth()==3.5 for handle in legend.get_lines())
  if j==1:
   ax.set_ylim(0,1.3);ax.yaxis.set_major_locator(FixedLocator([0,.2,.4,.6,.8,1,1.2,1.3]));ax.axhline(1,color='#777777',ls='--',lw=.8,zorder=1)
  else:ax.set_ylim(0,2.65)
 axes[1].set_xlabel('');axes[1].set_xlim(start,end)
 fig.canvas.draw()
 for text in fig.findobj(Text):text.set_fontweight('bold')
 fig.canvas.draw();assert all(t.get_fontweight()=='bold' for t in fig.findobj(Text));assert all(sp.get_visible() for ax in axes for sp in ax.spines.values())
 assert axes[1].get_ylim()==(0.,1.3) and all(np.nanmax(line.get_ydata())<=1 for line in plotlines)
 if scope_label:axes[0].text(.99,.96,scope_label,transform=axes[0].transAxes,ha='right',va='top',fontsize=11,fontweight='bold')
 fig.canvas.draw()
 assert all(len(ax.get_legend().get_texts())==len(product_keys)+1 for ax in axes)
 assert all(t.get_rotation()==45 for ax in axes for t in ax.get_xticklabels())
 audit={'TET_values_above_1_hidden_as_NaN_gaps':hidden,'TET_ylim':[0,1.3],'legends_both_panels':True,'legend_labels':[[t.get_text() for t in ax.get_legend().get_texts()] for ax in axes],'monthly_tick_labels_both_panels':True,'year_labels':'centered below each visible calendar-year interval','shading':'growing season Apr–September','all_text_bold':True,'four_spines_and_inward_ticks':True,'original_computation_tables_modified':False}
 audit['month_label_rotation_degrees']=45
 (AUX/'plot_display_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');meta['plot_display']=audit;(AUX/'data_sources_and_methods.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 for ext,kw in [('png',{'dpi':220}),('tiff',{'dpi':600,'pil_kwargs':{'compression':'tiff_lzw'}}),('pdf',{'dpi':600})]:
  tmp=FIG/f'.T_and_TET_temporal.writing.{ext}';fig.savefig(tmp,**kw);os.replace(tmp,FIG/f'T_and_TET_temporal.{ext}')
 plt.close(fig);log('DRAWN',FIG)

if __name__=='__main__':draw()
