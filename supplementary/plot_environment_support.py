from pathlib import Path
import json,sys,importlib.util
import numpy as np,pandas as pd
from osgeo import ogr,osr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.text import Text
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy
O=RESULTS/'supplementary';O.mkdir(parents=True,exist_ok=True)
audit=pd.read_csv(DATA/'08_supplementary/NDVI90_area/01_统计结果/pixel_area_and_masks_audit.csv');z=np.load(DATA/'06_final_tables_figures/01_Fig4/03_复现资料/plot_data.npz');region=z['region'];gt=z['GT'];shape=region.shape;assert len(audit)==int(region.sum());total=float(audit.area_km2.sum());low=(audit.NDVI90>0)&(audit.NDVI90<.55)&(audit.GPP_mean_gC_m2_day>0);high=(audit.NDVI90>=.55)&(audit.GPP_mean_gC_m2_day>0);other=~(low|high);areas=[float(audit.loc[v,'area_km2'].sum()) for v in [low,high,other]];pct=100*np.array(areas)/total
classes=np.full(shape,np.nan);classes[audit.row.values,audit.col.values]=np.select([low,high],[1,2],default=3)
env=pd.read_csv(DATA/'08_supplementary/transfer_diagnostics/02_统计底表/single_variable_environment_differences.csv');env=env[env.period=='all_dates'].copy();env['area_inside_individual_range_pct']=100-env.low_area_outside_high_full_range_pct;env.to_csv(O/'univariate_background_range_support.csv',index=False,encoding='utf-8-sig')
labels={'Qv': '$Q_v$', 'temp': '$T_a$', 'shum': '$q_a$', 'wind': '$u$', 'VPD': '$VPD$', 'GPPgradtime': '$\\partial GPP/\\partial t$', 'Rg_pot': '$Rg_{pot}$', 'Rgpot_grady': '$\\partial Rg_{pot}/\\partial y$', 'Rgpot_gradtime': '$\\partial Rg_{pot}/\\partial t$', 'CSWI': '$CSWI$', 'LAI': '$LAI$', 'SWC': '$SWC$', 'pres': '$P_a$', 'Rn': '$R_n$', 'pH': '$pH$', 'SOC': '$SOC$', 'CEC': '$CEC$', 'clay': '$clay$', 'silt': '$silt$', 'sand': '$sand$', 'bd': '$\\rho_b$', 'cf': '$cf$', 'thickness': '$H_{\\mathrm{soil}}$', 'tn': '$TN$'}
plt.rcParams.update({'font.family':'Arial','font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','axes.linewidth':1.2,'font.size':10,'mathtext.fontset':'custom','mathtext.rm':'Arial:bold','mathtext.it':'Arial:bold:italic','mathtext.bfit':'Arial:bold:italic','pdf.fonttype':42})
fig=plt.figure(figsize=(11.2,6.7));ax=fig.add_axes([.055,.30,.53,.61]);colors=['#e99179','#52a6b0','#dedede'];cmap=ListedColormap(colors);ax.imshow(classes,extent=(gt[0],gt[0]+shape[1]*gt[1],gt[3]+shape[0]*gt[5],gt[3]),cmap=cmap,norm=BoundaryNorm([.5,1.5,2.5,3.5],3),interpolation='none');lines=json.loads((DATA/'06_final_tables_figures/01_Fig4/03_复现资料/map_lines.json').read_text(encoding='utf-8'));ax.add_collection(LineCollection(lines['boundary'],colors='#444444',linewidths=.8));ax.set_xlim(89.1,102.8);ax.set_ylim(31.4,37.5);ax.set_xticks([90,93,96,99,102]);ax.set_xticklabels([f'{v}°E' for v in [90,93,96,99,102]]);ax.set_yticks([32,34,36]);ax.set_yticklabels([f'{v}°N' for v in [32,34,36]]);ax.tick_params(direction='in',top=True,right=True);ax.text(-.08,1.04,'a.',transform=ax.transAxes,fontsize=23);ax.set_title('Vegetation-cover classes',pad=12,fontsize=13)
names=[r'Low cover: $0<NDVI_{90\%}<0.55$',r'Higher cover: $NDVI_{90\%}\geq0.55$','Other / unavailable'];fig.legend(handles=[Patch(facecolor=k,label=n) for k,n in zip(colors,names)],loc='upper left',bbox_to_anchor=(.055,.335),frameon=False,fontsize=9)
bar=fig.add_axes([.08,.075,.48,.045]);left=0
for value,color in zip(pct,colors):
 bar.barh([0],[value],left=left,color=color,height=.8,edgecolor='white');bar.text(left+value/2,0,f'{value:.1f}%',ha='center',va='center',fontsize=10);left+=value
bar.set_xlim(0,100);bar.set_ylim(-.5,.5);bar.axis('off');fig.text(.32,.14,'Share of the full TRHR boundary area',ha='center',fontsize=10)
ax=fig.add_axes([.76,.13,.215,.77]);env=env.iloc[::-1];yy=np.arange(len(env));cols=['#52a6b0' if g=='climate' else '#e99179' for g in env.group];ax.barh(yy,env.area_inside_individual_range_pct,color=cols,height=.72);ax.set_yticks(yy);ax.set_yticklabels([labels.get(v,v) for v in env.feature],fontsize=10);ax.set_xlim(0,105);ax.set_xticks([0,25,50,75,100]);ax.tick_params(direction='in',top=True,right=True);ax.grid(axis='x',alpha=.18);ax.set_xlabel('Low-cover area within\nindividual reference range (%)',fontsize=10);ax.set_title('Environmental support',fontsize=13,pad=12);ax.text(-.51,1.04,'b.',transform=ax.transAxes,fontsize=23)
for y,v in zip(yy,env.area_inside_individual_range_pct):ax.text(min(v+1,100),y,f'{v:.0f}',va='center',fontsize=7.5)
for t in fig.findobj(Text):t.set_fontweight('bold')
for ext,kw in [('png',{'dpi':220}),('tiff',{'dpi':600,'pil_kwargs':{'compression':'tiff_lzw'}}),('pdf',{'dpi':600})]:fig.savefig(O/f'FigS1_low_cover_and_environmental_support.{ext}',**kw)
plt.close(fig)
summary={'full_boundary_raster_area_km2':total,'low_cover_area_km2':areas[0],'low_cover_fraction_of_full_TRHR_percent':float(pct[0]),'higher_cover_area_km2':areas[1],'other_area_km2':areas[2],'low_cover_definition':'0<NDVI90<0.55 and positive mean FluxSat GPP; all region pixels included in denominator; no stable-grassland restriction for area fraction','area_method':'WGS84 ellipsoidal equal-area cell areas with polygon pixel-center inclusion, inherited independently verified area audit','panel_b_scope':'individual long-term environmental range support of stable low-cover grassland against high-cover training locations within the study boundary; not an AOA classification','references':['Meyer and Pebesma (2021), doi:10.1111/2041-210X.13650'],'multivariate_results_in_figure':False};(O/'area_and_figure_methods.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
