from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Polygon,Rectangle,Patch
from matplotlib.text import Text
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy
F=RESULTS/'figures';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.weight':'bold','font.size':13,'axes.labelweight':'bold','axes.titleweight':'bold','mathtext.fontset':'custom','mathtext.rm':'Arial:bold','mathtext.it':'Arial:bold:italic','mathtext.bfit':'Arial:bold:italic','pdf.fonttype':42})
def save(fig,stem):
 for t in fig.findobj(Text):t.set_fontweight('bold')
 for ext,kw in [('png',{'dpi':220}),('tiff',{'dpi':600,'pil_kwargs':{'compression':'tiff_lzw'}}),('pdf',{})]:fig.savefig(F/f'{stem}.{ext}',**kw)
 plt.close(fig)

src=DATA/'02_final_RF_outputs/feature_importances.csv';df=pd.read_csv(src).sort_values('importance',ascending=False);labels={'temp':r'$T_a$','Rn':r'$R_n$','shum':r'$q_a$','Rgpot_grady':r'$\partial Rg_{pot}/\partial y$','Rg_pot':r'$Rg_{pot}$','pres':r'$P_a$','Rgpot_gradtime':r'$\partial Rg_{pot}/\partial t$','LAI':r'$LAI$','wind':r'$u$','CSWI':r'$CSWI$','GPPgradtime':r'$\partial GPP/\partial t$'};veg={'LAI','GPPgradtime'};df['group']=df.feature.map(lambda k:'Vegetation / productivity' if k in veg else 'Hydroclimatic predictors');df['importance_percent']=df.importance*100;totals=df.groupby('group').importance_percent.sum().to_dict();colors={'Vegetation / productivity':'#4a9b68','Hydroclimatic predictors':'#57a5b1'}
fig,ax=plt.subplots(figsize=(8.5,8.5));fig.subplots_adjust(left=.29,right=.94,bottom=.13,top=.85);bars=ax.barh(np.arange(len(df)),df.importance_percent,color='#57a5b1',edgecolor='#555555',lw=.5,height=.72);ax.invert_yaxis();ax.set_yticks(range(len(df)));ax.set_yticklabels([labels[k] for k in df.feature],fontsize=14);ax.set_xlim(0,36);ax.tick_params(direction='in',top=True,right=True,width=1.2,labelsize=12);ax.tick_params(axis='y',length=0,pad=8);ax.set_xlabel('RF importance (%)',fontsize=16,labelpad=12);ax.grid(axis='x',color='#dedede',lw=.6);ax.set_axisbelow(True)
for b,v in zip(bars,df.importance_percent):ax.text(v+.35,b.get_y()+b.get_height()/2,f'{v:.2f}',va='center',fontsize=12.5)
fig.suptitle('Predictor importance in RTEM',x=.54,y=.97,fontsize=21);fig.text(.54,.90,r'RF prediction of daily $WUE_t$',ha='center',fontsize=15)
save(fig,'FigS2_RF_importance');df[['feature','importance','importance_percent']].to_csv(F/'FigS2_RF_importance.csv',index=False,encoding='utf-8-sig');print('STATIC_FIGURES_COMPLETE; no grouping or group legend')
