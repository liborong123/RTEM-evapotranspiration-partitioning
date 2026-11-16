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

fig,ax=plt.subplots(figsize=(12.8,8.1));fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,12.8),ylim=(0,8.1));ax.axis('off');blue='#075575';green='#b0d2ab';darkgreen='#a7cba3';orange='#f7c99e'
def box(x,y,w,h,fc,rounded=True,lw=1.6):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.14' if rounded else 'square,pad=0',facecolor=fc,edgecolor='black',lw=lw);ax.add_patch(p)
def text(x,y,s,size=13,color='black',ha='center'):ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,fontweight='bold')
def arrow(points):
 for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=blue,lw=1.9)
 ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops={'arrowstyle':'-|>','color':blue,'lw':1.9,'mutation_scale':15})
box(.08,6.99,12.64,.98,orange);text(.35,7.48,'Input\nData',16,blue,ha='left')
for i,label in enumerate(['Meteorology\nDaily','Radiation\nDaily','Carbon Flux\nDaily','Land Cover\nAnnual','Canopy State\nDaily']):
 x=1.57+i*2.18;ax.add_patch(Polygon([(x,7.11),(x+.45,7.86),(x+2.47,7.86),(x+2.02,7.11)],facecolor=orange,edgecolor='black',lw=1.4));text(x+1.24,7.48,label,12.5)
box(.08,5.48,7.05,1.02,green);text(.35,6.23,'Data Preprocessing',16,blue,ha='left');box(.32,5.63,2.65,.47,green,False,1);text(1.65,5.87,'Quality control',12.5);box(4.05,5.63,2.73,.47,green,False,1);text(5.42,5.87,'Resample to 0.05°',12.5);arrow([(3.02,5.87),(3.96,5.87)])
box(.08,3.75,7.05,1.14,green);text(.35,4.64,'Calculate ET',17,blue,ha='left');box(.32,3.94,3.7,.47,green,False,1);text(2.16,4.18,'Surface Flux Equilibrium (SFE)',12);ax.add_patch(Polygon([(4.05,3.94),(4.5,4.5),(6.8,4.5),(6.36,3.94)],facecolor=green,edgecolor='black'));text(5.5,4.20,'Evapotranspiration\n($ET$) Daily',12)
box(.08,1.23,7.05,1.95,green);text(.35,2.82,'Transpiration Estimation\n(High vegetation cover areas)',16,blue,ha='left');box(.32,1.48,3.7,.82,green,False,1);text(2.17,1.89,'Underlying water-use efficiency\n($uWUE$)',12);box(4.26,1.48,2.52,.82,green,False,1);text(5.52,1.89,'Transpiration ($T$)\nDaily',12)
box(7.46,1.23,5.26,5.27,green);text(7.72,5.95,'Transpiration Estimation\n(Low vegetation cover areas)',16,blue,ha='left');box(7.75,4.36,4.65,1.0,green,False,1);text(10.08,4.86,'$WUE_t$ training = gross primary productivity\n($GPP$) / $T$',12);box(7.75,2.79,4.65,.81,darkgreen,False,1);text(10.08,3.20,'Machine Learning Model\n(Random Forest)',14,blue);ax.add_patch(Polygon([(7.75,1.57),(8.42,2.24),(12.4,2.24),(11.73,1.57)],facecolor=green,edgecolor='black'));text(10.08,1.91,'$WUE_{t,\\mathrm{pred}}$\nDaily 0.05°',12)
box(.08,.07,12.64,.73,'#6ba6c3');text(.35,.43,'Final\nProducts',16,blue,ha='left');text(8.1,.43,'Regional vegetation transpiration\nTranspiration fraction ($T/ET$)',14)
for pts in [[(2.78,6.96),(2.78,6.55)],[(10.08,6.96),(10.08,6.55)],[(2.78,5.44),(2.78,4.94)],[(2.78,3.71),(2.78,3.22)],[(7.15,4.28),(7.32,4.28),(7.32,4.86),(7.70,4.86)],[(6.82,1.89),(7.32,1.89),(7.32,4.86),(7.70,4.86)],[(10.08,4.31),(10.08,3.65)],[(10.08,2.74),(10.08,2.29)],[(10.08,1.19),(10.08,.84)]]:arrow(pts)
save(fig,'Fig2_framework')

