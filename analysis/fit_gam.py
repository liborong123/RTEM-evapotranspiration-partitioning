import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,json,platform,pickle
from pathlib import Path
import numpy as np,pandas as pd
from pygam import LinearGAM,s
from pygam.terms import TermList
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pygam,sklearn,scipy
sys.stdout.reconfigure(encoding='utf-8')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy
O=working_copy('05_GAM');T=O/'02_计算底表与指标/统计底表';A=O/'03_复现资料'
CLIMATE=['prec','shum','VPD','SWC','Rn','srad','temp','wind'];SOIL=['bd','CEC','SOC','TN','TP','clay','silt','sand','pH','thickness','cf']
GROUPS={'Climate Water':['prec','shum','VPD','SWC'],'Climate Energy':['Rn','srad','temp'],'Climate Dynamics':['wind'],'Soil Fertility':['CEC','SOC','TN'],'Soil Texture':['clay','silt','sand','cf'],'Soil Chemistry':['pH'],'Soil Physical':['bd','thickness']}
def log(*a):print(*a,flush=True)
def fit(X,y):
 model=LinearGAM(TermList(*[s(i,n_splines=20) for i in range(X.shape[1])]),fit_intercept=True)
 return model.gridsearch(X,y,progress=False)
def metrics(y,p):
 v=(y>0)&np.isfinite(y)&np.isfinite(p);return {'n':int(v.sum()),'R2':float(r2_score(y[v],p[v])),'MAE':float(mean_absolute_error(y[v],p[v])),'RMSE':float(np.sqrt(mean_squared_error(y[v],p[v])))}
if __name__=='__main__':
 df=pd.read_csv(T/'pixel_means_and_environment.csv');raw=CLIMATE+SOIL
 valid=np.logical_and.reduce([np.isfinite(df[k].values) for k in ['T','DEM']+raw]);df=df.loc[valid].copy();X=df[raw].values.astype('float32');y=df['T'].values.astype('float32');indices=np.arange(len(df));dem=df.DEM.values.astype('float32')
 itr,ite=train_test_split(indices,test_size=.2,random_state=42);Xtr0=X[itr];Xte0=X[ite];ytr=y[itr];yte=y[ite]
 log('TRAIN/TEST',len(itr),len(ite),'valid input',len(df))
 features=[];trcols=[];tecols=[];diag=[];loadings=[];transforms={};groupsnew={}
 for group,members in GROUPS.items():
  cols=[raw.index(v) for v in members];scale=StandardScaler();train=scale.fit_transform(Xtr0[:,cols]);test=scale.transform(Xte0[:,cols]);pca=PCA();ztr=pca.fit_transform(train);zte=pca.transform(test);cum=np.cumsum(pca.explained_variance_ratio_);n=max(1,int(np.argmax(cum>=.95)+1));groupsnew[group]=[]
  for j in range(n):
   name=f'{group}_PC{j+1}';features.append(name);groupsnew[group].append(name);trcols.append(ztr[:,j]);tecols.append(zte[:,j]);diag.append({'group':group,'pc':name,'explained_variance_ratio':float(pca.explained_variance_ratio_[j]),'cum_explained_variance':float(cum[j])})
   loadings.extend({'group':group,'pc':name,'variable':v,'loading':float(value)} for v,value in zip(members,pca.components_[j]))
  transforms[group]={'members':members,'scaler':scale,'pca':pca,'n_pc':n};log('PCA',group,n,'/',len(members),'variance',float(cum[n-1]))
 Xtr=np.column_stack(trcols);Xte=np.column_stack(tecols);mu=np.nanmean(Xtr,axis=0);sd=np.nanstd(Xtr,axis=0);sd[sd==0]=1;Xtr=(Xtr-mu)/sd;Xte=(Xte-mu)/sd
 pd.DataFrame(diag).to_csv(T/'Fig7_PCA_trainfit_diagnostics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(loadings).to_csv(T/'Fig7_PCA_trainfit_loadings.csv',index=False,encoding='utf-8-sig')
 log('FIT FULL',len(features),'features');full=fit(Xtr,ytr);pseudo=float(full.statistics_['pseudo_r2']['explained_deviance']);pred=full.predict(Xte);scores=metrics(yte,pred);scores['training_pseudo_R2']=pseudo;scores['train_n']=len(itr);scores['valid_input_n']=len(df);scores['selected_lambda']=np.array(full.lam).ravel().tolist();log('FULL',scores)

 with (A/'Fig7_full_GAM_and_train_transforms.pkl').open('wb') as f:pickle.dump({'model':full,'groups':transforms,'global_mean':mu,'global_sd':sd,'features':features,'train_indices':itr,'test_indices':ite,'row_col':df[['row','col']].values},f)
 pairs=df.iloc[ite][['row','col','longitude','latitude','DEM']].copy();pairs['RTEM_T']=yte;pairs['GAM_T']=pred;pairs['included_in_metrics']=(yte>0)&np.isfinite(pred);pairs.to_csv(T/'Fig7_test_predictions.csv',index=False,encoding='utf-8-sig')
 (A/'Fig7_model_metrics.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
 rows=[]
 for group,names in groupsnew.items():
  log('DROP',group);keep=[j for j,n in enumerate(features) if n not in names];reduced=fit(Xtr[:,keep],ytr);rp=float(reduced.statistics_['pseudo_r2']['explained_deviance']);rm=metrics(yte,reduced.predict(Xte[:,keep]));rows.append({'name':group,'domain':'Soil' if group.startswith('Soil') else 'Climate','n_features_dropped':len(names),'pseudo_R2_full':pseudo,'pseudo_R2_reduced':rp,'delta_pseudo_r2':pseudo-rp,'test_R2_full':scores['R2'],'test_R2_reduced':rm['R2'],'delta_test_r2':scores['R2']-rm['R2'],'delta_test_rmse':rm['RMSE']-scores['RMSE'],'selected_lambda':float(np.array(reduced.lam).ravel()[0])});log('DROP RESULT',rows[-1])
 result=pd.DataFrame(rows).sort_values('delta_pseudo_r2',ascending=False);total=result.delta_pseudo_r2.sum();result['relative_importance_%']=result.delta_pseudo_r2/total*100 if total>0 else np.nan;result.to_csv(T/'Fig7_drop_group_importance.csv',index=False,encoding='utf-8-sig')
 meta=json.loads((A/'data_sources_and_methods.json').read_text(encoding='utf-8'));meta['GAM']={'groups':GROUPS,'variance_threshold':.95,'n_splines':20,'test_size':.2,'random_seed':42,'lambda_search':'pyGAM default gridsearch; common lambda over spline terms','DEM_used_for_prediction':False,'TP':'loaded for common-valid input mask but not assigned to a predictor group, common-valid input mask','PCA_fit':'training set only','drop_importance':'decline in training explained-deviance pseudo-R2; relative share normalized by sum, no clipping','evaluation':'test RTEM_T>0 only','software':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pygam':pygam.__version__}}
 meta['status']='GAM fitted';(A/'data_sources_and_methods.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 log('GAM COMPLETE',scores,result.to_dict('records'))
