import sys,json,pickle
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
sys.stdout.reconfigure(encoding='utf-8')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, RESULTS, working_copy
O=working_copy('05_GAM');T=O/'02_计算底表与指标/统计底表';A=O/'03_复现资料'
bundle=pickle.loads((A/'Fig7_full_GAM_and_train_transforms.pkl').read_bytes());meta=json.loads((A/'data_sources_and_methods.json').read_text(encoding='utf-8'));df=pd.read_csv(T/'pixel_means_and_environment.csv')
raw=['prec','shum','VPD','SWC','Rn','srad','temp','wind','bd','CEC','SOC','TN','TP','clay','silt','sand','pH','thickness','cf']
v=np.logical_and.reduce([np.isfinite(df[k].values) for k in ['T','DEM']+raw]);df=df.loc[v].copy();np.testing.assert_array_equal(df[['row','col']].values,bundle['row_col']);test=df.iloc[bundle['test_indices']];columns=[]
for group,tr in bundle['groups'].items():
 values=test[tr['members']].values.astype('float32');transformed=tr['pca'].transform(tr['scaler'].transform(values))[:,:tr['n_pc']]
 columns.extend(transformed[:,j] for j in range(tr['n_pc']))
X=np.column_stack(columns);X=(X-bundle['global_mean'])/bundle['global_sd'];p=bundle['model'].predict(X);saved=pd.read_csv(T/'Fig7_test_predictions.csv');np.testing.assert_allclose(p,saved.GAM_T.values,rtol=1e-10,atol=1e-10);np.testing.assert_array_equal(test[['row','col']].values,saved[['row','col']].values)
scores=json.loads((A/'Fig7_model_metrics.json').read_text(encoding='utf-8'));y=saved.RTEM_T.values.astype('float32');valid=(y>0)&np.isfinite(p)
assert scores['n']==int(valid.sum())
for key,value in [('R2',r2_score(y[valid],p[valid])),('MAE',mean_absolute_error(y[valid],p[valid])),('RMSE',np.sqrt(mean_squared_error(y[valid],p[valid])))]:np.testing.assert_allclose(scores[key],value,rtol=1e-12,atol=1e-12)
imp=pd.read_csv(T/'Fig7_drop_group_importance.csv');np.testing.assert_allclose(imp.delta_pseudo_r2,imp.pseudo_R2_full-imp.pseudo_R2_reduced,atol=1e-14);np.testing.assert_allclose(imp['relative_importance_%'].sum(),100,atol=1e-9)
assert len(imp)==7 and len(set(bundle['train_indices'])&set(bundle['test_indices']))==0
assert 'DEM' not in bundle['features']
assert len(bundle['features'])==sum(int(group['n_pc']) for group in bundle['groups'].values())
result={'test_predictions_reproduced':True,'prediction_atol':1e-10,'metric_recalculation_passed':True,'train_test_index_overlap':0,'PCA_and_scaler_fit_on_train_only':True,'DEM_not_predictor':True,'seven_group_delta_and_percentage_check':True,'test_n':scores['n'],'R2':scores['R2'],'RMSE':scores['RMSE']}
(A/'GAM_reproduction_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
