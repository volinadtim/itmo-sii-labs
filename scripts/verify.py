"""Verify executed deliverables and independently recompute reported test metrics."""
from pathlib import Path
import hashlib
import json
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
R=Path(__file__).resolve().parents[1]
n=nbformat.read(R/'lab3/Lab3_Danilov_Pivovarov.ipynb',as_version=4)
nbformat.validate(n)
t=json.loads((R/'materials/Lab3.ipynb').read_text())
original=[''.join(c['source']) for c in t['cells'] if c['cell_type']=='markdown']
actual=[c.source for c in n.cells if c.cell_type=='markdown']
assert all(s in actual for s in original), 'Assignment text missing'
code=[c for c in n.cells if c.cell_type=='code' and c.source.strip()]
assert all(c.execution_count is not None for c in code), 'Unexecuted cells'
assert not any(o.output_type=='error' for c in code for o in c.outputs)
assert any('CHECKS PASSED' in o.get('text','') for c in code for o in c.outputs)
result=R/'lab3/results'
summary=json.loads((result/'data_summary.json').read_text())
assert summary['csv_sha256']==hashlib.sha256((R/'lab1/data/portugal_listinigs.csv').read_bytes()).hexdigest()
split=pd.read_csv(result/'split.csv')
assert split.row_index.is_unique
train=set(split.loc[split.split=='train','row_index']); test=set(split.loc[split.split=='test','row_index'])
assert train.isdisjoint(test)
assert len(train)==summary['n_train'] and len(test)==summary['n_test']
reported=pd.read_csv(result/'classification_metrics.csv',index_col=0)
for model in ['Dummy','LogisticRegression','SVC_linear']:
    p=pd.read_csv(result/(model+'_test.csv'))
    assert set(p.row_index)==test and p.row_index.is_unique
    target=split.set_index('row_index').loc[p.row_index,'target'].to_numpy()
    assert np.array_equal(target,p.y_true)
    y,hat,score=p.y_true,p.prediction,p.score
    values=dict(accuracy=accuracy_score(y,hat),balanced_accuracy=balanced_accuracy_score(y,hat),
        precision=precision_score(y,hat,zero_division=0),recall=recall_score(y,hat),
        f1=f1_score(y,hat),roc_auc=roc_auc_score(y,score),AP=average_precision_score(y,score))
    for key,value in values.items(): assert np.isclose(value,reported.loc[model,key],atol=1e-10)
a=pd.read_csv(result/'oneclass_test.csv')
assert set(a.row_index)==test and np.isfinite(a.score).all()
assert np.array_equal(a.score<0,a.outlier)
sensitivity=pd.read_csv(result/'oneclass_sensitivity.csv')
assert np.isclose(a.outlier.mean(),sensitivity.loc[np.isclose(sensitivity.nu,.05),'test_outlier_share'].iloc[0])
for f in ['confusion_matrices.png','roc_pr.png','oneclass.png']:
    assert (result/f).stat().st_size>1000
print(f'PASS: {len(original)} original assignment cells preserved; {len(code)} code cells executed without errors.')
print('PASS: data checksum, disjoint split, all 21 classification metrics recomputed, anomaly scores and plots verified.')

# Assignment 3: verify original slide transcription and all manual artifacts.
Y=np.array([[1,1,0],[0,1,1],[0,1,0],[1,0,0]])
S=np.array([[.7,.6,.5],[.3,.4,.6],[.5,.9,.2],[.4,.5,.1]])
assert np.array_equal(pd.read_csv(result/'slide43_labels.csv',index_col=0).to_numpy(),Y)
assert np.array_equal(pd.read_csv(result/'slide43_scores.csv',index_col=0).to_numpy(),S)
avgs=pd.read_csv(result/'slide43_auc_averages.csv').set_index('method')
for method,value in {'macro':29/36,'weighted':3/4,'micro':5/6,'samples':7/8}.items():
    assert np.isclose(avgs.loc[method,'auc'],value)
    assert round(value,3)==avgs.loc[method,'slide_rounded']
vertices=pd.read_csv(result/'slide43_roc_vertices.csv')
for j in range(3):
    table=vertices[vertices['class']==f'class {j+1}']
    assert np.array_equal(table.iloc[0][['FPR','TPR']].astype(float),[0.,0.])
    assert np.array_equal(table.iloc[-1][['FPR','TPR']].astype(float),[1.,1.])
    for row in table.itertuples():
        pred=S[:,j]>=row.threshold
        tp=int(np.sum(pred & (Y[:,j]==1))); fp=int(np.sum(pred & (Y[:,j]==0)))
        assert (row.TP,row.FP)==(tp,fp)
        assert np.isclose(row.FPR,fp/(4-Y[:,j].sum()))
        assert np.isclose(row.TPR,tp/Y[:,j].sum())
    points=table[['FPR','TPR']].to_numpy()
    area=sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(points,points[1:]))
    assert np.isclose(area,[3/4,2/3,1][j])
thresholds=pd.read_csv(result/'slide43_thresholds.csv')
assert np.array_equal(thresholds[['TP','FP','FN','TN']].to_numpy(),[[3,1,0,0],[2,1,1,0],[1,0,2,1]])
predictions=pd.read_csv(result/'LogisticRegression_test.csv')
estate=pd.read_csv(result/'estate_thresholds_manual.csv')
for table,labels,scores in [(thresholds,Y[:,1],S[:,1]),(estate,predictions.y_true.to_numpy(),predictions.score.to_numpy())]:
    for row in table.itertuples():
        pred=scores>=row.threshold
        tp=int(sum(pred & (labels==1)));fp=int(sum(pred & (labels==0)))
        fn=int(sum(~pred & (labels==1)));tn=int(sum(~pred & (labels==0)))
        assert (tp,fp,fn,tn)==(row.TP,row.FP,row.FN,row.TN)
        assert np.allclose([row.accuracy,row.precision,row.recall],[(tp+tn)/len(labels),tp/(tp+fp),tp/(tp+fn)])
for col in ['accuracy','precision','recall']: assert estate[col].nunique()==3
for name in ['slide43_class_roc.png','slide43_micro_samples_roc.png','slide43_precision_recall_sets.png','manual_pr_sketches.png']:
    assert (result/name).stat().st_size>1000
assert (R/'materials/lecture4_slide43.png').exists()
assert not (R/'lab3/Lab3_Danilov.ipynb').exists()
print('PASS: slide 43 transcription, 3 manual ROC areas, all 4 AUC averages, both threshold cases and 4 figures verified.')

# Check manual ROC handling of ties and extreme rankings independently of the lecture example.
import ast
from fractions import Fraction
manual_cell=next(c.source for c in n.cells if c.cell_type=='code' and 'def manual_pair_auc' in c.source)
definitions=[node for node in ast.parse(manual_cell).body if isinstance(node,ast.FunctionDef) and node.name in {'manual_pair_auc','manual_roc'}]
scope={'np':np,'Fraction':Fraction}
exec(compile(ast.Module(body=definitions,type_ignores=[]),'<manual-auc-check>','exec'),scope)
for scores,expected in [([.8,.2],Fraction(1)),([.2,.8],Fraction(0)),([.5,.5],Fraction(1,2))]:
    assert scope['manual_pair_auc']([1,0],scores)[0]==expected
    assert scope['manual_roc']([1,0],scores)[1]==expected
for fn in ['manual_pair_auc','manual_roc']:
    try: scope[fn]([1,1],[.2,.8])
    except ValueError: pass
    else: raise AssertionError('Undefined AUC must be rejected')
print('PASS: manual AUC perfect/reversed/tied rankings and undefined-class cases.')
