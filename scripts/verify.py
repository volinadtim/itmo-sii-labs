"""Verify executed deliverables and independently recompute reported test metrics."""
from pathlib import Path
import hashlib
import json
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
R=Path(__file__).resolve().parents[1]
n=nbformat.read(R/'lab3/Lab3_Danilov.ipynb',as_version=4)
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
