import pandas as pd, numpy as np, glob
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support
d=pd.concat([pd.read_csv(f) for f in glob.glob('outputs/e2_pred_*.csv')]).drop_duplicates(['id_','fold','variant'])
print(d.groupby('variant').id_.nunique())
fo=pd.read_csv('outputs/manual_features_p2names.csv')
rng=np.random.default_rng(0)
V=['R_bruto','F_pureza0.6','D_relabel_dom','DF_relabel_dom_filtro0.6','M_manual']
P={v:d[d.variant==v].set_index('id_') for v in V}
ids=sorted(set.intersection(*[set(P[v].index) for v in V])); print('n comum',len(ids))
y=P['R_bruto'].loc[ids,'y7'].values
# bloco: usa fold como cluster (folds espaciais de 10 km)
fold=P['R_bruto'].loc[ids,'fold'].values
def m(v,idx=None):
    p=P[v].loc[ids,'pred7'].values
    if idx is not None: return f1_score(y[idx],p[idx],average='macro'),balanced_accuracy_score(y[idx],p[idx])
    return f1_score(y,p,average='macro'),balanced_accuracy_score(y,p)
res={v:m(v) for v in V}
B=1000; boots={v:[] for v in V}; n=len(ids); byf={k:np.where(fold==k)[0] for k in np.unique(fold)}
for _ in range(B):
    ks=rng.choice(list(byf),len(byf)); idx=np.concatenate([byf[k] for k in ks])
    for v in V: boots[v].append(m(v,idx))
print('\nmacro-F1 / bal.acc (IC95% bootstrap por fold-bloco)')
for v in V:
    b=np.array(boots[v]); print(f'{v:26s} F1 {res[v][0]:.3f} [{np.percentile(b[:,0],2.5):.3f},{np.percentile(b[:,0],97.5):.3f}]  BA {res[v][1]:.3f} [{np.percentile(b[:,1],2.5):.3f},{np.percentile(b[:,1],97.5):.3f}]')
print('\ndiferenças pareadas macro-F1 vs R_bruto')
for v in V[1:]:
    b=np.array(boots[v])[:,0]-np.array(boots['R_bruto'])[:,0]; print(f'{v:26s} {res[v][0]-res["R_bruto"][0]:+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]')
print('\nM_manual vs R_bruto, F, D')
for v in ['R_bruto','F_pureza0.6','D_relabel_dom']:
    b=np.array(boots['M_manual'])[:,0]-np.array(boots[v])[:,0]; print(f'M - {v:15s} {res["M_manual"][0]-res[v][0]:+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]')
cl=sorted(set(y)); rows=[]
for v in V:
    p,r,f,s=precision_recall_fscore_support(y,P[v].loc[ids,'pred7'].values,labels=cl,zero_division=0)
    rows+= [dict(variant=v,classe=c,prec=a,rec=b,f1=e,n=int(g)) for c,a,b,e,g in zip(cl,p,r,f,s)]
t=pd.DataFrame(rows); t.to_csv('outputs/e2_perclass.csv',index=False)
print('\nF1 por classe'); print(t.pivot(index='classe',columns='variant',values='f1').round(3)[V]); print(t.groupby('classe').n.first())
