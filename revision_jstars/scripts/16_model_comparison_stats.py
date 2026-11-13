import pandas as pd, numpy as np
from scipy import stats
O="outputs/"; rng=np.random.default_rng(2026)
P=pd.read_csv(O+"st_pred.csv"); F=pd.read_csv(O+"st_folds.csv")[["supercells_global","blk"]]
P=P.merge(F,on="supercells_global"); CL=sorted(P.y.unique()); ci={c:i for i,c in enumerate(CL)}; C=len(CL)
MODELS=["RF_flat","XGB_flat","XGB_hier","ABL_embeddings","ABL_espectrais"]
def cm(y,p): return np.bincount(np.array([ci[a] for a in y])*C+np.array([ci.get(b,0) for b in p]),minlength=C*C).reshape(C,C)
def mets(M):
    tp=np.diag(M).astype(float); rec=tp/np.maximum(M.sum(1),1); prec=tp/np.maximum(M.sum(0),1); f1=np.where(prec+rec>0,2*prec*rec/np.maximum(prec+rec,1e-12),0)
    n=M.sum(); po=tp.sum()/n; pe=(M.sum(0)*M.sum(1)).sum()/n**2
    sup=M.sum(1)>0
    return dict(f1_macro=f1[sup].mean(),bal_acc=rec[sup].mean(),acc=po,kappa=(po-pe)/(1-pe))
print("n polígonos filtrados:",P[P.model=="XGB_flat"].shape[0],"| classes:",C)
# por fold
FM={m:pd.DataFrame([mets(cm(g.y,g.pred)) for _,g in P[P.model==m].groupby("fold")]) for m in MODELS}
# pooled + blocos
blocks=np.sort(P.blk.unique()); bidx={b:i for i,b in enumerate(blocks)}
BCM={}
for m in MODELS:
    g=P[P.model==m]; arr=np.zeros((len(blocks),C,C),int)
    for b,gb in g.groupby("blk"): arr[bidx[b]]=cm(gb.y,gb.pred)
    BCM[m]=arr
B=2000; W=np.stack([np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks)) for _ in range(B)])
def boot(m): 
    S=np.tensordot(W,BCM[m],axes=(1,0)); return pd.DataFrame([mets(S[i]) for i in range(B)])
BT={m:boot(m) for m in MODELS}
pool={m:mets(BCM[m].sum(0)) for m in MODELS}
print("\n=== Métricas agregadas (predições fora-da-amostra) com IC95%% por bootstrap de blocos de 10 km (%d blocos) ==="%len(blocks))
for m in MODELS:
    print(f"{m:15s}"+"  ".join(f"{k} {pool[m][k]:.3f} [{BT[m][k].quantile(.025):.3f},{BT[m][k].quantile(.975):.3f}]" for k in ["f1_macro","bal_acc","acc","kappa"]))
def holm(ps):
    o=np.argsort(ps); adj=np.empty(len(ps)); run=0
    for r,i in enumerate(o): run=max(run,(len(ps)-r)*ps[i]); adj[i]=min(1,run)
    return adj
main=["RF_flat","XGB_flat","XGB_hier"]; K=10; ntr_te=1/9
print("\n=== Friedman entre os 3 modelos (por fold; descritivo, folds não são independentes) ===")
for k in ["f1_macro","bal_acc","acc","kappa"]:
    st,p=stats.friedmanchisquare(*[FM[m][k].values for m in main]); print(f"{k:9s} chi2={st:.2f} p={p:.4f}")
print("\n=== Diferenças pareadas (A − B): média por fold, IC95% de blocos (métrica agregada), t corrigido de Nadeau–Bengio e Wilcoxon, ambos com Holm ===")
pairs=[("XGB_flat","RF_flat"),("XGB_hier","RF_flat"),("XGB_hier","XGB_flat")]
for k in ["f1_macro","bal_acc","acc","kappa"]:
    rows=[]
    for a,b in pairs:
        dlt=FM[a][k].values-FM[b][k].values; t=dlt.mean()/np.sqrt((1/K+ntr_te)*dlt.var(ddof=1)); pnb=2*stats.t.sf(abs(t),K-1)
        try: pw=stats.wilcoxon(dlt).pvalue
        except Exception: pw=1.0
        bd=BT[a][k]-BT[b][k]; rows.append(dict(par=f"{a} − {b}",dif_agregada=pool[a][k]-pool[b][k],ic_lo=bd.quantile(.025),ic_hi=bd.quantile(.975),p_NB=pnb,p_wilcoxon=pw))
    r=pd.DataFrame(rows); r["p_NB_holm"]=holm(r.p_NB.values); r["p_wilc_holm"]=holm(r.p_wilcoxon.values); print(f"\n[{k}]"); print(r.round(4).to_string(index=False))
print("\n=== Equivalência flat vs hierárquico (XGB), macro-F1, margem ±0,02: IC90% de blocos da diferença ===")
bd=BT["XGB_hier"]["f1_macro"]-BT["XGB_flat"]["f1_macro"]; lo,hi=bd.quantile(.05),bd.quantile(.95)
print(f"dif {pool['XGB_hier']['f1_macro']-pool['XGB_flat']['f1_macro']:+.3f}  IC90% [{lo:+.3f},{hi:+.3f}]  dentro de ±0,02? {'SIM (equivalentes nessa margem)' if (lo>-0.02 and hi<0.02) else 'NÃO (não se pode afirmar equivalência)'}")
print("\n=== Ablação (XGB hierárquico, base filtrada): macro-F1 e diferenças pareadas (IC95% de blocos) ===")
for a,b in [("XGB_hier","ABL_espectrais"),("XGB_hier","ABL_embeddings"),("ABL_embeddings","ABL_espectrais")]:
    for k in ["f1_macro","bal_acc"]:
        bd=BT[a][k]-BT[b][k]; print(f"{a} − {b} [{k}]: {pool[a][k]-pool[b][k]:+.3f} [{bd.quantile(.025):+.3f},{bd.quantile(.975):+.3f}]")
pd.DataFrame({m:pool[m] for m in MODELS}).T.to_csv(O+"st_pooled.csv")
