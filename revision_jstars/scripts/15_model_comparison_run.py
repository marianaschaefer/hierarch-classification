"""Rodada ÚNICA para estatística: base filtrada (P_label>=0.6, Cerrado fundido), folds espaciais IDÊNTICOS para todos os modelos,
buffer de 2 km. Modelos: RF_flat, XGB_flat, XGB_hier (280 atributos) + ablação da hierárquica (só embeddings, só espectrais).
Hiperparâmetros iguais aos de x18/x19 (XGB 300 árvores, depth 6, lr 0.05, subsample/colsample 0.8; RF 300, balanced)."""
import sys, os, re, time, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
from xgboost import XGBClassifier
O="outputs/"; OUT=O+"st_pred.csv"
FOLDS=[int(x) for x in os.environ.get("FOLDS","0,1,2,3,4,5,6,7,8,9").split(",")]
d,feats=load(); emb=[c for c in feats if re.fullmatch(r"A\d\d",c)]; spec=[c for c in feats if c not in emb]
f=d[d.prop_label_m>=0.6].reset_index(drop=True); print("filtrados:",len(f),"| espectrais",len(spec),"embeddings",len(emb),flush=True)
f["y"]=f.classe_nivel2_m
R={"Formação Campestre","Pastagem","Solo Exposto","Agricultura","Cerrado"}; f["y1"]=np.where(f.y.isin(R),"Vegetação Rasteira",f.y)
bx=np.floor(f.cx/10000).astype(int); by=np.floor(f.cy/10000).astype(int); f["blk"]=pd.factorize(bx.astype(str)+"_"+by.astype(str))[0]
fold=np.empty(len(f),int)
for k,(_,te) in enumerate(StratifiedGroupKFold(10,shuffle=True,random_state=SEED).split(f,f.y,groups=f.blk)): fold[te]=k
f["fold"]=fold; f[["supercells_global","y","y1","blk","fold","cx","cy"]].to_csv(O+"st_folds.csv",index=False)
print(f.groupby("fold").size().to_dict(),flush=True)
def xgb(): return XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,tree_method="hist",n_jobs=2,random_state=SEED)
def fitpred(model,Xtr,ytr,Xte):
    cl,yi=np.unique(ytr,return_inverse=True); model.fit(Xtr,yi); return cl[model.predict(Xte)]
def hier(Tr,Te,cols):
    p1=fitpred(xgb(),Tr[cols],Tr.y1.values,Te[cols]); out=p1.astype(object).copy()
    vr=Tr[Tr.y1=="Vegetação Rasteira"]; m=p1=="Vegetação Rasteira"
    if m.any(): out[m]=fitpred(xgb(),vr[cols],vr.y.values,Te[cols][m])
    return out
done=set(pd.read_csv(OUT,usecols=["fold"]).fold.unique()) if os.path.exists(OUT) else set()
for k in FOLDS:
    if k in done: continue
    t0=time.time(); te=f[f.fold==k]; tree=cKDTree(te[["cx","cy"]].values)
    tr=f[f.fold!=k]; dist,_=tree.query(tr[["cx","cy"]].values); tr=tr[dist>=2000]
    allc=spec+emb; res={}
    res["RF_flat"]=fitpred(RandomForestClassifier(n_estimators=300,n_jobs=2,random_state=SEED,class_weight="balanced"),tr[allc],tr.y.values,te[allc])
    res["XGB_flat"]=fitpred(xgb(),tr[allc],tr.y.values,te[allc])
    res["XGB_hier"]=hier(tr,te,allc)
    res["ABL_embeddings"]=hier(tr,te,emb)
    res["ABL_espectrais"]=hier(tr,te,spec)
    rows=[pd.DataFrame(dict(supercells_global=te.supercells_global.values,fold=k,model=m,y=te.y.values,pred=p,n_train=len(tr))) for m,p in res.items()]
    pd.concat(rows).to_csv(OUT,mode="a",header=not os.path.exists(OUT),index=False); print(f"fold {k} {time.time()-t0:.0f}s",flush=True)
print("FIM",flush=True)
