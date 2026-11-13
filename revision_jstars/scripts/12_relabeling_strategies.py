"""Variantes HÍBRIDAS (mesmos folds espaciais, buffer, modelo leve e 216 atributos do n12):
H1 = rótulo de ponto se P_label>=0.6, senão classe dominante; usa TODOS os polígonos.
H2 = H1 + descarta polígonos com P_dominante < 0.5 (nem a dominante é confiável)."""
import sys, os, re, time, numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
from xgboost import XGBClassifier
O="outputs/"; FOLDS=[int(x) for x in os.environ.get("FOLDS","0,1,2,3,4,5,6,7,8,9").split(",")]
to7=lambda c: "Vegetação Rasteira" if c in ("Formação Campestre","Pastagem","Solo Exposto","Cerrado","Agricultura") else c
d,feats=load(); f216=[c for c in feats if not re.fullmatch(r"A\d\d",c)]
d=d.merge(pd.read_csv(O+"folds_spatial.csv"),on="supercells_global").reset_index(drop=True)
man=pd.read_csv(O+"manual_features_p2names.csv")
CL=sorted(d.classe_nivel2_m.unique()); cidx={c:i for i,c in enumerate(CL)}
d["y_h"]=d.dom_m
V={"D_relabel_dom":lambda A:A,"DF_relabel_filtroDom0.5":lambda A:A[A.prop_dom_m>=0.5]}
def mdl(): return XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,tree_method="hist",n_jobs=2,random_state=SEED)
def app(df,n): df.to_csv(O+n,mode="a",header=not os.path.exists(O+n),index=False)
done=set(pd.read_csv(O+"r300_manual.csv").fold.unique()) if os.path.exists(O+"r300_manual.csv") else set()
for k in FOLDS:
    if k in done: continue
    t0=time.time(); te=d[d.fold==k]; mt=man[man.fold==k]
    tree=cKDTree(np.vstack([te[["cx","cy"]].values,mt[["cx","cy"]].values]))
    A=d[d.fold!=k]; dist,_=tree.query(A[["cx","cy"]].values); A=A[dist>=2000]
    ro,rm=[],[]
    for name,fn in V.items():
        T=fn(A); y=T.y_h.map(cidx).values; pres=np.unique(y); rmap={c:i for i,c in enumerate(pres)}
        m=mdl(); m.fit(T[f216],np.vectorize(rmap.get)(y)); P=lambda X: np.array(CL)[pres[m.predict(X)]]
        ro.append(pd.DataFrame(dict(supercells_global=te.supercells_global.values,fold=k,variant=name,y_point=te.classe_nivel2_m.values,y_dom=te.dom_m.values,P_label=te.prop_label_m.values,pred=P(te[f216]),n_train=len(T))))
        pm=P(mt[f216]); rm.append(pd.DataFrame(dict(id_=mt.id_.values,fold=k,variant=name,y7=mt.y7.values,pred7=[to7(x) for x in pm],n_train=len(T))))
    app(pd.concat(ro),"r300_outer.csv"); app(pd.concat(rm),"r300_manual.csv"); print(f"fold {k} {time.time()-t0:.0f}s",flush=True)
print("FIM")
