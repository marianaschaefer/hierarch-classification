"""Seleção ANINHADA do limiar de pureza (só com dados de treino) + curva completa de tau no teste externo.
Externo: folds espaciais de 10 km (folds_spatial.csv), buffer 2 km. Interno: StratifiedGroupKFold(3) por blocos de 10 km
dentro do treino, validação NÃO filtrada. Modelo leve (XGB 150 árvores), 216 atributos (mesmos do E2).
Saídas: n300_inner.csv (F1 interno por tau, vs rótulo de ponto e vs classe dominante), n300_outer.csv (predições externas por tau,
polígonos automáticos), n300_manual.csv (predições por tau nos polígonos manuais do fold)."""
import sys, os, re, time, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
from xgboost import XGBClassifier
TAUS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
FOLDS = [int(x) for x in os.environ.get("FOLDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
O = "outputs/"
to7 = lambda c: "Vegetação Rasteira" if c in ("Formação Campestre","Pastagem","Solo Exposto","Cerrado","Agricultura") else c
d, feats = load(); f216 = [c for c in feats if not re.fullmatch(r"A\d\d", c)]
fo = pd.read_csv(O + "folds_spatial.csv"); d = d.merge(fo, on="supercells_global").reset_index(drop=True)
man = pd.read_csv(O + "manual_features_p2names.csv")
CL = sorted(d.classe_nivel2_m.unique()); cidx = {c: i for i, c in enumerate(CL)}
def mdl(): return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", n_jobs=2, random_state=SEED)
def fit(T):
    y = T.classe_nivel2_m.map(cidx).values; pres = np.unique(y); rm = {c: i for i, c in enumerate(pres)}
    m = mdl(); m.fit(T[f216], np.vectorize(rm.get)(y)); return m, pres
def pred(m, pres, X): return np.array(CL)[pres[m.predict(X)]]
def done_folds(name):
    p = O + name
    return set(pd.read_csv(p, usecols=["fold"]).fold.unique()) if os.path.exists(p) else set()
def app(df, name): df.to_csv(O + name, mode="a", header=not os.path.exists(O + name), index=False)
dn = done_folds("n300_manual.csv")
bx = np.floor(d.cx.values/10000).astype(int); by = np.floor(d.cy.values/10000).astype(int)
d["blk"] = pd.factorize(pd.Series(bx.astype(str)+"_"+by.astype(str)))[0]
for k in FOLDS:
    if k in dn: continue
    t0 = time.time()
    te = d[d.fold == k]; mt = man[man.fold == k]
    tree = cKDTree(np.vstack([te[["cx","cy"]].values, mt[["cx","cy"]].values]))
    A = d[d.fold != k]; dist, _ = tree.query(A[["cx","cy"]].values); A = A[dist >= 2000].reset_index(drop=True)
    # interno
    sgk = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
    rows = []
    for j, (itr, iva) in enumerate(sgk.split(A, A.classe_nivel2_m, groups=A.blk)):
        Tr, Va = A.iloc[itr], A.iloc[iva]
        for tau in TAUS:
            Tm = Tr[Tr.prop_label_m >= tau] if tau > 0 else Tr
            m, pres = fit(Tm); p = pred(m, pres, Va[f216])
            rows.append(dict(fold=k, inner=j, tau=tau, n_train=len(Tm),
                             f1_point=f1_score(Va.classe_nivel2_m, p, average="macro"),
                             f1_dom=f1_score(Va.dom_m, p, average="macro")))
    app(pd.DataFrame(rows), "n300_inner.csv")
    # externo
    ro, rm_ = [], []
    for tau in TAUS:
        Tm = A[A.prop_label_m >= tau] if tau > 0 else A
        m, pres = fit(Tm)
        p = pred(m, pres, te[f216])
        ro.append(pd.DataFrame(dict(supercells_global=te.supercells_global.values, fold=k, tau=tau, y_point=te.classe_nivel2_m.values,
                                    y_dom=te.dom_m.values, P_label=te.prop_label_m.values, pred=p, n_train=len(Tm))))
        pm = pred(m, pres, mt[f216])
        rm_.append(pd.DataFrame(dict(id_=mt.id_.values, fold=k, tau=tau, y7=mt.y7.values, pred7=[to7(x) for x in pm], n_train=len(Tm))))
    app(pd.concat(ro), "n300_outer.csv"); app(pd.concat(rm_), "n300_manual.csv")
    print(f"fold {k} ok {time.time()-t0:.0f}s", flush=True)
print("FIM", flush=True)
