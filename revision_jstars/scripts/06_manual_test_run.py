"""E2: teste INDEPENDENTE = polígonos manuais (7 classes, interpretação visual). Treino automático (R/F/D/DF) vs treino manual (M),
216 atributos, folds espaciais de 10 km, buffer de 2 km entre teste e treino."""
import sys, os, time, numpy as np, pandas as pd, re
from scipy.spatial import cKDTree
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
d, feats = load(); f216 = [c for c in feats if not re.fullmatch(r"A\d\d", c)]
fo = pd.read_csv("outputs/folds_spatial.csv"); d = d.merge(fo, on="supercells_global")
man = pd.read_csv("outputs/manual_features_p2names.csv")
import glob
FOLDS = [int(x) for x in os.environ.get("FOLDS","0,1,2,3,4,5,6,7,8,9").split(",")]
OUT = f"outputs/e2_pred_{os.environ.get('TAG','a')}.csv"
PREV = glob.glob("outputs/e2_pred*.csv")
to7 = lambda c: "Vegetação Rasteira" if c in ("Formação Campestre","Pastagem","Solo Exposto","Cerrado","Agricultura") else c
done = set()
for f in PREV: done |= set(map(tuple, pd.read_csv(f, usecols=["fold","variant"]).drop_duplicates().values))
auto_var = {
  "R_bruto": (lambda t: np.ones(len(t), bool), "classe_nivel2_m"),
  "F_pureza0.6": (lambda t: t.prop_label_m.values >= 0.6, "classe_nivel2_m"),
  "D_relabel_dom": (lambda t: np.ones(len(t), bool), "dom_m"),
  "DF_relabel_dom_filtro0.6": (lambda t: t.prop_dom_m.values >= 0.6, "dom_m")}
def fit_predict(Xtr, ytr, Xte):
    cl = np.unique(ytr); mp = {c: i for i, c in enumerate(cl)}
    m = new_model(n_jobs=int(os.environ.get("NJ", "2"))); m.fit(Xtr, np.vectorize(mp.get)(ytr)); return cl[m.predict(Xte)]
for k in FOLDS:
    te = man[man.fold == k]
    if te.empty: continue
    tree = cKDTree(te[["cx","cy"]].values)
    # automáticos: fora do fold k (blocos) e a >= 2 km de qualquer teste manual
    A = d[d.fold != k]; dist, _ = tree.query(A[["cx","cy"]].values); A = A[dist >= 2000]
    Mtr = man[man.fold != k]; dist, _ = tree.query(Mtr[["cx","cy"]].values); Mtr = Mtr[dist >= 2000]
    for name, (mask, ycol) in auto_var.items():
        if (k, name) in done: continue
        t0 = time.time(); Tm = A[mask(A)]
        p = fit_predict(Tm[f216], Tm[ycol].values, te[f216])
        pd.DataFrame({"id_": te.id_.values, "fold": k, "variant": name, "y7": te.y7.values, "pred7": [to7(x) for x in p], "n_train": len(Tm)}
                     ).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"fold {k} {name}: n_train={len(Tm)} n_test={len(te)} {time.time()-t0:.0f}s", flush=True)
    if (k, "M_manual") not in done:
        p = fit_predict(Mtr[f216], Mtr.y7.values, te[f216])
        pd.DataFrame({"id_": te.id_.values, "fold": k, "variant": "M_manual", "y7": te.y7.values, "pred7": p, "n_train": len(Mtr)}
                     ).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"fold {k} M_manual: n_train={len(Mtr)}", flush=True)
print("FIM E2", flush=True)
