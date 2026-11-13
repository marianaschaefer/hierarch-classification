"""E1: efeito do filtro de pureza com TESTE COMUM (não filtrado). 4 variantes de treino x folds aleatórios ou espaciais.
uso: python 04_e1_run.py random|spatial"""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
scheme = sys.argv[1]; OUT = f"outputs/e1_pred_{scheme}.csv"
d, feats = load()
folds = make_folds(d, scheme, 10, block_km=10)
buffer_m = 2000 if scheme == "spatial" else 0
pd.DataFrame({"supercells_global": d.supercells_global, "fold": folds}).to_csv(f"outputs/folds_{scheme}.csv", index=False)
CLASSES = sorted(d.classe_nivel2_m.unique()); cidx = {c: i for i, c in enumerate(CLASSES)}
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, usecols=["fold", "variant"]).drop_duplicates()
    done = set(map(tuple, prev.values))

variants = {   # nome: (máscara de treino, coluna de rótulo)
    "R_bruto":        (lambda t: np.ones(len(t), bool),     "classe_nivel2_m"),
    "F_pureza0.6":    (lambda t: t.prop_label_m.values >= 0.6, "classe_nivel2_m"),
    "D_relabel_dom":  (lambda t: np.ones(len(t), bool),     "dom_m"),
    "DF_relabel_dom_filtro0.6": (lambda t: t.prop_dom_m.values >= 0.6, "dom_m"),
}
for k in range(10):
    tr, te = train_idx_with_buffer(d, folds, k, buffer_m)
    T, V = d.loc[tr], d.loc[te]
    for name, (mask, ycol) in variants.items():
        if (k, name) in done: continue
        t0 = time.time(); m = mask(T); Tm = T[m]
        ytr = Tm[ycol].map(cidx).values
        present = np.unique(ytr); remap = {c: i for i, c in enumerate(present)}
        mdl = new_model(n_jobs=1); mdl.fit(Tm[feats], np.vectorize(remap.get)(ytr))
        pr = present[mdl.predict(V[feats])]
        out = pd.DataFrame({"supercells_global": V.supercells_global.values, "fold": k, "variant": name,
                            "y_point": V.classe_nivel2_m.values, "y_dom": V.dom_m.values,
                            "P_label": V.prop_label_m.values, "P_dom": V.prop_dom_m.values,
                            "pred": [CLASSES[i] for i in pr], "n_train": len(Tm)})
        out.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"[{scheme}] fold {k} {name}: n_train={len(Tm)} n_test={len(V)} {time.time()-t0:.0f}s", flush=True)
print("FIM", scheme, flush=True)
