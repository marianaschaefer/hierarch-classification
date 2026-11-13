"""Pipeline 1 (amostras manuais): efeito da seleção de variáveis feita FORA da partição de treino.
L = conjunto fixo de 54 variáveis (RFE do caret escolhido uma vez), 100 hold-outs 75/25  [como nos scripts R lidos]
N = nzv + Spearman(0.95) + RFE refeitos DENTRO de cada treino (54 variáveis)            [protocolo sem vazamento]"""
import numpy as np, pandas as pd, time, os
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, cohen_kappa_score
Z = "data/QFGeo-LULC Dataset/QFGeo-LULC Dataset/06_atributos_pipeline1"
full = pd.read_csv(f"{Z}/estatisticas_completas/estatisticas_poligonos_pipeline1.csv")
sel = pd.read_csv(f"{Z}/base_modelo_pronta/base_modelo_pipeline1.csv")
y = full.Classe_Fin.values; Xall = full.drop(columns=["id_", "Classe_Fin"])
f54 = [c for c in sel.columns if c not in ("id_", "Classe_Fin")]
assert len(f54) == 54 and set(f54) <= set(Xall.columns)
print("amostras:", len(y), " atributos brutos:", Xall.shape[1], flush=True)

def nzv_corr_filter(X, cutoff=0.95):
    X = X.loc[:, X.nunique() > 1]
    C = np.abs(spearmanr(X.values, nan_policy="omit")[0]); np.fill_diagonal(C, 0); keep = list(range(X.shape[1]))
    while True:      # algoritmo do caret::findCorrelation (remove a variável de maior correlação média absoluta)
        sub = C[np.ix_(keep, keep)]
        if sub.max() <= cutoff: break
        i, j = np.unravel_index(sub.argmax(), sub.shape)
        drop = i if sub[i].mean() > sub[j].mean() else j
        keep.pop(drop)
    return list(X.columns[keep])
def metr(yt, yp): return dict(f1_macro=f1_score(yt, yp, average="macro"), bal_acc=balanced_accuracy_score(yt, yp), acc=accuracy_score(yt, yp), kappa=cohen_kappa_score(yt, yp))
NREP = 100; OUT = "outputs/p1_leak.csv"; rows = []
for r, (tr, te) in enumerate(StratifiedShuffleSplit(NREP, test_size=0.25, random_state=2026).split(Xall, y)):
    t0 = time.time()
    rf = RandomForestClassifier(300, random_state=r, n_jobs=1).fit(Xall.iloc[tr][f54], y[tr])
    rows.append(dict(rep=r, esquema="L_selecao_fixa_54", **metr(y[te], rf.predict(Xall.iloc[te][f54]))))
    keep = nzv_corr_filter(Xall.iloc[tr])
    rfe = RFE(RandomForestClassifier(100, random_state=r, n_jobs=1), n_features_to_select=54, step=0.1).fit(Xall.iloc[tr][keep], y[tr])
    f = [c for c, s in zip(keep, rfe.support_) if s]
    rf2 = RandomForestClassifier(300, random_state=r, n_jobs=1).fit(Xall.iloc[tr][f], y[tr])
    rows.append(dict(rep=r, esquema="N_selecao_dentro_do_treino", n_pre_rfe=len(keep), overlap_com_54=len(set(f) & set(f54)), **metr(y[te], rf2.predict(Xall.iloc[te][f]))))
    if r % 5 == 4:
        pd.DataFrame(rows).to_csv(OUT, index=False); print(f"rep {r+1}/{NREP}  {time.time()-t0:.0f}s/rep", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False); print("FIM P1", flush=True)
