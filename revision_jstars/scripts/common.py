"""Funções comuns: dados, partições (aleatória e espacial com buffer), modelo e métricas."""
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, cohen_kappa_score
from xgboost import XGBClassifier

UP = "data"
FEATS_CSV = f"{UP}/QFGeo-LULC Dataset/QFGeo-LULC Dataset/05_atributos_pipeline2/dataset_FINAL_COMPLETO_global_v1.csv"
GPKG = f"{UP}/poligonos_finais_global_v1.gpkg"
PUR = "outputs/pureza_v2.csv"
SEED = 42
ID_COLS = ["supercells_global", "classe_nivel1", "classe_nivel2", "sc_id"]

def load():
    X = pd.read_csv(FEATS_CSV)
    P = pd.read_csv(PUR, usecols=["supercells_global","classe_nivel2_m","prop_label_m","dom_m","prop_dom_m","n_poly_px"])
    G = gpd.read_file(GPKG)[["supercells_global","geometry"]]
    c = G.geometry.centroid
    G = pd.DataFrame({"supercells_global": G.supercells_global, "cx": c.x.values, "cy": c.y.values})
    d = X.merge(P, on="supercells_global").merge(G, on="supercells_global")
    feats = [c for c in X.columns if c not in ID_COLS]
    assert len(feats) == 280, len(feats)
    return d.reset_index(drop=True), feats

def make_folds(d, kind, n_splits=10, block_km=10, seed=SEED):
    """Retorna vetor de fold por polígono. kind='random' (estratificado) ou 'spatial' (blocos)."""
    y = d.classe_nivel2_m.values
    if kind == "random":
        f = np.empty(len(d), int)
        for k, (_, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(d, y)): f[te] = k
        return f
    bx = np.floor(d.cx.values / (block_km*1000)).astype(int); by = np.floor(d.cy.values / (block_km*1000)).astype(int)
    blocks = pd.factorize(pd.Series(bx.astype(str) + "_" + by.astype(str)))[0]
    f = np.empty(len(d), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(d, y, groups=blocks)): f[te] = k
    return f

def train_idx_with_buffer(d, folds, k, buffer_m=0):
    te = np.where(folds == k)[0]; tr = np.where(folds != k)[0]
    if buffer_m > 0:
        tree = cKDTree(d.loc[te, ["cx","cy"]].values)
        dist, _ = tree.query(d.loc[tr, ["cx","cy"]].values)
        tr = tr[dist >= buffer_m]
    return tr, te

def new_model(n_estimators=300, n_jobs=2):
    return XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", n_jobs=n_jobs, random_state=SEED)

def metrics(y, p):
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                f1_macro=f1_score(y, p, average="macro"), kappa=cohen_kappa_score(y, p))
