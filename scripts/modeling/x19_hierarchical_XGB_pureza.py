import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    cohen_kappa_score
)

# =========================
# CONFIG
# =========================

DATA_PATH = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1_pureza_0.6.csv"

RANDOM_STATE = 42
N_SPLITS = 10
MIN_SAMPLES = 50

# =========================
# LOAD
# =========================

df = pd.read_csv(DATA_PATH)

print("\n=== BASE ORIGINAL FILTRADA ===")
print("Shape:", df.shape)

print("\nClasses antes do filtro:")
print(df["classe_nivel2"].value_counts())


# =========================
# AGRUPAMENTO EXPERIMENTAL
# =========================

df["classe_nivel2"] = df["classe_nivel2"].replace({
    "Cerrado": "Formação Campestre"
})

print("\nApós agrupar Cerrado → Formação Campestre:")
print(df["classe_nivel2"].value_counts())


# =========================
# FEATURES
# =========================

cols_drop = [
    "classe_nivel1",
    "classe_nivel2",
    "supercells_global",
    "sc_id",
    "prop_label",
    "classe_dominante",
    "prop_dominante"
]

X = df.drop(columns=[c for c in cols_drop if c in df.columns])
y = df["classe_nivel2"]

print("\nFeatures:", X.shape[1])

# =========================
# NIVEL 1 (macro)
# =========================

def map_nivel1(classe):
    if classe in ["Agricultura", "Pastagem", "Formação Campestre", "Cerrado"]:
        return "Vegetação Rasteira"
    else:
        return classe

y_n1 = y.apply(map_nivel1)

print("\nClasses nível 1:")
print(y_n1.value_counts())

# =========================
# MODELOS
# =========================

model_n1 = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

model_n2 = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# =========================
# CV PIPELINE
# =========================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

y_true_all = []
y_pred_all = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_n1)):

    print(f"\n===== FOLD {fold+1}/{N_SPLITS} =====")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_n1, y_test_n1 = y_n1.iloc[train_idx], y_n1.iloc[test_idx]
    y_train_full, y_test_full = y.iloc[train_idx], y.iloc[test_idx]

    # =========================
    # NÍVEL 1
    # =========================

    # encoding local do nível 1
    y_train_n1_enc, classes_n1 = pd.factorize(y_train_n1)
    n1_inv = {i: c for i, c in enumerate(classes_n1)}

    model_n1.fit(X_train, y_train_n1_enc)

    pred_n1_enc = model_n1.predict(X_test)
    pred_n1 = pd.Series([n1_inv[i] for i in pred_n1_enc], index=X_test.index)

    # =========================
    # NÍVEL 2 (VR)
    # =========================

    mask_train_vr = y_train_n1 == "Vegetação Rasteira"
    mask_test_vr = pred_n1 == "Vegetação Rasteira"

    X_train_vr = X_train[mask_train_vr]
    y_train_vr = y_train_full[mask_train_vr]

    X_test_vr = X_test[mask_test_vr]

    # encoding local
    y_train_vr_enc, classes_vr = pd.factorize(y_train_vr)

    model_n2.fit(X_train_vr, y_train_vr_enc)

    pred_final = []

    idx_vr = 0

    for i, p in enumerate(pred_n1):

        if p != "Vegetação Rasteira":
            pred_final.append(p)
        else:
            pred_vr_enc = model_n2.predict(X_test_vr.iloc[[idx_vr]])[0]
            pred_vr = classes_vr[pred_vr_enc]
            pred_final.append(pred_vr)
            idx_vr += 1

    y_true_all.extend(y_test_full)
    y_pred_all.extend(pred_final)

# =========================
# MÉTRICAS
# =========================

acc = accuracy_score(y_true_all, y_pred_all)
f1 = f1_score(y_true_all, y_pred_all, average="macro")
bacc = balanced_accuracy_score(y_true_all, y_pred_all)
kappa = cohen_kappa_score(y_true_all, y_pred_all)

print("\n=== RESULTADO FINAL HIERÁRQUICO (PUREZA) ===")
print(f"Accuracy:          {acc:.4f}")
print(f"Balanced accuracy: {bacc:.4f}")
print(f"F1 macro:          {f1:.4f}")
print(f"Kappa:             {kappa:.4f}")

print("\nRelatório:")
print(classification_report(y_true_all, y_pred_all))