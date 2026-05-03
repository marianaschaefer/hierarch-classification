import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

# =========================================================
# CONFIG
# =========================================================

path = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1.csv"
out_dir = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed"

RANDOM_STATE = 42
N_SPLITS = 10

# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(path)

cols_drop = ["classe_nivel1", "classe_nivel2", "supercells_global", "sc_id"]
X = df.drop(columns=[c for c in cols_drop if c in df.columns])

y_n1 = df["classe_nivel1"].copy()
y_final = df["classe_nivel2"].copy()

print("Shape:", df.shape)
print("Features:", X.shape[1])
print("\nClasse nível 1:")
print(y_n1.value_counts())
print("\nClasse final:")
print(y_final.value_counts())

# =========================================================
# STORAGE
# =========================================================

pred_final = pd.Series(index=df.index, dtype="object")
pred_n1_all = pd.Series(index=df.index, dtype="object")
fold_id = pd.Series(index=df.index, dtype="int")

# =========================================================
# CV HIERÁRQUICO REAL
# =========================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_final), start=1):
    print(f"\n===== FOLD {fold}/{N_SPLITS} =====")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_n1_train = y_n1.iloc[train_idx]
    y_final_train = y_final.iloc[train_idx]

    # -------------------------
    # Modelo nível 1
    # -------------------------
    model_n1 = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    model_n1.fit(X_train, y_n1_train)
    pred_n1 = model_n1.predict(X_test)

    pred_n1_all.iloc[test_idx] = pred_n1
    fold_id.iloc[test_idx] = fold

    # -------------------------
    # Modelo nível 2
    # Treinado apenas nas amostras de Vegetação Rasteira DO TREINO
    # -------------------------
    mask_train_vr = y_n1_train == "Vegetação Rasteira"

    X_train_vr = X_train.loc[mask_train_vr]
    y_train_vr = y_final_train.loc[mask_train_vr]

    model_n2 = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    model_n2.fit(X_train_vr, y_train_vr)

    # -------------------------
    # Composição top-down no teste
    # -------------------------
    pred_fold_final = pd.Series(pred_n1, index=X_test.index, dtype="object")

    # Se nível 1 previu Vegetação Rasteira, chamar nível 2
    idx_pred_vr = X_test.index[pred_n1 == "Vegetação Rasteira"]

    if len(idx_pred_vr) > 0:
        pred_n2 = model_n2.predict(X_test.loc[idx_pred_vr])
        pred_fold_final.loc[idx_pred_vr] = pred_n2

    # Para classes nível 1 que já são classes finais, manter o próprio nome
    # Ex: Mata -> Mata, Eucalipto -> Eucalipto etc.
    pred_final.iloc[test_idx] = pred_fold_final

    print("Teste:", len(test_idx))
    print("Preditos como Vegetação Rasteira no N1:", len(idx_pred_vr))

# =========================================================
# AVALIAÇÃO FINAL
# =========================================================

y_true = y_final
y_pred = pred_final

print("\n=== RESULTADO FINAL HIERÁRQUICO RF - CV PIPELINE ===")
print("F1 macro:", f1_score(y_true, y_pred, average="macro"))
print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))
print("\nRelatório:")
print(classification_report(y_true, y_pred, zero_division=0))

# =========================================================
# SALVAR
# =========================================================

report = pd.DataFrame(
    classification_report(y_true, y_pred, output_dict=True, zero_division=0)
).transpose()

report.to_csv(f"{out_dir}/exp04_hierRF_cv_pipeline_report_final.csv")

labels = sorted(y_true.unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)

pd.DataFrame(
    cm,
    index=labels,
    columns=labels
).to_csv(f"{out_dir}/exp04_hierRF_cv_pipeline_matriz_confusao_final.csv")

df_pred = df[["supercells_global", "classe_nivel1", "classe_nivel2"]].copy()
df_pred["fold"] = fold_id
df_pred["pred_nivel1"] = pred_n1_all
df_pred["pred_final_hier"] = pred_final

df_pred.to_csv(f"{out_dir}/exp04_hierRF_cv_pipeline_predicoes.csv", index=False)

print("\nArquivos salvos:")
print(f"{out_dir}/exp04_hierRF_cv_pipeline_report_final.csv")
print(f"{out_dir}/exp04_hierRF_cv_pipeline_matriz_confusao_final.csv")
print(f"{out_dir}/exp04_hierRF_cv_pipeline_predicoes.csv")