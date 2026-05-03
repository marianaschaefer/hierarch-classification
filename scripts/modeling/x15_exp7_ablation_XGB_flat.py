import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

# =========================
# CONFIG
# =========================

path = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1.csv"
out_dir = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed"

RANDOM_STATE = 42
N_SPLITS = 10

# =========================
# LOAD
# =========================

df = pd.read_csv(path)

y = df["classe_nivel2"]

# =========================
# FEATURE GROUPS
# =========================

cols_id = ["classe_nivel1", "classe_nivel2", "supercells_global", "sc_id"]

embedding_cols = [f"A{i:02d}" for i in range(64)]

spectral_cols = [
    c for c in df.columns
    if c not in cols_id and c not in embedding_cols
]

feature_sets = {
    "spectral_indices_mde": spectral_cols,
    "embeddings": embedding_cols,
    "all_features": spectral_cols + embedding_cols
}

print("Shape:", df.shape)
print("Classes:")
print(y.value_counts())

print("\nFeature groups:")
for name, cols in feature_sets.items():
    print(name, len(cols))

# =========================
# MODEL / CV
# =========================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

summary = []

# =========================
# RUN EXPERIMENTS
# =========================

for exp_name, cols in feature_sets.items():
    print(f"\n==============================")
    print(f"EXPERIMENTO: {exp_name}")
    print(f"Features: {len(cols)}")
    print(f"==============================")

    X = df[cols].copy()

    y_encoded, class_names = pd.factorize(y)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    y_pred_encoded = cross_val_predict(
        model,
        X,
        y_encoded,
        cv=cv,
        n_jobs=-1
    )

    y_pred = class_names[y_pred_encoded]

    f1 = f1_score(y, y_pred, average="macro")
    bacc = balanced_accuracy_score(y, y_pred)

    print("F1 macro:", f1)
    print("Balanced accuracy:", bacc)

    report = classification_report(
        y,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f"{out_dir}/exp07_ablation_XGB_{exp_name}_report.csv")

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    ).to_csv(f"{out_dir}/exp07_ablation_XGB_{exp_name}_matriz_confusao.csv")

    summary.append({
        "experiment": exp_name,
        "n_features": len(cols),
        "f1_macro": f1,
        "balanced_accuracy": bacc,
        "cerrado_f1": df_report.loc["Cerrado", "f1-score"],
        "cerrado_recall": df_report.loc["Cerrado", "recall"],
        "agricultura_f1": df_report.loc["Agricultura", "f1-score"],
        "pastagem_f1": df_report.loc["Pastagem", "f1-score"],
        "formacao_campestre_f1": df_report.loc["Formação Campestre", "f1-score"],
        "solo_exposto_f1": df_report.loc["Solo Exposto", "f1-score"],
    })

# =========================
# SAVE SUMMARY
# =========================

df_summary = pd.DataFrame(summary)
df_summary.to_csv(f"{out_dir}/exp07_ablation_XGB_summary.csv", index=False)

print("\n=== RESUMO ABLATION ===")
print(df_summary)
print(f"\nResumo salvo em: {out_dir}/exp07_ablation_XGB_summary.csv")