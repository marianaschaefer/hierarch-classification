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
# PATHS
# =========================

path_data = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1.csv"
path_pureza = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/pureza_poligonos_mapbiomas.csv"

out_dir = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed"

# =========================
# LOAD
# =========================

df = pd.read_csv(path_data)
df_p = pd.read_csv(path_pureza)

df = df.merge(
    df_p[["supercells_global", "prop_label"]],
    on="supercells_global",
    how="inner"
)

print("Shape total:", df.shape)

# =========================
# FEATURES / TARGET
# =========================

cols_drop = [
    "classe_nivel1",
    "classe_nivel2",
    "supercells_global",
    "sc_id",
    "prop_label"
]

X_all = df.drop(columns=[c for c in cols_drop if c in df.columns])
y_all = df["classe_nivel2"]

print("Features:", X_all.shape[1])

# =========================
# CONFIG
# =========================

thresholds = [0.0, 0.6, 0.7, 0.8, 0.9]

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

summary = []

# =========================
# LOOP
# =========================

for thr in thresholds:

    print("\n==============================")
    print(f"THRESHOLD: {thr}")
    print("==============================")

    if thr == 0.0:
        df_sub = df.copy()
    else:
        df_sub = df[df["prop_label"] >= thr].copy()

    print("Amostras:", len(df_sub))

    if len(df_sub) < 1000:
        print("Poucas amostras, pulando...")
        continue

    X = df_sub[X_all.columns]
    y = df_sub["classe_nivel2"]

    # encode
    y_enc, class_names = pd.factorize(y)

    y_pred_enc = cross_val_predict(
        model,
        X,
        y_enc,
        cv=cv,
        n_jobs=-1
    )

    y_pred = class_names[y_pred_enc]

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

    # salvar por threshold
    df_report.to_csv(f"{out_dir}/exp08_thr_{thr}_report.csv")

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    ).to_csv(f"{out_dir}/exp08_thr_{thr}_cm.csv")

    summary.append({
        "threshold": thr,
        "n_samples": len(df_sub),
        "f1_macro": f1,
        "balanced_accuracy": bacc,
        "cerrado_f1": df_report.loc["Cerrado", "f1-score"] if "Cerrado" in df_report.index else None,
        "cerrado_recall": df_report.loc["Cerrado", "recall"] if "Cerrado" in df_report.index else None,
        "agricultura_f1": df_report.loc["Agricultura", "f1-score"],
        "pastagem_f1": df_report.loc["Pastagem", "f1-score"],
        "formacao_campestre_f1": df_report.loc["Formação Campestre", "f1-score"]
    })

# =========================
# SAVE SUMMARY
# =========================

df_summary = pd.DataFrame(summary)

df_summary.to_csv(f"{out_dir}/exp08_pureza_vs_performance_summary.csv", index=False)

print("\n=== RESUMO FINAL ===")
print(df_summary)
print(f"\nSalvo em: {out_dir}/exp08_pureza_vs_performance_summary.csv")