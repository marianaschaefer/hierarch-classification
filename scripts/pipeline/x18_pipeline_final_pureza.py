import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    accuracy_score
)

# =========================================================
# CONFIG
# =========================================================

DATA_PATH = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1.csv"
PUREZA_PATH = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/pureza_poligonos_mapbiomas.csv"
OUT_DIR = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed"

THRESHOLD = 0.6
RANDOM_STATE = 42
N_SPLITS = 10

# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(DATA_PATH)
df_p = pd.read_csv(PUREZA_PATH)

df = df.merge(
    df_p[["supercells_global", "prop_label", "classe_dominante", "prop_dominante"]],
    on="supercells_global",
    how="inner"
)

print("\n=== BASE ORIGINAL ===")
print("Shape:", df.shape)
print("\nClasse nível 1:")
print(df["classe_nivel1"].value_counts())
print("\nClasse nível 2:")
print(df["classe_nivel2"].value_counts())

# =========================================================
# FILTRAR POR PUREZA
# =========================================================

df_f = df[df["prop_label"] >= THRESHOLD].copy()

print("\n" + "=" * 60)
print(f"BASE FILTRADA POR PUREZA >= {THRESHOLD}")
print("=" * 60)
print("Shape:", df_f.shape)

print("\nAmostras por classe_nivel1:")
print(df_f["classe_nivel1"].value_counts())

print("\nAmostras por classe_nivel2:")
print(df_f["classe_nivel2"].value_counts())

print("\nResumo prop_label:")
print(df_f["prop_label"].describe())

# salvar base filtrada
filtered_path = f"{OUT_DIR}/dataset_FINAL_COMPLETO_global_v1_pureza_{THRESHOLD}.csv"
df_f.to_csv(filtered_path, index=False)
print("\nBase filtrada salva em:", filtered_path)

# =========================================================
# FEATURES / TARGET
# =========================================================

cols_drop = [
    "classe_nivel1",
    "classe_nivel2",
    "supercells_global",
    "sc_id",
    "prop_label",
    "classe_dominante",
    "prop_dominante"
]

X = df_f.drop(columns=[c for c in cols_drop if c in df_f.columns])
y = df_f["classe_nivel2"]

print("\nFeatures utilizadas:", X.shape[1])

# alerta de classes pequenas
counts = y.value_counts()
print("\nMenor classe:", counts.min())
print(counts.sort_values())

if counts.min() < N_SPLITS:
    print(
        f"\nATENÇÃO: há classe com menos de {N_SPLITS} amostras. "
        "Reduza N_SPLITS ou remova essa classe para validação estável."
    )

# =========================================================
# CV
# =========================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# =========================================================
# FUNÇÃO DE AVALIAÇÃO
# =========================================================

def avaliar_modelo(nome_modelo, model, X, y):
    print("\n" + "=" * 60)
    print(f"RODANDO: {nome_modelo}")
    print("=" * 60)

    if nome_modelo.startswith("XGB"):
        y_enc, class_names = pd.factorize(y)

        y_pred_enc = cross_val_predict(
            model,
            X,
            y_enc,
            cv=cv,
            n_jobs=-1
        )

        y_pred = class_names[y_pred_enc]

    else:
        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            n_jobs=-1
        )

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    bacc = balanced_accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)

    print("\nMétricas:")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bacc:.4f}")
    print(f"F1 macro:          {f1:.4f}")
    print(f"Kappa:             {kappa:.4f}")

    report = classification_report(
        y,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    df_report = pd.DataFrame(report).transpose()
    print("\nRelatório por classe:")
    print(df_report)

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)

    prefix = f"{OUT_DIR}/final_pureza_{THRESHOLD}_{nome_modelo}"

    df_report.to_csv(f"{prefix}_report.csv")

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    ).to_csv(f"{prefix}_matriz_confusao.csv")

    df_pred = pd.DataFrame({
        "y_true": y.values,
        "y_pred": y_pred
    })
    df_pred.to_csv(f"{prefix}_predicoes.csv", index=False)

    return {
        "modelo": nome_modelo,
        "threshold": THRESHOLD,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1_macro": f1,
        "kappa": kappa
    }

# =========================================================
# MODELOS
# =========================================================

models = {
    "RF_flat": RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),

    "XGB_flat": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
}

# =========================================================
# EXECUTAR
# =========================================================

summary = []

for nome, model in models.items():
    res = avaliar_modelo(nome, model, X, y)
    summary.append(res)

df_summary = pd.DataFrame(summary)
summary_path = f"{OUT_DIR}/final_pureza_{THRESHOLD}_summary.csv"
df_summary.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(df_summary)

print("\nArquivos principais salvos:")
print(filtered_path)
print(summary_path)