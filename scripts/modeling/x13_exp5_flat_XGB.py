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
# 1. Load
# =========================

path = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1.csv"
out_dir = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed"

df = pd.read_csv(path)

print("Shape:", df.shape)

# =========================
# 2. Features / Target
# =========================

cols_drop = [
    "classe_nivel1",
    "classe_nivel2",
    "supercells_global",
    "sc_id"
]

X = df.drop(columns=[c for c in cols_drop if c in df.columns])
y = df["classe_nivel2"]

print("Features:", X.shape[1])
print("Classes:")
print(y.value_counts())

# =========================
# 3. Encode labels (XGB precisa)
# =========================

y_encoded, class_names = pd.factorize(y)

print("\nMapeamento classes:")
for i, c in enumerate(class_names):
    print(i, "->", c)

# =========================
# 4. Modelo
# =========================

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

# =========================
# 5. CV
# =========================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# =========================
# 6. Predict
# =========================

print("\nRodando cross_val_predict...")

y_pred_encoded = cross_val_predict(
    model,
    X,
    y_encoded,
    cv=cv,
    n_jobs=-1
)

# voltar para labels originais
y_pred = class_names[y_pred_encoded]

# =========================
# 7. Métricas
# =========================

print("\n=== RESULTADOS FLAT XGBOOST ===")

print("F1 macro:", f1_score(y, y_pred, average="macro"))
print("Balanced accuracy:", balanced_accuracy_score(y, y_pred))

report = classification_report(y, y_pred, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()

print(df_report)

# =========================
# 8. Salvar
# =========================

df_report.to_csv(f"{out_dir}/exp05_flatXGB_report.csv")

labels = sorted(y.unique())
cm = confusion_matrix(y, y_pred, labels=labels)

pd.DataFrame(
    cm,
    index=labels,
    columns=labels
).to_csv(f"{out_dir}/exp05_flatXGB_matriz_confusao.csv")

print("\nArquivos salvos:")
print(f"{out_dir}/exp05_flatXGB_report.csv")
print(f"{out_dir}/exp05_flatXGB_matriz_confusao.csv")