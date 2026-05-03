import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

# =========================
# 1. Carregar dados
# =========================

path = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1.csv"
out_dir = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed"

df = pd.read_csv(path)

print("Shape:", df.shape)

# =========================
# 2. Definir features
# =========================

cols_drop = [
    "classe_nivel1",
    "classe_nivel2",
    "supercells_global",
    "sc_id"
]

X = df.drop(columns=[c for c in cols_drop if c in df.columns])

y_nivel1 = df["classe_nivel1"]
y_nivel2 = df["classe_nivel2"]

print("Features:", X.shape[1])
print("\nClasses nível 1:")
print(y_nivel1.value_counts())

print("\nClasses nível 2:")
print(y_nivel2.value_counts())

# =========================
# 3. Configuração CV
# =========================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# =========================
# 4. Modelo nível 1
# =========================

model_n1 = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

print("\nTreinando/predizendo nível 1...")

pred_n1 = cross_val_predict(
    model_n1,
    X,
    y_nivel1,
    cv=cv,
    n_jobs=-1
)

print("\n=== RESULTADOS NÍVEL 1 ===")
print("F1 macro:", f1_score(y_nivel1, pred_n1, average="macro"))
print("Balanced accuracy:", balanced_accuracy_score(y_nivel1, pred_n1))
print(classification_report(y_nivel1, pred_n1))

# Salvar report nível 1
report_n1 = pd.DataFrame(
    classification_report(y_nivel1, pred_n1, output_dict=True)
).transpose()

report_n1.to_csv(f"{out_dir}/exp03_hierRF_report_nivel1.csv")

cm_n1 = confusion_matrix(y_nivel1, pred_n1, labels=sorted(y_nivel1.unique()))
pd.DataFrame(
    cm_n1,
    index=sorted(y_nivel1.unique()),
    columns=sorted(y_nivel1.unique())
).to_csv(f"{out_dir}/exp03_hierRF_matriz_confusao_nivel1.csv")

# =========================
# 5. Modelo nível 2
# =========================

mask_vr = df["classe_nivel1"] == "Vegetação Rasteira"

X_vr = X.loc[mask_vr].copy()
y_vr = df.loc[mask_vr, "classe_nivel2"].copy()

print("\nSubconjunto Vegetação Rasteira:", X_vr.shape)
print(y_vr.value_counts())

model_n2 = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

print("\nTreinando/predizendo nível 2...")

pred_n2 = cross_val_predict(
    model_n2,
    X_vr,
    y_vr,
    cv=cv,
    n_jobs=-1
)

print("\n=== RESULTADOS NÍVEL 2: VEGETAÇÃO RASTEIRA ===")
print("F1 macro:", f1_score(y_vr, pred_n2, average="macro"))
print("Balanced accuracy:", balanced_accuracy_score(y_vr, pred_n2))
print(classification_report(y_vr, pred_n2))

# Salvar report nível 2
report_n2 = pd.DataFrame(
    classification_report(y_vr, pred_n2, output_dict=True)
).transpose()

report_n2.to_csv(f"{out_dir}/exp03_hierRF_report_nivel2_vegetacao_rasteira.csv")

cm_n2 = confusion_matrix(y_vr, pred_n2, labels=sorted(y_vr.unique()))
pd.DataFrame(
    cm_n2,
    index=sorted(y_vr.unique()),
    columns=sorted(y_vr.unique())
).to_csv(f"{out_dir}/exp03_hierRF_matriz_confusao_nivel2_vegetacao_rasteira.csv")

# =========================
# 6. Predição final hierárquica
# =========================

df_pred = df[["supercells_global", "classe_nivel1", "classe_nivel2"]].copy()
df_pred["pred_nivel1"] = pred_n1
df_pred["pred_final_hier"] = df_pred["pred_nivel1"]

# inserir predições de nível 2 somente onde o nível 1 previu Vegetação Rasteira
# Atenção: para avaliação CV aproximada, usamos pred_n2 gerado no subconjunto verdadeiro de Vegetação Rasteira.
# Para casos em que o nível 1 previu Vegetação Rasteira, mas a classe real não era Vegetação Rasteira,
# não há pred_n2 correspondente. Nesses casos, mantemos "Vegetação Rasteira" como erro final.

df_pred.loc[mask_vr, "pred_nivel2_real_subset"] = pred_n2

idx_vr_real = df_pred.index[mask_vr]
df_pred.loc[idx_vr_real, "pred_final_hier"] = pred_n2

# Para avaliação final, a classe verdadeira é classe_nivel2
y_true_final = df_pred["classe_nivel2"]
y_pred_final = df_pred["pred_final_hier"]

print("\n=== RESULTADO FINAL HIERÁRQUICO RF ===")
print("F1 macro:", f1_score(y_true_final, y_pred_final, average="macro"))
print("Balanced accuracy:", balanced_accuracy_score(y_true_final, y_pred_final))
print(classification_report(y_true_final, y_pred_final))

# =========================
# 7. Salvar resultados finais
# =========================

report_final = pd.DataFrame(
    classification_report(y_true_final, y_pred_final, output_dict=True)
).transpose()

report_final.to_csv(f"{out_dir}/exp03_hierRF_report_final.csv")

labels_final = sorted(y_true_final.unique())
cm_final = confusion_matrix(y_true_final, y_pred_final, labels=labels_final)

pd.DataFrame(
    cm_final,
    index=labels_final,
    columns=labels_final
).to_csv(f"{out_dir}/exp03_hierRF_matriz_confusao_final.csv")

df_pred.to_csv(f"{out_dir}/exp03_hierRF_predicoes.csv", index=False)

print("\nArquivos salvos:")
print(f"{out_dir}/exp03_hierRF_report_nivel1.csv")
print(f"{out_dir}/exp03_hierRF_report_nivel2_vegetacao_rasteira.csv")
print(f"{out_dir}/exp03_hierRF_report_final.csv")
print(f"{out_dir}/exp03_hierRF_matriz_confusao_final.csv")
print(f"{out_dir}/exp03_hierRF_predicoes.csv")