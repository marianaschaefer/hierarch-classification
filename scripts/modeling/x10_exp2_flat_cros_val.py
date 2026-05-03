import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score

path = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1.csv"
out_dir = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed"

df = pd.read_csv(path)

y = df["classe_nivel2"]

cols_drop = ["classe_nivel1", "classe_nivel2", "supercells_global", "sc_id"]
X = df.drop(columns=[c for c in cols_drop if c in df.columns])

model = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

print("F1 macro:", f1_score(y, y_pred, average="macro"))
print("Balanced accuracy:", balanced_accuracy_score(y, y_pred))

report = classification_report(y, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(f"{out_dir}/exp01_flatRF_report_por_classe.csv")

labels = sorted(y.unique())
cm = confusion_matrix(y, y_pred, labels=labels)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
df_cm.to_csv(f"{out_dir}/exp01_flatRF_matriz_confusao.csv")

print(df_report)
print("\nArquivos salvos:")
print(f"{out_dir}/exp01_flatRF_report_por_classe.csv")
print(f"{out_dir}/exp01_flatRF_matriz_confusao.csv")