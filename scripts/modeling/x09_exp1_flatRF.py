import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score

# =========================
# 1. Carregar dados
# =========================
path = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1.csv"

df = pd.read_csv(path)

print("Shape:", df.shape)

# =========================
# 2. Definir X e y
# =========================
y = df["classe_nivel2"]

cols_drop = [
    "classe_nivel1",
    "classe_nivel2",
    "supercells_global",
    "sc_id"
]

X = df.drop(columns=[c for c in cols_drop if c in df.columns])

print("Features:", X.shape[1])

# =========================
# 3. Modelo
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

# =========================
# 4. Validação
# =========================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scoring = {
    "f1_macro": "f1_macro",
    "balanced_acc": "balanced_accuracy"
}

# =========================
# 5. Avaliação
# =========================
results = {}

for name, scorer in scoring.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    
    results[name] = {
        "mean": scores.mean(),
        "std": scores.std()
    }

# =========================
# 6. Output
# =========================
print("\n=== RESULTADOS ===")
for metric, values in results.items():
    print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")