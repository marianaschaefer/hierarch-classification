import pandas as pd
import numpy as np
import os
import time
import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs("results", exist_ok=True)

def calcular_metricas_hierarquicas(y1_true, y2_true, y1_pred, y2_pred):
    # Forçar que todos sejam vetores 1D (corrige o erro do CatBoost)
    y1_true, y2_true = y1_true.ravel(), y2_true.ravel()
    y1_pred, y2_pred = y1_pred.ravel(), y2_pred.ravel()
    
    total = len(y1_true)
    same_n1 = (y1_true == y1_pred)
    same_n2 = same_n1 & (y2_true == y2_pred)
    
    matches = 2 * same_n2.sum() + 1 * (same_n1.sum() - same_n2.sum())
    return matches / (2 * total)

def criar_modelos_boosting_flat(y_nivel2):
    classes = np.unique(y_nivel2)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_nivel2)
    ratio_max = cw.max() / cw.min() if cw.min() > 0 else 1.0

    return {
        "XGBoost_Flat": XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42, scale_pos_weight=ratio_max, eval_metric="mlogloss"),
        "LightGBM_Flat": LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbose=-1, class_weight="balanced"),
        "CatBoost_Flat": CatBoostClassifier(n_estimators=100, silent=True, random_state=42, auto_class_weights='Balanced')
    }

def benchmark_flat_stacked():
    print("--- Gerando Dados Finais: Flat Stacked (Corrigido) ---")
    df = pd.read_csv("data_processed/dataset_MULTIMODAL_LIMPO.csv")
    
    cols_features = [c for c in df.columns if c not in ["supercells", "classe_nivel1", "classe_nivel2"]]
    X = df[cols_features].values
    
    le1, le2 = LabelEncoder(), LabelEncoder()
    y1 = le1.fit_transform(df["classe_nivel1"])
    y2 = le2.fit_transform(df["classe_nivel2"])

    modelos = criar_modelos_boosting_flat(y2)
    resultados_resumo = []

    for nome, estimador_base in modelos.items():
        print(f"  Avaliando {nome}...")
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        m_hf1, m_acc1, m_acc2, m_f1m2 = [], [], [], []

        for i, (train_idx, test_idx) in enumerate(sss.split(X, y2), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y1_train, y1_test = y1[train_idx], y1[test_idx]
            y2_train, y2_test = y2[train_idx], y2[test_idx]

            # Treino Nível 1
            clf1 = estimador_base.__class__(**estimador_base.get_params())
            clf1.fit(X_train, y1_train)
            p1_test = clf1.predict(X_test).ravel() # .ravel() garante 1D

            # Treino Nível 2 (Stacking)
            p1_train_rich = clf1.predict(X_train).reshape(-1, 1)
            X_tr_rich = np.hstack([X_train, p1_train_rich])
            X_te_rich = np.hstack([X_test, p1_test.reshape(-1, 1)])

            clf2 = estimador_base.__class__(**estimador_base.get_params())
            clf2.fit(X_tr_rich, y2_train)
            p2_test = clf2.predict(X_te_rich).ravel() # .ravel() garante 1D

            m_hf1.append(calcular_metricas_hierarquicas(y1_test, y2_test, p1_test, p2_test))
            m_acc1.append(accuracy_score(y1_test, p1_test))
            m_acc2.append(accuracy_score(y2_test, p2_test))
            m_f1m2.append(f1_score(y2_test, p2_test, average="macro"))

        resultados_resumo.append({
            "Modelo": nome, "hF1_medio": np.mean(m_hf1), "hF1_std": np.std(m_hf1),
            "Acc_N1_media": np.mean(m_acc1), "Acc_N2_media": np.mean(m_acc2),
            "F1_macro_N2_medio": np.mean(m_f1m2)
        })

    df_res = pd.DataFrame(resultados_resumo)
    df_res.to_csv("results/boosting_flat_resultados_resumo.csv", index=False)
    print("\n" + "="*40 + "\nRESUMO FLAT CORRIGIDO\n" + "="*40)
    print(df_res)

if __name__ == "__main__":
    benchmark_flat_stacked()