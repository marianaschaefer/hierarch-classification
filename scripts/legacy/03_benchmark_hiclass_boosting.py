import pandas as pd
import numpy as np
import os
import time
import warnings

from hiclass import LocalClassifierPerNode
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Silenciar avisos estéticos do LightGBM/Sklearn para limpar o terminal
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
os.makedirs("results", exist_ok=True)

def calcular_metricas_hierarquicas(y_true, y_pred):
    total = len(y_true)
    same_n1 = (y_true[:, 0] == y_pred[:, 0])
    same_n2 = same_n1 & (y_true[:, 1] == y_pred[:, 1])

    acertos_n2 = same_n2.sum()
    acertos_n1 = same_n1.sum()
    acertos_somente_n1 = acertos_n1 - acertos_n2

    matches = 2 * acertos_n2 + 1 * acertos_somente_n1
    hP = matches / (2 * total)
    hR = hP
    hF1 = 2 * hP * hR / (hP + hR) if (hP + hR) > 0 else 0.0
    return hF1

def criar_modelos_boosting_com_pesos(y_nivel2):
    """
    Cria dicionário de modelos com tratamento de desbalanceamento robusto para HiClass.
    """
    classes = np.unique(y_nivel2)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_nivel2)
    ratio_max = cw.max() / cw.min() if cw.min() > 0 else 1.0

    modelos = {
        "XGBoost_HiClass": XGBClassifier(
            n_estimators=100, n_jobs=-1, random_state=42,
            scale_pos_weight=ratio_max, eval_metric="mlogloss"
        ),
        "LightGBM_HiClass": LGBMClassifier(
            n_estimators=100, n_jobs=-1, random_state=42,
            verbose=-1, class_weight="balanced"
        ),
        "CatBoost_HiClass": CatBoostClassifier(
            n_estimators=100, silent=True, thread_count=-1, random_state=42,
            auto_class_weights='Balanced' 
        )
    }
    return modelos

def benchmark_boosting_hiclass():
    print("--- Benchmark HiClass Boosting (Balanced) ---")

    df = pd.read_csv("data_processed/dataset_MULTIMODAL_LIMPO.csv")
    supercells = df["supercells"].values
    cols_features = [c for c in df.columns if c not in ["supercells", "classe_nivel1", "classe_nivel2"]]
    
    X = df[cols_features].values
    y = df[["classe_nivel1", "classe_nivel2"]].values
    y_n2 = df["classe_nivel2"].values

    modelos = criar_modelos_boosting_com_pesos(y_n2)

    resultados_detalhados = []
    resultados_resumo = []
    mapa_preds_por_modelo = {nome: np.zeros(len(df), dtype=object) for nome in modelos}

    for nome, estimador_base in modelos.items():
        print(f"\n--- Avaliando {nome} ---")
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

        hF1_scores, acc_n1_scores, acc_n2_scores = [], [], []
        f1_m1_scores, f1_m2_scores = [], []
        t_treino_scores, t_pred_scores = [], []
        votos_n2 = {idx: {} for idx in range(len(df))}

        for split_id, (train_idx, test_idx) in enumerate(sss.split(X, y[:, 1]), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Instanciar novo classificador preservando os parâmetros de balanceamento
            if "XGBoost" in nome:
                estimador = XGBClassifier(**estimador_base.get_params())
            elif "LightGBM" in nome:
                estimador = LGBMClassifier(**estimador_base.get_params())
            else:
                estimador = CatBoostClassifier(**estimador_base.get_params())

            classifier = LocalClassifierPerNode(local_classifier=estimador)

            t0 = time.time()
            classifier.fit(X_train, y_train)
            t1 = time.time()
            preds = classifier.predict(X_test)
            t2 = time.time()

            # Métricas
            hf1 = calcular_metricas_hierarquicas(y_test, preds)
            acc1 = accuracy_score(y_test[:, 0], preds[:, 0])
            acc2 = accuracy_score(y_test[:, 1], preds[:, 1])
            f1_m1 = f1_score(y_test[:, 0], preds[:, 0], average="macro")
            f1_m2 = f1_score(y_test[:, 1], preds[:, 1], average="macro")

            hF1_scores.append(hf1); acc_n1_scores.append(acc1); acc_n2_scores.append(acc2)
            f1_m1_scores.append(f1_m1); f1_m2_scores.append(f1_m2)
            t_treino_scores.append(t1-t0); t_pred_scores.append(t2-t1)

            for idx_local, idx_global in enumerate(test_idx):
                label_n2 = preds[idx_local, 1]
                votos_n2[idx_global][label_n2] = votos_n2[idx_global].get(label_n2, 0) + 1

        # Voto majoritário para o CSV do mapa
        y2_pred_final = []
        for i in range(len(df)):
            y2_pred_final.append(max(votos_n2[i], key=votos_n2[i].get) if votos_n2[i] else np.nan)
        mapa_preds_por_modelo[nome] = np.array(y2_pred_final, dtype=object)

        resultados_resumo.append({
            "Modelo": nome,
            "hF1_medio": np.mean(hF1_scores), "hF1_std": np.std(hF1_scores),
            "Acc_N1_media": np.mean(acc_n1_scores), "Acc_N2_media": np.mean(acc_n2_scores),
            "F1_macro_N2_medio": np.mean(f1_m2_scores)
        })

    pd.DataFrame(resultados_resumo).to_csv("results/boosting_hiclass_resultados_resumo.csv", index=False)
    
    # Gerar CSV para o mapa
    map_df = pd.DataFrame({"supercells": supercells, "classe_nivel2_real": y_n2})
    for n, p in mapa_preds_por_modelo.items(): map_df[f"classe_nivel2_pred_{n}"] = p
    map_df.to_csv("results/map_data_hiclass.csv", index=False)
    
    print("\n" + "="*40 + "\nRESUMO HICLASS BALANCEADO\n" + "="*40)
    print(pd.DataFrame(resultados_resumo))

if __name__ == "__main__":
    benchmark_boosting_hiclass()