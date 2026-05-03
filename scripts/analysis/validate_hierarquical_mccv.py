import pandas as pd
import numpy as np
import os
from hiclass import LocalClassifierPerNode
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# Criar pasta de resultados se não existir
os.makedirs("results", exist_ok=True)

def run_hiclass_benchmark():
    df = pd.read_csv('data_processed/dataset_MULTIMODAL_LIMPO.csv')
    cols_features = [c for c in df.columns if c not in ['supercells', 'classe_nivel1', 'classe_nivel2']]
    X = df[cols_features].values
    y = df[['classe_nivel1', 'classe_nivel2']].values

    modelos = {
        'RF_HiClass': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'XGB_HiClass': XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'CatBoost_HiClass': CatBoostClassifier(n_estimators=100, silent=True, random_state=42)
    }

    all_results = []

    for nome, base_clf in modelos.items():
        print(f"Executando {nome}...")
        # Usaremos apenas 1 split para salvar as predições detalhadas (CSV de análise)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        
        for train_idx, test_idx in sss.split(X, y[:, 1]):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            classifier = LocalClassifierPerNode(local_classifier=base_clf)
            classifier.fit(X_train, y_train)
            preds = classifier.predict(X_test)
            
            # 1. Salvar Predições para análises futuras
            df_preds = pd.DataFrame({
                'true_n1': y_test[:, 0],
                'true_n2': y_test[:, 1],
                'pred_n1': preds[:, 0],
                'pred_n2': preds[:, 1]
            })
            df_preds.to_csv(f'results/preds_{nome}.csv', index=False)
            
            # 2. Gerar Classification Report para Nível 2
            report = classification_report(y_test[:, 1], preds[:, 1], output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            df_report.to_csv(f'results/metrics_{nome}_n2.csv')
            
            print(f"✅ {nome} concluído. Resultados salvos em results/.")

if __name__ == "__main__":
    run_hiclass_benchmark()