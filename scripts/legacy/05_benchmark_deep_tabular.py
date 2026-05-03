import pandas as pd
import numpy as np
import os
import time
import warnings

# Importações específicas do PyTorch Tabular
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, FTTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

# Silenciar avisos desnecessários de telemetria
warnings.filterwarnings("ignore", ".*is_categorical_dtype*.")
os.makedirs("results", exist_ok=True)

def calcular_hf1(y1_true, y2_true, y1_pred, y2_pred):
    total = len(y1_true)
    same_n1 = (y1_true == y1_pred)
    same_n2 = same_n1 & (y2_true == y2_pred)
    matches = 2 * same_n2.sum() + 1 * (same_n1.sum() - same_n2.sum())
    return matches / (2 * total)

def benchmark_deep_tabular():
    print("--- Benchmark Deep Tabular (DeepMLP vs FT-Transformer) ---")

    df = pd.read_csv('data_processed/dataset_MULTIMODAL_LIMPO.csv')

    target = ["classe_nivel1", "classe_nivel2"]
    num_cols = [c for c in df.columns if c not in ['supercells'] + target]

    data_config = DataConfig(
        target=target,
        continuous_cols=num_cols,
        categorical_cols=[],
    )

    trainer_config = TrainerConfig(
        batch_size=128,
        max_epochs=50,
        early_stopping="valid_loss",
        early_stopping_patience=10,
        accelerator="auto", # Detecta GPU automaticamente
    )

    optimizer_config = OptimizerConfig()

    # Definição das arquiteturas Deep
    model_configs = {
        'DeepMLP': CategoryEmbeddingModelConfig(
            task="classification", 
            layers="256-128-64", 
            learning_rate=1e-3
        ),
        'FTTransformer': FTTransformerConfig(
            task="classification", 
            learning_rate=1e-4
        ),
    }

    resultados_detalhados = []
    resultados_resumo = []

    for nome, m_config in model_configs.items():
        print(f"\n--- Treinando {nome} ---")

        # 5 splits devido ao tempo de treino de DL
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        hF1_list, acc_n1_list, acc_n2_list = [], [], []
        f1_n1_list, f1_n2_list = [], []
        t_treino_list, t_pred_list = [], []

        for i, (train_idx, val_idx) in enumerate(sss.split(df, df["classe_nivel2"]), start=1):
            print(f"  Split {i}/5...")
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            tabular_model = TabularModel(
                data_config=data_config,
                model_config=m_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
            )

            t0 = time.time()
            tabular_model.fit(train=train_df, validation=val_df)
            t1 = time.time()

            preds = tabular_model.predict(val_df)
            t2 = time.time()

            # Extração das predições multitarefa
            p1 = preds[f"{target[0]}_prediction"].values
            p2 = preds[f"{target[1]}_prediction"].values
            y1_true = val_df[target[0]].values
            y2_true = val_df[target[1]].values

            hf1 = calcular_hf1(y1_true, y2_true, p1, p2)
            acc1 = accuracy_score(y1_true, p1)
            acc2 = accuracy_score(y2_true, p2)
            f1_n1 = f1_score(y1_true, p1, average='macro')
            f1_n2 = f1_score(y2_true, p2, average='macro')

            t_treino, t_pred = t1 - t0, t2 - t1

            hF1_list.append(hf1)
            acc_n1_list.append(acc1); acc_n2_list.append(acc2)
            f1_n1_list.append(f1_n1); f1_n2_list.append(f1_n2)
            t_treino_list.append(t_treino); t_pred_list.append(t_pred)

            resultados_detalhados.append({
                'Modelo': nome, 'Split': i, 'hF1': hf1,
                'Acc_N1': acc1, 'Acc_N2': acc2,
                'F1_macro_N1': f1_n1, 'F1_macro_N2': f1_n2,
                'Tempo_treino_s': t_treino, 'Tempo_pred_s': t_pred,
            })
            print(f"    hF1: {hf1:.4f} | Acc_N2: {acc2:.4f}")

        resultados_resumo.append({
            'Modelo': nome,
            'hF1_medio': np.mean(hF1_list), 'hF1_std': np.std(hF1_list),
            'Acc_N1_media': np.mean(acc_n1_list), 'Acc_N2_media': np.mean(acc_n2_list),
            'F1_macro_N1_medio': np.mean(f1_n1_list), 'F1_macro_N2_medio': np.mean(f1_n2_list),
            'Tempo_treino_medio_s': np.mean(t_treino_list), 'Tempo_pred_medio_s': np.mean(t_pred_list),
        })

    pd.DataFrame(resultados_detalhados).to_csv("results/deep_tabular_resultados_detalhados.csv", index=False)
    df_res = pd.DataFrame(resultados_resumo)
    df_res.to_csv("results/deep_tabular_resultados_resumo.csv", index=False)

    print("\n" + "="*60)
    print("QUADRO COMPARATIVO FINAL (DEEP LEARNING TABULAR)")
    print(df_res)
    print("="*60)

if __name__ == "__main__":
    benchmark_deep_tabular()