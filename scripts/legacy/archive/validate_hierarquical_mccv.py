import pandas as pd
import numpy as np
from hiclass import LocalClassifierPerNode
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')


def calcular_metricas_hierarquicas(y_true, y_pred):
    """
    Calcula hP, hR e hF1 para hierarquia de 2 níveis.
    P_i = {nivel1_pred, nivel2_pred}
    T_i = {nivel1_true, nivel2_true}
    |P_i| = |T_i| = 2; logo hP = hR e hF1 = hP.
    """
    total = len(y_true)

    same_n1 = (y_true[:, 0] == y_pred[:, 0])
    same_n2 = same_n1 & (y_true[:, 1] == y_pred[:, 1])

    acertos_n2 = same_n2.sum()
    acertos_n1 = same_n1.sum()
    acertos_somente_n1 = acertos_n1 - acertos_n2

    # |P_i ∩ T_i| = 2 se acerta N1 e N2, 1 se só N1, 0 caso contrário
    matches = 2 * acertos_n2 + 1 * acertos_somente_n1

    hP = matches / (2 * total)
    hR = hP
    if hP + hR > 0:
        hF1 = 2 * hP * hR / (hP + hR)
    else:
        hF1 = 0.0

    return hP, hR, hF1


def validar_robusto():
    print("--- Validação Monte Carlo (10 iterações, 80/20 estratificado) ---")
    df = pd.read_csv('data_processed/dataset_MULTIMODAL_LIMPO.csv')

    cols_features = [c for c in df.columns
                     if c not in ['supercells', 'classe_nivel1', 'classe_nivel2']]
    X = df[cols_features].values
    y = df[['classe_nivel1', 'classe_nivel2']].values

    # Estratifica pelas subclasses (Nível 2)
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    hP_list, hR_list, hF1_list = [], [], []
    acc_n1_list, acc_n2_list = [], []

    for i, (train_index, test_index) in enumerate(sss.split(X, y[:, 1])):
        print(f"\n=== Rodada {i + 1} ===")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LocalClassifierPerNode(
            local_classifier=RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=42
            )
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Métricas hierárquicas
        hP, hR, hF1 = calcular_metricas_hierarquicas(y_test, preds)
        hP_list.append(hP)
        hR_list.append(hR)
        hF1_list.append(hF1)

        print(f"Métricas hierárquicas: hP={hP:.4f}, hR={hR:.4f}, hF1={hF1:.4f}")

        # Relatório por nível
        print("\nNível 1 (Macro):")
        print(classification_report(y_test[:, 0], preds[:, 0]))
        print("Nível 2 (Subclasses):")
        print(classification_report(y_test[:, 1], preds[:, 1]))

        # Acurácias simples por nível (para resumo)
        acc_n1 = accuracy_score(y_test[:, 0], preds[:, 0])
        acc_n2 = accuracy_score(y_test[:, 1], preds[:, 1])
        acc_n1_list.append(acc_n1)
        acc_n2_list.append(acc_n2)
        print(f"Acurácia Nível 1: {acc_n1:.4f}")
        print(f"Acurácia Nível 2: {acc_n2:.4f}")

    print("\n" + "-" * 50)
    print("Resumo Monte Carlo (10 iterações):")
    print(f"hP médio:  {np.mean(hP_list):.4f} (+/- {np.std(hP_list):.4f})")
    print(f"hR médio:  {np.mean(hR_list):.4f} (+/- {np.std(hR_list):.4f})")
    print(f"hF1 médio: {np.mean(hF1_list):.4f} (+/- {np.std(hF1_list):.4f})")
    print(f"Acurácia Nível 1 média: {np.mean(acc_n1_list):.4f} "
          f"(+/- {np.std(acc_n1_list):.4f})")
    print(f"Acurácia Nível 2 média: {np.mean(acc_n2_list):.4f} "
          f"(+/- {np.std(acc_n2_list):.4f})")
    print("--- Validação Concluída com Sucesso ---")


if __name__ == "__main__":
    validar_robusto()