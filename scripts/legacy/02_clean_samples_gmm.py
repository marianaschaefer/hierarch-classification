import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import os

def limpar_com_gmm(df, threshold=0.01):
    print(f"--- Iniciando Limpeza GMM (Threshold: {threshold}) ---")
    df_limpo = pd.DataFrame()
    
    # Selecionamos apenas as colunas de dados (Embeddings + Índices + MDE)
    # Excluímos colunas de ID e Target
    cols_features = [c for c in df.columns if c not in ['supercells', 'classe_nivel1', 'classe_nivel2']]
    
    for classe in df['classe_nivel1'].unique():
        subgroup = df[df['classe_nivel1'] == classe].copy()
        n_samples = len(subgroup)
        
        if n_samples < 10: # Evita erro em classes com pouquíssimas amostras
            df_limpo = pd.concat([df_limpo, subgroup])
            continue
            
        # [especulado] Ajustamos o GMM para encontrar a distribuição principal da classe
        # Usamos 2 componentes: 1 para a classe "pura" e 1 para capturar o ruído
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(subgroup[cols_features])
        
        # Calculamos a probabilidade de cada amostra pertencer à distribuição
        scores = gmm.score_samples(subgroup[cols_features])
        
        # Filtramos os outliers (menores scores = menor probabilidade)
        limiar_corte = np.quantile(scores, threshold)
        subgroup_filtrado = subgroup[scores > limiar_corte]
        
        print(f"Classe {classe:20} | Original: {n_samples:5} | Limpo: {len(subgroup_filtrado):5} | Removidos: {n_samples - len(subgroup_filtrado)}")
        df_limpo = pd.concat([df_limpo, subgroup_filtrado])
        
    return df_limpo

if __name__ == "__main__":
    path_input = 'data_processed/dataset_MULTIMODAL_FINAL.csv'
    path_output = 'data_processed/dataset_MULTIMODAL_LIMPO.csv'
    
    if os.path.exists(path_input):
        df = pd.read_csv(path_input)
        df_final = limpar_com_gmm(df)
        df_final.to_csv(path_output, index=False)
        print(f"\n✅ Limpeza concluída! Novo dataset salvo em: {path_output}")
    else:
        print("❌ Erro: Dataset original não encontrado.")