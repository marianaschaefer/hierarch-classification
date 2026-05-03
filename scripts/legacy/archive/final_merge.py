import pandas as pd

def realizar_merge_final():
    print("--- Unindo Embeddings (A00-A63) com Rótulos (Nível 1 e 2) ---")
    
    # 1. Carregar os Embeddings (X) - O que veio do GEE agora
    df_x = pd.read_csv('data_raw/Embeddings_2023_Final_Tese_A00_A63.csv')
    
    # 2. Carregar os Rótulos (y) - O que limpamos antes
    df_y = pd.read_csv('data_processed/dataset_FINAL_CLEAN_tese.csv')
    
    # 3. Merge por ID
    # [especulado] how='inner' remove qualquer polígono que não tenha ambos os dados
    df_final = pd.merge(df_x, df_y, on='supercells', how='inner')
    
    # 4. Salvar Dataset Mestre
    output_path = 'data_processed/X_y_train_hierarchical.csv'
    df_final.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"✅ DATASET PRONTO PARA O DOUTORADO!")
    print(f"Amostras finais: {len(df_final)}")
    print(f"Total de colunas: {len(df_final.columns)}")
    print(f"Arquivo gerado: {output_path}")

if __name__ == "__main__":
    realizar_merge_final()