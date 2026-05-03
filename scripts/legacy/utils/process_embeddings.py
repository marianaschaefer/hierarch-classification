import pandas as pd
from pathlib import Path

pasta = Path("/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_raw/external") 

arquivos = sorted(pasta.glob("Embeddings_2023_lote_*.csv"))

dfs = [pd.read_csv(a) for a in arquivos]
df_emb = pd.concat(dfs, ignore_index=True)

print("Linhas totais:", len(df_emb))
print("Supercells únicos:", df_emb["supercells"].nunique())

# Validação
duplicados = df_emb["supercells"].duplicated().sum()
print("Duplicatas:", duplicados)

# Limpar colunas inúteis
df_emb = df_emb.drop(columns=["system:index", ".geo"], errors="ignore")


# Carregar a base limpa
df_base = pd.read_csv("data_processed/dataset_FINAL_CLEAN_tese_v2.csv")

print("Base limpa:", len(df_base))
print("Supercells únicos:", df_base["supercells"].nunique())

# Merge final
df_final = df_base.merge(df_emb, on="supercells", how="inner")

print("Linhas após merge:", len(df_final))
print("Supercells únicos após merge:", df_final["supercells"].nunique())

# Verificar perdas
faltando = set(df_base["supercells"]) - set(df_emb["supercells"])
print("Sem embedding:", len(faltando))

# Salvar dataset
df_final.to_csv(
    "data_processed/dataset_FINAL_tese_com_embeddings.csv",
    index=False
)

print(df_final.shape)

df_final.iloc[:, -64:].describe()