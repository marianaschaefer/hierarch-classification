import pandas as pd
from pathlib import Path

pasta = Path("/caminho/onde/estao/os/csvs")
arquivos = sorted(pasta.glob("Embeddings_2023_lote_*.csv"))

dfs = [pd.read_csv(arq) for arq in arquivos]
df_emb = pd.concat(dfs, ignore_index=True)

print("Linhas totais embeddings:", len(df_emb))
print("Supercells únicos embeddings:", df_emb["supercells"].nunique())

duplicados = df_emb["supercells"].duplicated().sum()
print("Duplicatas de supercells:", duplicados)

df_base = pd.read_csv("data_processed/dataset_FINAL_CLEAN_tese_v2.csv")

print("Linhas base limpa:", len(df_base))
print("Supercells únicos base limpa:", df_base["supercells"].nunique())

df_final = df_base.merge(df_emb, on="supercells", how="inner")

print("Linhas após merge:", len(df_final))
print("Supercells únicos após merge:", df_final["supercells"].nunique())

faltando_base = set(df_base["supercells"]) - set(df_emb["supercells"])
faltando_emb = set(df_emb["supercells"]) - set(df_base["supercells"])

print("Supercells da base sem embedding:", len(faltando_base))
print("Supercells dos embeddings fora da base:", len(faltando_emb))

df_final.to_csv("data_processed/dataset_FINAL_CLEAN_tese_v2_com_embeddings.csv", index=False)
print("Arquivo salvo: data_processed/dataset_FINAL_CLEAN_tese_v2_com_embeddings.csv")