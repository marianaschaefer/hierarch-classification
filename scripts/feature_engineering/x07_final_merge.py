import pandas as pd

stats_path = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_ESTATISTICAS_POLIGONOS_global_v1.csv"
emb_path = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_raw/external/Embeddings_2023_global_v1.csv"
output_path = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1.csv"

# =========================
# 1. Leitura
# =========================
df_stats = pd.read_csv(stats_path)
df_emb = pd.read_csv(emb_path)

print("Stats:", df_stats.shape)
print("Embeddings (bruto):", df_emb.shape)

# =========================
# 2. Garantir chave correta
# =========================
if "supercells_global" not in df_stats.columns:
    raise ValueError("A coluna 'supercells_global' não está no CSV de estatísticas.")

if "sc_id" not in df_emb.columns:
    raise ValueError("A coluna 'sc_id' não está no CSV de embeddings.")

# Converter para string (evita mismatch silencioso)
df_stats["supercells_global"] = df_stats["supercells_global"].astype(str)
df_emb["sc_id"] = df_emb["sc_id"].astype(str)

# =========================
# 3. Manter apenas embeddings válidos
# =========================
cols_emb = ["sc_id"] + [f"A{i:02d}" for i in range(64)]

faltando_cols = [c for c in cols_emb if c not in df_emb.columns]
if faltando_cols:
    raise ValueError(f"Colunas de embedding faltando: {faltando_cols}")

df_emb = df_emb[cols_emb].copy()

print("Embeddings (limpo):", df_emb.shape)

# =========================
# 4. Merge
# =========================
df_final = df_stats.merge(
    df_emb,
    left_on="supercells_global",
    right_on="sc_id",
    how="inner"
)

# =========================
# 5. Validação
# =========================
print("\n=== Validação ===")
print("Shape final:", df_final.shape)
print("IDs únicos stats:", df_stats["supercells_global"].nunique())
print("IDs únicos embeddings:", df_emb["sc_id"].nunique())
print("IDs únicos final:", df_final["supercells_global"].nunique())

faltando = set(df_stats["supercells_global"]) - set(df_emb["sc_id"])
print("Sem embedding:", len(faltando))

duplicados = df_final["supercells_global"].duplicated().sum()
print("Duplicados no final:", duplicados)

# =========================
# 6. Salvar
# =========================
df_final.to_csv(output_path, index=False)

print("\nArquivo salvo em:", output_path)