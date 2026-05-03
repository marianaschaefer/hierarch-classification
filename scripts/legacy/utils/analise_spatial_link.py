import pandas as pd

df = pd.read_csv("/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_ESTATISTICAS_POLIGONOS_global_v1.csv")

print(df.shape)
print(df.columns[:20].tolist())
print(df.columns[-20:].tolist())
print(df["supercells_global"].nunique())