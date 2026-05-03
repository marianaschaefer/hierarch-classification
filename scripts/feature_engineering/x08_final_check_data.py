import pandas as pd

path = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_COMPLETO_global_v1.csv"
df = pd.read_csv(path)

print(df.shape)
print("IDs únicos:", df["supercells_global"].nunique())
print("Total de NaN:", df.isna().sum().sum())
print(df["classe_nivel1"].value_counts())
print(df["classe_nivel2"].value_counts())