import pandas as pd

df = pd.read_csv("/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_CLEAN_tese_global_v1.csv")

print("Linhas:", len(df))
print("IDs únicos:", df["supercells_global"].nunique())

print("\n=== Contagem por classe_nivel1 ===")
print(df["classe_nivel1"].value_counts())

print("\n=== Contagem por classe_nivel2 ===")
print(df["classe_nivel2"].value_counts())