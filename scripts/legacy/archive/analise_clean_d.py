import pandas as pd

df = pd.read_csv("data_processed/dataset_FINAL_CLEAN_tese_v2.csv")

print("\n=== Contagem por classe_nivel1 ===")
print(df["classe_nivel1"].value_counts())

print("\n=== Contagem por classe_nivel2 ===")
print(df["classe_nivel2"].value_counts())


import geopandas as gpd

gdf = gpd.read_file("data_raw/processed/tile_completo_bruto.shp")

print("Total feições:", len(gdf))
print("Supercells únicos:", gdf["supercells"].nunique())

duplicados = gdf["supercells"].value_counts()
print("\nQuantos IDs aparecem mais de 1 vez:")
print((duplicados > 1).sum())

import geopandas as gpd

gdf = gpd.read_file("data_raw/processed/tile_completo_bruto.shp")

# contar quantas vezes cada ID aparece
counts = gdf["supercells"].value_counts()

# pegar IDs que aparecem muitas vezes
suspeitos = counts[counts > 10]

print("IDs suspeitos (muitos fragmentos):")
print(suspeitos.head())

