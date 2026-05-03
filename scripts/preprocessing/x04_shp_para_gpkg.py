import geopandas as gpd
import pandas as pd
import os

BASE_DIR = "/home/mschaefer/projetos/LULC_Hierarchical_Classification"

path_shp = os.path.join(BASE_DIR, "data_raw", "external", "tiles_unificados_global.shp")
path_labels = os.path.join(BASE_DIR, "data_processed", "dataset_FINAL_CLEAN_tese_global_v1.csv")
output_path = os.path.join(BASE_DIR, "data_processed", "poligonos_finais_global_v1.gpkg")

print("Lendo shapefile unificado...")
gdf = gpd.read_file(path_shp)

print("Lendo base limpa...")
df = pd.read_csv(path_labels)

ids_validos = set(df["supercells_global"].tolist())

print("Filtrando polígonos válidos...")
gdf = gdf[gdf["supercel_1"].isin(ids_validos)].copy()

print("Dissolvendo por supercells_global...")
gdf_diss = gdf[["supercel_1", "geometry"]].dissolve(by="supercel_1", as_index=False)

gdf_diss = gdf_diss.rename(columns={"supercel_1": "supercells_global"})

print("Juntando rótulos...")
gdf_diss = gdf_diss.merge(df, on="supercells_global", how="inner")

print("Salvando GeoPackage...")
gdf_diss.to_file(output_path, driver="GPKG")

print("Concluído.")
print("Linhas:", len(gdf_diss))
print("IDs únicos:", gdf_diss["supercells_global"].nunique())
print("Arquivo salvo em:", output_path)