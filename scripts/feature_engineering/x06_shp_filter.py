import geopandas as gpd
import pandas as pd
import os

BASE_DIR = "/home/mschaefer/projetos/LULC_Hierarchical_Classification"

path_shp = os.path.join(BASE_DIR, "data_raw", "external", "tiles_unificados_global.shp")
path_labels = os.path.join(BASE_DIR, "data_processed", "dataset_FINAL_CLEAN_tese_global_v1.csv")

output_path = os.path.join(BASE_DIR, "data_processed", "poligonos_finais_global_v1.shp")

print("Lendo shapefile completo...")
gdf = gpd.read_file(path_shp)

print("Lendo base limpa...")
df = pd.read_csv(path_labels)

ids_validos = set(df["supercells_global"])

print("Filtrando polígonos finais...")
gdf = gdf[gdf["supercel_1"].isin(ids_validos)].copy()

print("Dissolvendo...")
gdf = gdf.dissolve(by="supercel_1", as_index=False)

# renomear pra evitar truncamento ruim
gdf = gdf.rename(columns={"supercel_1": "sc_id"})

print("Salvando shapefile final...")
gdf.to_file(output_path)

print("Concluído.")
print("Linhas:", len(gdf))