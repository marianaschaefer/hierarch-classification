import geopandas as gpd
import pandas as pd
import glob
import os

# Une os 4 tiles de data_raw/external sem olhar para nenhum CSV
tile_files = glob.glob("data_raw/external/tile*.shp")
gdfs = [gpd.read_file(f) for f in tile_files]
gdf_unido = pd.concat(gdfs).reset_index(drop=True)

# Salva o "Mapa Geral" de polígonos
gdf_unido.to_file("data_raw/processed/tile_completo_bruto.shp")
print(f"Mapa total de polígonos criado com {len(gdf_unido)} feições.")