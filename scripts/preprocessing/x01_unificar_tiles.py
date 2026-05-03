import geopandas as gpd
import os

BASE_DIR = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_raw/external"

tiles = [
    ("tile1", "tile1_compac05_area08_step15.shp"),
    ("tile2", "tile2_compac05_area08_step15.shp"),
    ("tile3", "tile3_compac05_area08_step15.shp"),
    ("tile4", "tile4_compac05_area08_step15.shp"),
]

gdfs = []

for tile_id, shp_name in tiles:
    path = os.path.join(BASE_DIR, shp_name)
    print(f"Lendo {path}...")

    gdf = gpd.read_file(path)

    if "supercells" not in gdf.columns:
        raise ValueError(f"{shp_name} não possui coluna 'supercells'")

    # criar ID global
    gdf["supercells_global"] = tile_id + "_" + gdf["supercells"].astype(str)

    gdf["tile_id"] = tile_id

    gdfs.append(gdf)

# unir tudo
gdf_final = gpd.GeoDataFrame(
    gpd.pd.concat(gdfs, ignore_index=True),
    crs=gdfs[0].crs
)

print("\nResumo:")
print("Total feições:", len(gdf_final))
print("IDs únicos:", gdf_final["supercells_global"].nunique())

# salvar
output_path = os.path.join(BASE_DIR, "tiles_unificados_global.shp")
gdf_final.to_file(output_path)

print(f"\nArquivo salvo em: {output_path}")