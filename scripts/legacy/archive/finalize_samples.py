import geopandas as gpd
import pandas as pd

# 1. Carregar os pontos estratificados
df_pontos = pd.read_csv('data_raw/amostras_estratificadas_mapbiomas.csv')
gdf_pontos = gpd.GeoDataFrame(
    df_pontos, 
    geometry=gpd.points_from_xy(df_pontos.x, df_pontos.y),
    crs="EPSG:31983" # [não verificado] Verifique se o seu raster está em UTM 23S
)

# 2. Carregar o shapefile unido
gdf_poligonos = gpd.read_file('data_raw/processed/tile_completo.shp')

# 3. Spatial Join (Atribui o ID do polígono ao ponto)
# [não verificado] Substitua 'supercells' pelo nome da coluna de ID no seu SHP
gdf_final = gpd.sjoin(gdf_pontos, gdf_poligonos[['supercells', 'geometry']], how='inner', predicate='within')

# 4. Salvar o CSV definitivo para a Tese
gdf_final.drop(columns='geometry').to_csv('data_processed/dataset_final_com_ids.csv', index=False)

print(f"✅ Sucesso: {len(gdf_final)} amostras vinculadas aos polígonos.")