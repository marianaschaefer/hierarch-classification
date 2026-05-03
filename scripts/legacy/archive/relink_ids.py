import geopandas as gpd
import pandas as pd

# 1. Carregar o arquivo de amostras que você acabou de achar
df_pontos = pd.read_csv('amostras_estratificadas_mapbiomas.csv')
# Converter para GeoDataFrame usando as colunas x e y
gdf_pontos = gpd.GeoDataFrame(
    df_pontos, geometry=gpd.points_from_xy(df_pontos.x, df_pontos.y), crs="EPSG:31983" # Ajuste o CRS se for o caso
)

# 2. Carregar os tiles unidos (o que você vai reconstruir)
gdf_poligonos = gpd.read_file('data_raw/processed/tile_completo.shp')

# 3. Spatial Join: Descobrir o ID do polígono para cada ponto de amostra
# Isso garante que o polygon_id seja o ATUAL e CORRETO
gdf_validado = gpd.sjoin(gdf_pontos, gdf_poligonos[['supercells', 'geometry']], how='left', predicate='within')

# 4. Salvar o novo dataset de referência
gdf_validado.to_csv('data_raw/processed/dataset_amostras_validado.csv', index=False)