import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import xy


mapbiomas_raster = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_raw/external/mapbiomas_todasclasses.tif"

# === CONFIGURAÇÕES ===
tamanho_celula = 2000  # tamanho da célula da grade (ex.: 2000 = 2km)
n_amostras_por_celula = 1  # número de amostras por célula por classe

# === DICIONÁRIO DE CLASSES E SUBCLASSES ===
classes = {
    'Afloramento Rochoso': {'codigos': [29]},
    'Vegetação Rasteira': {
        'codigos': [12, 15, 25, 4, 39, 41, 46, 48, 21],
        'subclasses': {
            'Formação Campestre': [12],
            'Pastagem': [15],
            'Solo Exposto': [25],
            'Cerrado': [4],
            'Agricultura': [21, 39, 41, 46, 48]
        }
    },
    'Mineração': {'codigos': [30]},
    'Área Urbana': {'codigos': [24]},
    'Eucalipto': {'codigos': [9]},
    'Mata': {'codigos': [3, 5, 6, 49]},
    'Rios e Lagoas': {'codigos': [31, 33]}
}

print("DEBUG CLASSES MATA:", classes['Mata'])

# === FUNÇÃO: criar grade ===
def criar_grade(raster_path, tamanho_celula):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    xmin, ymin, xmax, ymax = bounds

    linhas = np.arange(ymin, ymax, tamanho_celula)
    colunas = np.arange(xmin, xmax, tamanho_celula)

    caixas = []
    for x in colunas:
        for y in linhas:
            caixas.append(box(x, y, x + tamanho_celula, y + tamanho_celula))

    grid = gpd.GeoDataFrame({'geometry': caixas}, crs=crs)
    return grid


# === FUNÇÃO: amostragem estratificada ===
def amostragem_estratificada_multiclasse(raster_path, classes_dict, grid, n_por_celula=1):
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        rows, cols = data.shape

        amostras = []

        for classe_nivel1, info in classes_dict.items():
            codigos = info['codigos']

            if 'subclasses' in info:
                # Subclasses presentes
                for subclasse_nome, sub_codigos in info['subclasses'].items():
                    for _, cell in grid.iterrows():
                        minx, miny, maxx, maxy = cell.geometry.bounds

                        row_min, col_min = ~transform * (minx, maxy)
                        row_max, col_max = ~transform * (maxx, miny)

                        row_min, row_max = int(np.clip(np.floor(row_min), 0, rows-1)), int(np.clip(np.ceil(row_max), 0, rows-1))
                        col_min, col_max = int(np.clip(np.floor(col_min), 0, cols-1)), int(np.clip(np.ceil(col_max), 0, cols-1))

                        window = data[row_min:row_max+1, col_min:col_max+1]

                        mask = np.isin(window, sub_codigos)
                        indices = np.argwhere(mask)

                        if len(indices) == 0:
                            continue

                        n = min(n_por_celula, len(indices))
                        selecionados = indices[np.random.choice(indices.shape[0], n, replace=False)]

                        for row_offset, col_offset in selecionados:
                            row = row_min + row_offset
                            col = col_min + col_offset
                            x, y = xy(transform, row, col)

                            amostras.append({
                                'x': x,
                                'y': y,
                                'classe_nivel1': classe_nivel1,
                                'classe_nivel2': subclasse_nome
                            })

            else:
                # Sem subclasses
                for _, cell in grid.iterrows():
                    minx, miny, maxx, maxy = cell.geometry.bounds

                    row_min, col_min = ~transform * (minx, maxy)
                    row_max, col_max = ~transform * (maxx, miny)

                    row_min, row_max = int(np.clip(np.floor(row_min), 0, rows-1)), int(np.clip(np.ceil(row_max), 0, rows-1))
                    col_min, col_max = int(np.clip(np.floor(col_min), 0, cols-1)), int(np.clip(np.ceil(col_max), 0, cols-1))

                    window = data[row_min:row_max+1, col_min:col_max+1]

                    mask = np.isin(window, codigos)
                    indices = np.argwhere(mask)

                    if len(indices) == 0:
                        continue

                    n = min(n_por_celula, len(indices))
                    selecionados = indices[np.random.choice(indices.shape[0], n, replace=False)]

                    for row_offset, col_offset in selecionados:
                        row = row_min + row_offset
                        col = col_min + col_offset
                        x, y = xy(transform, row, col)

                        amostras.append({
                            'x': x,
                            'y': y,
                            'classe_nivel1': classe_nivel1,
                            'classe_nivel2': classe_nivel1
                        })

        df = pd.DataFrame(amostras)
        return df


# === EXECUTAR ===

print('🔧 Criando grade...')
grid = criar_grade(mapbiomas_raster, tamanho_celula)

print('🚀 Iniciando amostragem...')
df_amostras = amostragem_estratificada_multiclasse(
    raster_path=mapbiomas_raster,
    classes_dict=classes,
    grid=grid,
    n_por_celula=n_amostras_por_celula
)

print(f"✅ Total de amostras coletadas: {len(df_amostras)}")
print(df_amostras.head())

# === SALVAR CSV ===
df_amostras.to_csv('data_raw/amostras_estratificadas_mapbiomas.csv', index=False)
print("💾 Arquivo salvo: amostras_estratificadas_mapbiomas.csv")


print("\n=== Contagem por classe nível 1 ===")
print(df_amostras['classe_nivel1'].value_counts())

print("\n=== Contagem por classe nível 2 ===")
print(df_amostras['classe_nivel2'].value_counts())