import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from collections import Counter

# =========================
# PATHS
# =========================

path_gpkg = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/poligonos_finais_global_v1.gpkg"
path_raster = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_raw/external/mapbiomas_todasclasses.tif"

output = "/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/pureza_poligonos_mapbiomas.csv"

# =========================
# DICIONÁRIO MAPBIOMAS -> CLASSES DO ESTUDO
# =========================

mapbiomas_to_label = {
    3: "Mata",
    5: "Mata",
    6: "Mata",
    49: "Mata",

    4: "Cerrado",
    12: "Formação Campestre",
    15: "Pastagem",
    25: "Solo Exposto",
    21: "Agricultura",
    39: "Agricultura",
    41: "Agricultura",
    46: "Agricultura",
    48: "Agricultura",

    9: "Eucalipto",
    24: "Área Urbana",
    29: "Afloramento Rochoso",
    30: "Mineração",
    31: "Rios e Lagoas",
    33: "Rios e Lagoas",
}

# =========================
# LOAD
# =========================

gdf = gpd.read_file(path_gpkg)

print("Polígonos:", len(gdf))
print("CRS polígonos:", gdf.crs)

resultados = []

with rasterio.open(path_raster) as src:
    print("CRS raster:", src.crs)
    print("Nodata raster:", src.nodata)

    nodata = src.nodata

    if gdf.crs != src.crs:
        print("Reprojetando polígonos para CRS do raster...")
        gdf = gdf.to_crs(src.crs)

    for i, row in gdf.iterrows():
        if i % 500 == 0:
            print(f"{i}/{len(gdf)}")

        geom = [row.geometry]

        try:
            out_image, _ = mask(src, geom, crop=True, filled=True)
            data = out_image[0].ravel()

            # remover nodata
            if nodata is not None:
                data = data[data != nodata]

            # remover valores inválidos
            data = data[np.isfinite(data)]

            if data.size == 0:
                resultados.append({
                    "supercells_global": row["supercells_global"],
                    "classe_label": row["classe_nivel2"],
                    "classe_dominante": None,
                    "prop_dominante": np.nan,
                    "prop_label": np.nan,
                    "n_pixels_validos": 0,
                    "n_pixels_mapeados": 0,
                    "flag_sem_pixels": True
                })
                continue

            # converter códigos MapBiomas para rótulos do estudo
            labels = []
            for v in data:
                try:
                    codigo = int(v)
                    classe = mapbiomas_to_label.get(codigo, None)
                    if classe is not None:
                        labels.append(classe)
                except Exception:
                    continue

            if len(labels) == 0:
                resultados.append({
                    "supercells_global": row["supercells_global"],
                    "classe_label": row["classe_nivel2"],
                    "classe_dominante": None,
                    "prop_dominante": np.nan,
                    "prop_label": np.nan,
                    "n_pixels_validos": int(data.size),
                    "n_pixels_mapeados": 0,
                    "flag_sem_pixels": False
                })
                continue

            counts = Counter(labels)
            total_mapeado = sum(counts.values())

            classe_dominante, count_dominante = counts.most_common(1)[0]
            prop_dominante = count_dominante / total_mapeado

            classe_label = row["classe_nivel2"]
            prop_label = counts.get(classe_label, 0) / total_mapeado

            resultados.append({
                "supercells_global": row["supercells_global"],
                "classe_label": classe_label,
                "classe_dominante": classe_dominante,
                "prop_dominante": prop_dominante,
                "prop_label": prop_label,
                "n_pixels_validos": int(data.size),
                "n_pixels_mapeados": int(total_mapeado),
                "flag_sem_pixels": False
            })

        except Exception as e:
            resultados.append({
                "supercells_global": row["supercells_global"],
                "classe_label": row["classe_nivel2"],
                "classe_dominante": None,
                "prop_dominante": np.nan,
                "prop_label": np.nan,
                "n_pixels_validos": 0,
                "n_pixels_mapeados": 0,
                "flag_sem_pixels": True,
                "erro": str(e)
            })

df = pd.DataFrame(resultados)

# =========================
# FLAGS DE PUREZA
# =========================

df["label_igual_dominante"] = df["classe_label"] == df["classe_dominante"]

df["pureza_60"] = df["prop_label"] >= 0.60
df["pureza_70"] = df["prop_label"] >= 0.70
df["pureza_80"] = df["prop_label"] >= 0.80
df["pureza_90"] = df["prop_label"] >= 0.90

# =========================
# SALVAR
# =========================

df.to_csv(output, index=False)

print("\nArquivo salvo em:", output)

print("\nShape:", df.shape)

print("\nResumo prop_dominante:")
print(df["prop_dominante"].describe())

print("\nResumo prop_label:")
print(df["prop_label"].describe())

print("\nLabel igual dominante:")
print(df["label_igual_dominante"].value_counts(dropna=False))

print("\nQuantidade por limiar de pureza do label:")
print(">= 60%:", df["pureza_60"].sum())
print(">= 70%:", df["pureza_70"].sum())
print(">= 80%:", df["pureza_80"].sum())
print(">= 90%:", df["pureza_90"].sum())

print("\nMédia de prop_label por classe:")
print(df.groupby("classe_label")["prop_label"].mean().sort_values(ascending=False))

print("\nMediana de prop_label por classe:")
print(df.groupby("classe_label")["prop_label"].median().sort_values(ascending=False))