import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = "/home/mschaefer/projetos/LULC_Hierarchical_Classification"

PATH_SHP = os.path.join(BASE_DIR, "data_raw", "processed", "tile_completo_bruto.shp")
PATH_LABELS = os.path.join(BASE_DIR, "data_processed", "dataset_FINAL_CLEAN_tese_v2.csv")

RASTERS = {
    "abr": os.path.join(BASE_DIR, "data_raw", "external", "mosaic_clip_composite_proj10m_abril.tif"),
    "ago": os.path.join(BASE_DIR, "data_raw", "external", "mosaic_clip_composite_proj10m_agosto.tif"),
    "mde": os.path.join(BASE_DIR, "data_raw", "external", "MDE.tif"),
}

OUTPUT_DIR = os.path.join(BASE_DIR, "data_processed", "estatisticas_lotes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BAND_NAMES = [
    "blue", "green", "red", "rededge1", "rededge2",
    "rededge3", "nir", "nirn", "swir1"
]

EPS = 1e-10


def aplicar_mascara_nodata(array: np.ndarray, nodata) -> np.ndarray:
    arr = array.astype("float32").copy()
    if nodata is not None:
        mask_invalid = np.isclose(arr, nodata)
    else:
        mask_invalid = ~np.isfinite(arr) | (arr < -1e10) | (arr > 1e10)
    arr[mask_invalid] = np.nan
    return arr


def calcular_estatisticas(array: np.ndarray, prefixo: str) -> dict:
    vals = array[np.isfinite(array)]
    if vals.size == 0:
        return {
            f"{prefixo}_min": np.nan,
            f"{prefixo}_mediana": np.nan,
            f"{prefixo}_max": np.nan,
            f"{prefixo}_std": np.nan,
            f"{prefixo}_p05": np.nan,
            f"{prefixo}_p25": np.nan,
            f"{prefixo}_p75": np.nan,
            f"{prefixo}_p95": np.nan,
        }

    return {
        f"{prefixo}_min": float(np.min(vals)),
        f"{prefixo}_mediana": float(np.median(vals)),
        f"{prefixo}_max": float(np.max(vals)),
        f"{prefixo}_std": float(np.std(vals)),
        f"{prefixo}_p05": float(np.percentile(vals, 5)),
        f"{prefixo}_p25": float(np.percentile(vals, 25)),
        f"{prefixo}_p75": float(np.percentile(vals, 75)),
        f"{prefixo}_p95": float(np.percentile(vals, 95)),
    }


def carregar_poligonos_dissolvidos():
    print("Lendo shapefile bruto...")
    gdf = gpd.read_file(PATH_SHP)

    print("Lendo base limpa...")
    df_labels = pd.read_csv(PATH_LABELS)

    ids_validos = set(df_labels["supercells"].tolist())
    gdf = gdf[gdf["supercells"].isin(ids_validos)].copy()

    print(f"Feições após filtro: {len(gdf)}")
    print(f"Supercells únicos antes do dissolve: {gdf['supercells'].nunique()}")

    gdf_diss = gdf[["supercells", "geometry"]].dissolve(by="supercells", as_index=False)

    print(f"Polígonos após dissolve: {len(gdf_diss)}")

    gdf_diss = gdf_diss.merge(
        df_labels[["supercells", "classe_nivel1", "classe_nivel2"]],
        on="supercells",
        how="inner"
    )

    print(f"Polígonos finais com rótulos: {len(gdf_diss)}")
    return gdf_diss


def processar_lote(gdf_lote: gpd.GeoDataFrame, numero_lote: int):
    print(f"\n=== Processando lote {numero_lote} com {len(gdf_lote)} polígonos ===")

    data_map = {
        row["supercells"]: {
            "supercells": row["supercells"],
            "classe_nivel1": row["classe_nivel1"],
            "classe_nivel2": row["classe_nivel2"],
        }
        for _, row in gdf_lote.iterrows()
    }

    for raster_key, raster_path in RASTERS.items():
        print(f"Raster: {raster_key}")

        with rasterio.open(raster_path) as src:
            nodata_val = src.nodata

            gdf_local = gdf_lote.copy()
            if gdf_local.crs != src.crs:
                gdf_local = gdf_local.to_crs(src.crs)

            total = len(gdf_local)

            for i, row in gdf_local.iterrows():
                sid = row["supercells"]
                geom = [row["geometry"]]

                if (i + 1) % 100 == 0 or i == 0 or (i + 1) == total:
                    print(f"  {raster_key}: {i + 1}/{total}")

                try:
                    out_image, _ = mask(src, geom, crop=True, filled=True)
                except Exception:
                    continue

                if raster_key in ("abr", "ago"):
                    bandas = []
                    for b_idx, b_name in enumerate(BAND_NAMES):
                        band_data = aplicar_mascara_nodata(out_image[b_idx], nodata_val)
                        bandas.append(band_data)
                        data_map[sid].update(
                            calcular_estatisticas(band_data, f"{raster_key}_{b_name}")
                        )

                    green = bandas[1]
                    red = bandas[2]
                    nir = bandas[6]
                    swir1 = bandas[8]

                    ndvi = (nir - red) / (nir + red + EPS)
                    ndbi = (swir1 - nir) / (swir1 + nir + EPS)
                    ndsi = (green - swir1) / (green + swir1 + EPS)
                    ebbi = (nir - red) / (10.0 * np.sqrt(nir + swir1 + EPS))

                    data_map[sid].update(calcular_estatisticas(ndvi, f"{raster_key}_ndvi"))
                    data_map[sid].update(calcular_estatisticas(ndbi, f"{raster_key}_ndbi"))
                    data_map[sid].update(calcular_estatisticas(ndsi, f"{raster_key}_ndsi"))
                    data_map[sid].update(calcular_estatisticas(ebbi, f"{raster_key}_ebbi"))

                elif raster_key == "mde":
                    elev = aplicar_mascara_nodata(out_image[0], nodata_val)
                    data_map[sid].update(calcular_estatisticas(elev, "mde_elev"))

    df_lote = pd.DataFrame(list(data_map.values()))
    output_path = os.path.join(OUTPUT_DIR, f"estatisticas_lote_{numero_lote}.csv")
    df_lote.to_csv(output_path, index=False)

    print(f"Lote {numero_lote} salvo em: {output_path}")
    print(f"Shape do lote: {df_lote.shape}")


if __name__ == "__main__":
    gdf = carregar_poligonos_dissolvidos()

    tamanho_lote = 500   
    numero_lote = 0      # mudar: 0, 1, 2, 3...

    inicio = numero_lote * tamanho_lote
    fim = min(inicio + tamanho_lote, len(gdf))

    gdf_lote = gdf.iloc[inicio:fim].copy()
    processar_lote(gdf_lote, numero_lote)