import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = "/home/mschaefer/projetos/LULC_Hierarchical_Classification"

PATH_GPKG = os.path.join(BASE_DIR, "data_processed", "poligonos_finais_global_v1.gpkg")

RASTERS = {
    "abr": os.path.join(BASE_DIR, "data_raw", "external", "mosaic_clip_composite_proj10m_abril.tif"),
    "ago": os.path.join(BASE_DIR, "data_raw", "external", "mosaic_clip_composite_proj10m_agosto.tif"),
    "mde": os.path.join(BASE_DIR, "data_raw", "external", "MDE.tif"),
}

OUTPUT_PATH = os.path.join(BASE_DIR, "data_processed", "dataset_ESTATISTICAS_POLIGONOS_global_v1.csv")

BAND_NAMES = [
    "blue", "green", "red", "rededge1", "rededge2",
    "rededge3", "nir", "nirn", "swir1"
]

EPS = 1e-10


def aplicar_mascara_nodata(array: np.ndarray, nodata) -> np.ndarray:
    arr = array.astype("float32").copy()

    mask_invalid = ~np.isfinite(arr)

    if nodata is not None:
        # Para valores extremos como -1.797693e+308
        if abs(float(nodata)) > 1e20:
            mask_invalid |= arr < -1e20
        else:
            mask_invalid |= arr == nodata

    # Segurança adicional
    mask_invalid |= (arr < -1e20) | (arr > 1e20)

    arr[mask_invalid] = np.nan
    return arr

def calcular_estatisticas(array: np.ndarray, prefixo: str) -> dict:
    # filtrar valores inválidos corretamente
    vals = array[np.isfinite(array)]

    # remover valores absurdos (nodata residual)
    vals = vals[(vals > -1e10) & (vals < 1e10)]

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

def extrair_estatisticas():
    print("=== Início da extração por polígono ===")

    gdf = gpd.read_file(PATH_GPKG)

    print("Linhas:", len(gdf))
    print("IDs únicos:", gdf["supercells_global"].nunique())

    data_map = {
        row["supercells_global"]: {
            "supercells_global": row["supercells_global"],
            "classe_nivel1": row["classe_nivel1"],
            "classe_nivel2": row["classe_nivel2"],
        }
        for _, row in gdf.iterrows()
    }

    for raster_key, raster_path in RASTERS.items():
        print(f"\nProcessando raster: {raster_key}")

        with rasterio.open(raster_path) as src:
            nodata_val = src.nodata

            gdf_local = gdf.copy()
            if gdf_local.crs != src.crs:
                print(f"Reprojetando de {gdf_local.crs} para {src.crs}")
                gdf_local = gdf_local.to_crs(src.crs)

            total = len(gdf_local)

            for i, row in gdf_local.iterrows():
                sid = row["supercells_global"]
                geom = [row["geometry"]]

                if (i + 1) % 250 == 0 or i == 0 or (i + 1) == total:
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

    df_final = pd.DataFrame(list(data_map.values()))
    df_final.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Concluído ===")
    print("Shape:", df_final.shape)
    print("IDs únicos:", df_final["supercells_global"].nunique())
    print("Arquivo salvo em:", OUTPUT_PATH)


if __name__ == "__main__":
    extrair_estatisticas()