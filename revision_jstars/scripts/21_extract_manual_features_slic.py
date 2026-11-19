"""Extract all attributes of the manual samples over their SLIC objects, with the same procedure as Pipeline 2.

In the original analysis, the attributes of the 1,395 manual samples came from the Pipeline 1 table, extracted in R
(exactextractr, area-weighted, from a different raster version). Their dispersion statistics were compressed
relative to the Pipeline 2 attributes used to train the models (e.g. standard deviation of the visible bands about
30% lower), so the independent test compared models with test objects described in a different way.
This script links each manual sample to the SLIC object that contains its representative point and extracts the
216 attributes (9 bands x 2 dates, 4 corrected indices x 2 dates, elevation; 8 statistics each) plus slope,
exactly as in x05_stats_ind_band_mde.py and 20_extract_corrected_indices.py.

Usage (from the dataset root):
    python 21_extract_manual_features_slic.py <dataset_root> <out.csv>
Inputs: 04_amostras/amostras_manuais/Amostra_base_final.shp, 03_segmentacao/segmentacao_slic_completa.gpkg,
02_raster/ (mosaics, MDE.tif, SLOPE.tif). id_ = row number of Amostra_base_final.shp (1-based), as in Pipeline 1.
Manuscript: Sec. III-H, IV-E.
"""
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

warnings.filterwarnings("ignore")
DS, OUT = sys.argv[1:3]
BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "swir1", "swir2"]
OFFSET, SCALE, EPS = 1000.0, 10000.0, 1e-10
STATS = ["min", "mediana", "max", "std", "p05", "p25", "p75", "p95"]

# manual samples -> SLIC objects
m = gpd.read_file(f"{DS}/04_amostras/amostras_manuais/Amostra_base_final.shp").reset_index(drop=True)
m["id_"] = np.arange(1, len(m) + 1)
pts = gpd.GeoDataFrame(m[["id_"]], geometry=m.geometry.representative_point(), crs=m.crs)
seg = gpd.read_file(f"{DS}/03_segmentacao/segmentacao_slic_completa.gpkg", mask=pts.geometry.union_all().buffer(1))
j = gpd.sjoin(pts, seg, how="inner", predicate="within").drop_duplicates("id_")
G = gpd.GeoDataFrame({"id_": j.id_.values}, geometry=seg.loc[j.index_right].geometry.values, crs=seg.crs)
print("manual samples", len(m), "linked to SLIC objects", len(G))


def clean(a, nodata):
    a = a.astype("float64").copy()
    bad = ~np.isfinite(a) | (np.abs(a) > 1e20)
    if nodata is not None and abs(float(nodata)) < 1e20:
        bad |= a == nodata
    a[bad] = np.nan
    return a


def stats(a, prefix):
    v = a[np.isfinite(a)]
    if v.size == 0:
        return {f"{prefix}_{s}": np.nan for s in STATS}
    q = np.percentile(v, [5, 25, 75, 95])
    return {f"{prefix}_{s}": x for s, x in zip(STATS, [v.min(), np.median(v), v.max(), v.std(), *q])}


rows = {i: {"id_": i} for i in G.id_}
RASTERS = {"abr": f"{DS}/02_raster/mosaic_clip_composite_proj10m_abril.tif",
           "ago": f"{DS}/02_raster/mosaic_clip_composite_proj10m_agosto.tif",
           "mde": f"{DS}/02_raster/MDE.tif", "slope": f"{DS}/02_raster/SLOPE.tif"}
for key, path in RASTERS.items():
    with rasterio.open(path) as src:
        Gl = G.to_crs(src.crs) if G.crs != src.crs else G
        for sid, geom in zip(Gl.id_, Gl.geometry):
            arrs = None
            for touched in (False, True):
                try:
                    im, _ = mask(src, [geom], crop=True, filled=True, all_touched=touched)
                except Exception:
                    continue
                arrs = [clean(im[b], src.nodata) for b in range(im.shape[0])]
                if np.isfinite(arrs[0]).any():
                    break
            if arrs is None:
                continue
            o = rows[sid]
            o[f"{key}_all_touched"] = touched
            if key in ("abr", "ago"):
                for b, name in enumerate(BANDS):
                    o.update(stats(arrs[b], f"{key}_{name}"))
                g, r, n, s = [(arrs[i] - OFFSET) / SCALE for i in (1, 2, 6, 7)]
                o.update(stats((n - r) / (n + r + EPS), f"{key}_ndvi"))
                o.update(stats((s - n) / (s + n + EPS), f"{key}_ndbi"))
                o.update(stats((g - s) / (g + s + EPS), f"{key}_ndsi"))
                o.update(stats((n - r) / (10.0 * np.sqrt(n + s + EPS)), f"{key}_ebbi"))
            elif key == "mde":
                o.update(stats(arrs[0], "mde_elev"))
            else:
                o.update(stats(arrs[0], "slope"))
pd.DataFrame(list(rows.values())).to_csv(OUT, index=False)
print("done", len(rows))
