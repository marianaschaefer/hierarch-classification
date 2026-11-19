"""Recompute the spectral-index statistics per polygon with the Sentinel-2 radiometric offset removed.

The L2A composites (processing baseline 04.00) store DN = 10000 * reflectance + 1000. The original attribute
table computed NDVI, NDBI, NDSI and EBBI* directly on DN. Here the bands are converted to reflectance,
(DN - 1000) / 10000, before computing the indices. The original (offset) values are also written, with the
prefix OLD_, so that the extraction can be checked against the deposited table (they agree to < 1e-6).

Band order of the composites: 1 blue (B2), 2 green (B3), 3 red (B4), 4-6 red edge (B5-B7), 7 nir (B8),
8 swir1 (B11), 9 swir2 (B12). Statistics follow ../../scripts/feature_engineering/x05_stats_ind_band_mde.py:
pixels whose centre falls inside the polygon (rasterio.mask, all_touched=False); if a polygon contains no pixel
centre, all_touched=True is used and flagged.

Usage (from the dataset root, the folder holding 02_raster/):
    python 20_extract_corrected_indices.py <dataset_root> <polygons.gpkg> <out.csv> <id_column>
Optional environment variables START/END process a slice of the polygons (to split long runs).
Manuscript: Sec. III-B (spectral indices), Supplementary S1.
"""
import os
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

warnings.filterwarnings("ignore")
DS, GP, OUT, IDCOL = sys.argv[1:5]
A, B = int(os.environ.get("START", 0)), int(os.environ.get("END", 10**9))
RASTERS = {"abr": f"{DS}/02_raster/mosaic_clip_composite_proj10m_abril.tif",
           "ago": f"{DS}/02_raster/mosaic_clip_composite_proj10m_agosto.tif"}
BANDS = [2, 3, 7, 8]          # green, red, nir (B8), swir1 (B11)
OFFSET, SCALE, EPS = 1000.0, 10000.0, 1e-10
STATS = ["min", "mediana", "max", "std", "p05", "p25", "p75", "p95"]


def clean(a):
    a = a.astype("float64").copy()
    a[~np.isfinite(a) | (np.abs(a) > 1e20)] = np.nan
    return a


def stats(a, prefix):
    v = a[np.isfinite(a)]
    if v.size == 0:
        return {f"{prefix}_{s}": np.nan for s in STATS}
    q = np.percentile(v, [5, 25, 75, 95])
    vals = [v.min(), np.median(v), v.max(), v.std(), *q]
    return {f"{prefix}_{s}": x for s, x in zip(STATS, vals)}


def indices(g, r, n, s, pre, out):
    out.update(stats((n - r) / (n + r + EPS), f"{pre}_ndvi"))
    out.update(stats((s - n) / (s + n + EPS), f"{pre}_ndbi"))
    out.update(stats((g - s) / (g + s + EPS), f"{pre}_ndsi"))
    out.update(stats((n - r) / (10.0 * np.sqrt(n + s + EPS)), f"{pre}_ebbi"))


G = gpd.read_file(GP).iloc[A:B]
rows = {i: {IDCOL: i} for i in G[IDCOL]}
for key, path in RASTERS.items():
    with rasterio.open(path) as src:
        Gl = G.to_crs(src.crs) if G.crs != src.crs else G
        for sid, geom in zip(Gl[IDCOL], Gl.geometry):
            for touched in (False, True):
                try:
                    im, _ = mask(src, [geom], crop=True, filled=True, all_touched=touched, indexes=BANDS)
                except Exception:
                    continue
                g, r, n, s = (clean(im[k]) for k in range(4))
                if np.isfinite(g).any():
                    break
            o = rows[sid]
            o[f"{key}_all_touched"] = touched
            indices(g, r, n, s, f"OLD_{key}", o)                                   # as in the original table
            indices(*[(x - OFFSET) / SCALE for x in (g, r, n, s)], key, o)        # corrected (reflectance)
pd.DataFrame(list(rows.values())).to_csv(OUT, index=False)
print("done", len(rows))
