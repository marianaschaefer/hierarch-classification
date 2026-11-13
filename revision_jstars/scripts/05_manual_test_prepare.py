"""E2 (preparação): features dos polígonos MANUAIS no padrão de nomes do Pipeline 2 (216 atributos) + geometria + checagens."""
import sys, re, numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import load, UP
Z = f"{UP}/QFGeo-LULC Dataset/QFGeo-LULC Dataset"
d, feats = load(); f216 = [c for c in feats if not re.fullmatch(r"A\d\d", c)]
print("P2 handcrafted:", len(f216))
s = pd.read_csv(f"{Z}/06_atributos_pipeline1/estatisticas_completas/estatisticas_poligonos_pipeline1.csv")
STAT = {"min":"min","median":"mediana","max":"max","stdev":"std","q05":"p05","q25":"p25","q75":"p75","q95":"p95"}
PER = {"abril":"abr","agost":"ago"}
BAND = {"blue":"blue","verde":"green","red":"red","red_edge1":"rededge1","red_edge2":"rededge2","red_edge3":"rededge3","nir":"nir","swir1":"swir1","swir2":"swir2",
        "ndvi":"ndvi","NDVI":"ndvi","NDBI":"ndbi","NDSI":"ndsi","EBBI":"ebbi"}
ren = {}
for c in s.columns[2:]:
    st, rest = c.split(".", 1)
    if rest == "MDE": ren[c] = f"mde_elev_{STAT[st]}"; continue
    if rest == "slope": continue
    m = re.fullmatch(r"(?:(abril|agost)_sentinel_(\w+)|(NDBI|NDSI|EBBI)_(abril|agost)_sentinel)", rest)
    per, band = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
    ren[c] = f"{PER[per]}_{BAND[band]}_{STAT[st]}"
M = s[["id_", "Classe_Fin"] + list(ren)].rename(columns=ren)
miss = set(f216) - set(M.columns); extra = set(M.columns[2:]) - set(f216)
print("faltando no manual:", sorted(miss)[:5], len(miss), "| sobrando:", len(extra))
# geometria: id_ = índice de linha do shapefile? validar pela classe
g = gpd.read_file(f"{Z}/04_amostras/amostras_manuais/Amostra_base_final.shp").reset_index(drop=True)
g["id_"] = np.arange(1, len(g) + 1)
j = M.merge(g[["id_", "Classe_Fin", "geometry"]], on="id_", suffixes=("", "_shp"))
print("id_ -> classe confere com o shapefile:", (j.Classe_Fin == j.Classe_Fin_shp).mean().round(4), "de", len(j))

# ---------- E2 completo: teste independente nos polígonos manuais (7 classes) ----------
import os
j = gpd.GeoDataFrame(j.drop(columns="Classe_Fin_shp"), geometry="geometry", crs="EPSG:31983")
c = j.geometry.centroid; j["cx"], j["cy"] = c.x.values, c.y.values
CLS7 = {"mata":"Mata","eucalipto":"Eucalipto","vegetação rasteira":"Vegetação Rasteira","mineração":"Mineração",
        "area urbana":"Área Urbana","afloramento rochoso":"Afloramento Rochoso","rios e lagoas":"Rios e Lagoas"}
j["y7"] = j.Classe_Fin.map(CLS7); assert j.y7.notna().all()
# folds espaciais dos manuais = fold do bloco de 10 km (mesmo esquema do E1); blocos sem polígono automático recebem fold por sorteio
fs = pd.read_csv("outputs/folds_spatial.csv").merge(d[["supercells_global","cx","cy"]], on="supercells_global")
bstr = lambda x, y: np.floor(x/10000).astype(int).astype(str) + "_" + np.floor(y/10000).astype(int).astype(str)
bm = dict(zip(bstr(fs["cx"], fs["cy"]), fs["fold"]))
j["bloco"] = bstr(j["cx"], j["cy"]); rng = np.random.default_rng(42)
nb = [b for b in j.bloco.unique() if b not in bm]
extra_f = {b: int(rng.integers(0, 10)) for b in nb}
j["fold"] = [bm.get(b, extra_f.get(b)) for b in j.bloco]
print("manuais:", len(j), "| blocos novos (sem polígono automático):", len(nb), "| por fold:", j.fold.value_counts().sort_index().tolist())
j.drop(columns="geometry").to_csv("outputs/manual_features_p2names.csv", index=False)
