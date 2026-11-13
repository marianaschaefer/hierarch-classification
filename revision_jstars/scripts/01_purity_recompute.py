"""Recalcula a composição MapBiomas por polígono e a pureza (com e sem fusão Cerrado->Formação Campestre).
Regra de pixel: centro dentro do polígono (all_touched=False), igual ao rasterio.mask do x16."""
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio import features

UP = "data"
GPKG = f"{UP}/poligonos_finais_global_v1.gpkg"
TIF = f"{UP}/Tese P2/Figuras FINAL/mapbiomas_todasclasses.tif"
OLD = f"{UP}/backup LULC HC/data_processed/pureza_poligonos_mapbiomas.csv"
OUT = "outputs/pureza_v2.csv"

MB2LABEL = {3:"Mata",5:"Mata",6:"Mata",49:"Mata",4:"Cerrado",12:"Formação Campestre",15:"Pastagem",
            25:"Solo Exposto",21:"Agricultura",39:"Agricultura",41:"Agricultura",46:"Agricultura",
            48:"Agricultura",9:"Eucalipto",24:"Área Urbana",29:"Afloramento Rochoso",30:"Mineração",
            31:"Rios e Lagoas",33:"Rios e Lagoas"}

g = gpd.read_file(GPKG).reset_index(drop=True)
src = rasterio.open(TIF)
A = src.read(1)
assert str(g.crs) == str(src.crs), (g.crs, src.crs)
ids = features.rasterize(((geom, i+1) for i, geom in enumerate(g.geometry)),
                         out_shape=A.shape, transform=src.transform, fill=0, dtype="int32")
m = ids > 0
key = ids[m].astype(np.int64) * 256 + A[m].astype(np.int64)
cnt = np.bincount(key, minlength=(len(g)+1)*256).reshape(-1, 256)[1:]
print("polígonos:", len(g), " pixels atribuídos:", int(m.sum()))
comp = pd.DataFrame(cnt, columns=[f"c{k}" for k in range(256)])
keep = [c for c in comp.columns if comp[c].sum() > 0]
comp = comp[keep]
print("códigos presentes:", [int(c[1:]) for c in keep])

def labelcounts(merge):
    lab = {k: ("Formação Campestre" if (merge and v == "Cerrado") else v) for k, v in MB2LABEL.items()}
    cols = {}
    for k, v in lab.items():
        c = f"c{k}"
        if c in comp: cols[v] = cols.get(v, 0) + comp[c].values
    return pd.DataFrame(cols)

res = g[["supercells_global", "classe_nivel1", "classe_nivel2"]].copy()
res["classe_nivel2_m"] = res.classe_nivel2.replace({"Cerrado": "Formação Campestre"})
res["n_poly_px"] = comp.sum(axis=1).values
res["n_zero_px"] = comp["c0"].values if "c0" in comp else 0
for tag, merge, ycol in [("v1", False, "classe_nivel2"), ("m", True, "classe_nivel2_m")]:
    L = labelcounts(merge)
    tot = L.sum(axis=1).replace(0, np.nan)
    res[f"n_mapped_{tag}"] = tot.values
    res[f"prop_label_{tag}"] = [L.at[i, y]/tot.iloc[i] if y in L else np.nan for i, y in enumerate(res[ycol])]
    dom = L.idxmax(axis=1)
    res[f"dom_{tag}"] = dom.values
    res[f"prop_dom_{tag}"] = (L.max(axis=1)/tot).values
res["n_unmapped_nonzero_px"] = res.n_poly_px - res.n_zero_px - res.n_mapped_v1
res = pd.concat([res, comp.add_prefix("mb_")], axis=1)
res.to_csv(OUT, index=False)

# validação contra o x16 original
old = pd.read_csv(OLD).set_index("supercells_global")
r = res.set_index("supercells_global").join(old[["prop_label", "n_pixels_mapeados", "classe_dominante"]])
print("\n== Validação vs pureza original ==")
print("n_mapped_v1 == n_pixels_mapeados:", (r.n_mapped_v1 == r.n_pixels_mapeados).mean().round(4))
print("max |prop_label_v1 - prop_label|:", (r.prop_label_v1 - r.prop_label).abs().max())
print("dom_v1 == classe_dominante:", (r.dom_v1 == r.classe_dominante).mean().round(4))
print("pixels não mapeados e != 0 (códigos fora do dicionário):", int(res.n_unmapped_nonzero_px.sum()),
      " em", int((res.n_unmapped_nonzero_px > 0).sum()), "polígonos")
print("\npureza média v1 / fundida:", res.prop_label_v1.mean().round(3), res.prop_label_m.mean().round(3))
for t in (0.6, 0.7, 0.8, 0.9):
    print(f"retidos >= {t}:  v1 {int((res.prop_label_v1>=t).sum())}   fundida {int((res.prop_label_m>=t).sum())}")
print("rótulo == dominante: v1", (res.classe_nivel2 == res.dom_v1).mean().round(3), " fundida", (res.classe_nivel2_m == res.dom_m).mean().round(3))
