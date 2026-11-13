"""Amostra estratificada (classe nível 2 fundida x faixa de pureza) para a auditoria visual cega."""
import numpy as np, pandas as pd, geopandas as gpd, os
UP = "data"
P = pd.read_csv("outputs/pureza_v2.csv", usecols=["supercells_global","classe_nivel1","classe_nivel2_m","prop_label_m","dom_m","prop_dom_m","n_poly_px"])
G = gpd.read_file(f"{UP}/poligonos_finais_global_v1.gpkg")[["supercells_global","geometry"]]
d = P.merge(G, on="supercells_global"); d = gpd.GeoDataFrame(d, crs="EPSG:31983")
d["faixa"] = pd.cut(d.prop_label_m, [-0.001, 0.4, 0.6, 1.0], labels=["baixa(<0.4)", "media(0.4-0.6)", "alta(>=0.6)"], right=False)
d.loc[d.prop_label_m >= 0.6, "faixa"] = "alta(>=0.6)"
print(pd.crosstab(d.classe_nivel2_m, d.faixa))
N = 10; rng = np.random.default_rng(2026)
sel = []
for (c, f), grp in d.groupby(["classe_nivel2_m", "faixa"], observed=True):
    sel.append(grp.sample(min(N, len(grp)), random_state=int(rng.integers(1e9))))
s = pd.concat(sel).sample(frac=1, random_state=7).reset_index(drop=True)
s["audit_id"] = [f"A{i+1:03d}" for i in range(len(s))]
cent = s.geometry.centroid.to_crs(4326)
s["lon"], s["lat"] = cent.x.round(6), cent.y.round(6)
s["area_ha"] = (s.geometry.area/1e4).round(2)
os.makedirs("outputs/auditoria", exist_ok=True)
blind = s[["audit_id","lon","lat","area_ha"]].copy()
for c in ["classe_visual","mistura_visual","mudanca_2021_2023","observacao"]: blind[c] = ""
blind.to_csv("outputs/auditoria/auditoria_formulario_cego.csv", index=False, encoding="utf-8-sig")
s[["audit_id","supercells_global","classe_nivel2_m","faixa","prop_label_m","dom_m","prop_dom_m","n_poly_px"]].rename(
  columns={"classe_nivel2_m":"classe_atribuida","prop_label_m":"P_label","dom_m":"dominante_mapbiomas","prop_dom_m":"P_dominante","n_poly_px":"n_px"}
  ).to_csv("outputs/auditoria/auditoria_CHAVE_nao_abrir_antes.csv", index=False, encoding="utf-8-sig")
g = s[["audit_id","geometry"]].to_crs(4326)
g.to_file("outputs/auditoria/auditoria_poligonos.gpkg", driver="GPKG")
try:
    g.rename(columns={"audit_id":"Name"}).to_file("outputs/auditoria/auditoria_poligonos.kml", driver="KML")
    print("KML ok")
except Exception as e: print("KML falhou:", e)
print("total:", len(s)); print(s.groupby(["classe_nivel2_m","faixa"],observed=True).size().unstack().fillna(0).astype(int))
