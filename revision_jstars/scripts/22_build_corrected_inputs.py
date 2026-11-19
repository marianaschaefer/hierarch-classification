"""Build the corrected input tables used by scripts 03-17 from the outputs of scripts 20 and 21.

1. Pipeline 2 table: the 64 index columns of dataset_FINAL_COMPLETO_global_v1.csv are replaced by the corrected
   values (all other 216 columns are unchanged). The script first checks that the OLD_ columns of the extraction
   reproduce the deposited values.
2. Manual samples (Pipeline 1 table, R column names): all attribute columns are replaced by the Python extraction
   over the SLIC objects (script 21), so that the manual test set is described exactly like the training data.
   The same is done for base_modelo_pipeline1.csv (the 54 attributes selected in Pipeline 1).
The original tables are kept with the suffix _ORIGINAL.

Usage (from revision_jstars/):
    python scripts/22_build_corrected_inputs.py <idx_pipeline2.csv from script 20> <manual_full.csv from script 21>
Manuscript: Sec. III-B, III-H.
"""
import re
import shutil
import sys

import numpy as np
import pandas as pd

IDX, MAN = sys.argv[1:3]
Z = "data/QFGeo-LULC Dataset/QFGeo-LULC Dataset/"
P2 = Z + "05_atributos_pipeline2/dataset_FINAL_COMPLETO_global_v1.csv"
P1 = Z + "06_atributos_pipeline1/estatisticas_completas/estatisticas_poligonos_pipeline1.csv"
P1B = Z + "06_atributos_pipeline1/base_modelo_pronta/base_modelo_pipeline1.csv"

# ---- 1. Pipeline 2 -------------------------------------------------------------------------------
d = pd.read_csv(P2)
ix = pd.read_csv(IDX).set_index("supercells_global")
cols = [c for c in d.columns if re.search(r"_(ndvi|ndbi|ndsi|ebbi)_", c)]
assert len(cols) == 64
old = ix.loc[d.supercells_global, ["OLD_" + c for c in cols]].values
print("max |OLD extraction - deposited| =", np.nanmax(np.abs(old - d[cols].values)))
shutil.copy(P2, P2.replace(".csv", "_ORIGINAL.csv"))
d[cols] = ix.loc[d.supercells_global, cols].values
d.to_csv(P2, index=False)

# ---- 2. Manual samples (R names -> Python names) ------------------------------------------------
STAT = {"min": "min", "median": "mediana", "max": "max", "stdev": "std",
        "q05": "p05", "q25": "p25", "q75": "p75", "q95": "p95"}
PER = {"abril": "abr", "agost": "ago"}
BAND = {"blue": "blue", "verde": "green", "red": "red", "red_edge1": "rededge1", "red_edge2": "rededge2",
        "red_edge3": "rededge3", "nir": "nir", "swir1": "swir1", "swir2": "swir2",
        "ndvi": "ndvi", "NDBI": "ndbi", "NDSI": "ndsi", "EBBI": "ebbi"}


def py_name(rc):
    st, rest = rc.split(".", 1)
    if rest == "MDE":
        return f"mde_elev_{STAT[st]}"
    if rest == "slope":
        return f"slope_{STAT[st]}"
    m = re.fullmatch(r"(?:(abril|agost)_sentinel_(\w+)|(NDBI|NDSI|EBBI)_(abril|agost)_sentinel)", rest)
    per, band = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
    return f"{PER[per]}_{BAND[band]}_{STAT[st]}"


f = pd.read_csv(MAN).set_index("id_")
for path in (P1, P1B):
    s = pd.read_csv(path)
    shutil.copy(path, path.replace(".csv", "_ORIGINAL.csv"))
    fr = f.reindex(s.id_)
    out = pd.concat([s[["id_", "Classe_Fin"]],
                     pd.DataFrame({rc: fr[py_name(rc)].values for rc in s.columns[2:]})], axis=1)
    out.to_csv(path, index=False)
    print(path, out.shape, "missing values:", int(out.isna().sum().sum()))
