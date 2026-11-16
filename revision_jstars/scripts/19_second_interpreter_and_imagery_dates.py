"""Second-interpreter agreement (Cohen's kappa) and sensitivity of the audit to the imagery date.

Inputs (results/audit/):
  auditoria_resultado_completo.csv   unblinded audit (first interpreter)
  second_interpreter_labels.csv      60-polygon subset: vs = second interpreter, vm = first interpreter
  imagery_2023_check.csv             whether Google Earth has a 2023 acquisition for each audited polygon
Manuscript: Sec. III-I, IV-F, V-G.
"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score

A = "results/audit/"
LEVEL1 = {"pastagem": "r", "formacao campestre": "r", "agricultura": "r", "solo exposto": "r", "cerrado": "r"}


def kappa_ci(a, b, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(a)
    bs = [cohen_kappa_score(a.iloc[i], b.iloc[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return cohen_kappa_score(a, b), np.nanpercentile(bs, [2.5, 97.5])


k = pd.read_csv(A + "second_interpreter_labels.csv")
both = k[~k.vs.eq("indeterminado") & ~k.vm.eq("indeterminado")]
for name, f in [("10 classes", lambda x: x), ("7 macro-classes", lambda x: LEVEL1.get(x, x))]:
    a, b = both.vs.map(f), both.vm.map(f)
    kap, ci = kappa_ci(a, b)
    print(f"{name}: n={len(both)} agreement={(a == b).mean():.2f} kappa={kap:.2f} CI95={ci.round(2)}")

aud = pd.read_csv(A + "auditoria_resultado_completo.csv").merge(pd.read_csv(A + "imagery_2023_check.csv"), on="audit_id")
print("\npolygons without 2023 imagery:", (~aud.has2023).sum(), "of", len(aud))
v = aud[aud.vis.notna() & ~aud.vis.astype(str).str.contains("indeterm")].copy()
v["ok_point"], v["ok_dom"] = v.ass == v.vis, v.dom == v.vis
for name, d in [("all", v), ("with 2023 imagery", v[v.has2023]), ("without 2023 imagery", v[~v.has2023])]:
    by = d.groupby("faixa3")[["ok_point", "ok_dom"]].mean().mul(100).round(1)
    print(f"\n{name} (n={len(d)}): point {d.ok_point.mean()*100:.1f}%, dominant {d.ok_dom.mean()*100:.1f}%\n{by}")
for f, d in v.groupby("faixa3"):
    print(f, "Fisher p =", round(fisher_exact(pd.crosstab(d.has2023, d.ok_point))[1], 3))
