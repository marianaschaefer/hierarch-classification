"""Análise do E1: métricas com IC (bootstrap por bloco de 10 km), por estrato de teste, por classe e diferenças pareadas."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import load
scheme = sys.argv[1]
d, _ = load()
bx = np.floor(d.cx / 10000).astype(int).astype(str) + "_" + np.floor(d.cy / 10000).astype(int).astype(str)
blk = pd.Series(pd.factorize(bx)[0], index=d.supercells_global)
pr = pd.read_csv(f"outputs/e1_pred_{scheme}.csv")
pr["block"] = blk.reindex(pr.supercells_global).values
CL = sorted(set(pr.y_point) | set(pr.y_dom) | set(pr.pred)); ci = {c: i for i, c in enumerate(CL)}; K = len(CL)
def f1m(y, p):
    cm = np.bincount(y * K + p, minlength=K*K).reshape(K, K).astype(float)
    tp = np.diag(cm); pp = cm.sum(0); aa = cm.sum(1)
    f1 = np.where(pp + aa > 0, 2*tp/np.maximum(pp + aa, 1), np.nan)
    present = aa > 0
    rec = np.where(aa > 0, tp/np.maximum(aa, 1), np.nan)
    return np.nanmean(f1[present]), np.nanmean(rec[present]), tp.sum()/cm.sum()   # F1 macro, bal.acc, acc
strata = {"todos": lambda t: np.ones(len(t), bool), "puros(P>=0.6)": lambda t: t.P_label.values >= 0.6,
          "impuros(P<0.6)": lambda t: t.P_label.values < 0.6}
rng = np.random.default_rng(0); B = 500
rows = []; boot_store = {}
for v, g in pr.groupby("variant"):
    for truth in ["y_point", "y_dom"]:
        for sn, sf in strata.items():
            t = g[sf(g)]
            y = t[truth].map(ci).values; p = t.pred.map(ci).values; b = t.block.values
            f, ba, acc = f1m(y, p)
            ub = np.unique(b); idx = {u: np.where(b == u)[0] for u in ub}
            bs = []
            for _ in range(B):
                s = np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])
                bs.append(f1m(y[s], p[s])[0])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(scheme=scheme, variant=v, truth=truth, test=sn, n_test=len(t), f1_macro=f, bal_acc=ba, acc=acc, f1_lo=lo, f1_hi=hi))
            boot_store[(v, truth, sn)] = None
S = pd.DataFrame(rows); S.to_csv(f"outputs/e1_summary_{scheme}.csv", index=False)
pd.set_option("display.width", 200)
print(S[S.truth == "y_point"].pivot(index="variant", columns="test", values="f1_macro").round(3))
print(S[(S.truth == "y_point") & (S.test == "todos")][["variant", "f1_macro", "f1_lo", "f1_hi", "bal_acc", "acc"]].round(3).to_string(index=False))
# por classe (teste = todos, rótulo do ponto)
pc = []
for v, g in pr.groupby("variant"):
    y = g.y_point.map(ci).values; p = g.pred.map(ci).values
    cm = np.bincount(y * K + p, minlength=K*K).reshape(K, K).astype(float)
    tp = np.diag(cm); prec = tp/np.maximum(cm.sum(0), 1); rec = tp/np.maximum(cm.sum(1), 1)
    for i, c in enumerate(CL): pc.append(dict(scheme=scheme, variant=v, classe=c, n=int(cm.sum(1)[i]), precisao=prec[i], recall=rec[i], f1=2*tp[i]/max(cm.sum(0)[i]+cm.sum(1)[i], 1)))
PC = pd.DataFrame(pc); PC.to_csv(f"outputs/e1_perclass_{scheme}.csv", index=False)
print(PC.pivot(index="classe", columns="variant", values="f1").round(2))
# diferenças pareadas (bootstrap por bloco) no teste comum, rótulo do ponto
vs = sorted(pr.variant.unique()); pairs = [(a, b) for a in vs for b in vs if a < b]
base = {v: pr[pr.variant == v].set_index("supercells_global").sort_index() for v in vs}
out = []
for a, b in pairs:
    A, Bv = base[a], base[b]; common = A.index.intersection(Bv.index)
    A, Bv = A.loc[common], Bv.loc[common]
    y = A.y_point.map(ci).values; pa = A.pred.map(ci).values; pb = Bv.pred.map(ci).values; bb = A.block.values
    ub = np.unique(bb); idx = {u: np.where(bb == u)[0] for u in ub}
    d0 = f1m(y, pb)[0] - f1m(y, pa)[0]; ds = []
    for _ in range(B):
        s = np.concatenate([idx[u] for u in rng.choice(ub, len(ub))]); ds.append(f1m(y[s], pb[s])[0] - f1m(y[s], pa[s])[0])
    out.append(dict(scheme=scheme, A=a, B=b, dif_f1_B_menos_A=d0, lo=np.percentile(ds, 2.5), hi=np.percentile(ds, 97.5)))
D = pd.DataFrame(out); D.to_csv(f"outputs/e1_paired_{scheme}.csv", index=False); print(D.round(3).to_string(index=False))
