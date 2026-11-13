import pandas as pd, numpy as np, unicodedata, re
O="outputs/"
f=pd.read_excel("data/audit/auditoria_formulario_preenchido.xlsx")
k=pd.read_csv("outputs/auditoria/auditoria_CHAVE_nao_abrir_antes.csv")
norm=lambda s: unicodedata.normalize("NFKD",str(s).strip().lower()).encode("ascii","ignore").decode()
fix={"fomacao campestre":"formacao campestre","inteterminado":"indeterminado"}
f["vis"]=f.classe_visual.map(norm).replace(fix)
k["ass"]=k.classe_atribuida.map(norm).replace({"cerrado":"formacao campestre"}); k["dom"]=k.dominante_mapbiomas.map(norm).replace({"cerrado":"formacao campestre"})
d=k.merge(f[["audit_id","vis","mistura_visual","mudanca_2021_2023","observacao"]],on="audit_id")
print("n",len(d),"| classes visuais fora das 10:",sorted(set(d.vis)-set(d.ass)-{'indeterminado'}))
d["faixa3"]=d.faixa.str.split("(").str[0]
ind=d[d.vis=="indeterminado"]; v=d[d.vis!="indeterminado"].copy()
print("\nIndeterminados:",len(ind), ind.faixa3.value_counts().to_dict(), ind.ass.value_counts().to_dict())
def wilson(x,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=x/n; c=(p+z*z/2/n)/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/(1+z*z/n); return c-h,c+h
v["ok_ass"]=v.vis==v.ass; v["ok_dom"]=v.vis==v.dom
print("\n=== Concordância do julgamento visual com cada rótulo (n=%d sem indeterminados) ==="%len(v))
rows=[]
for fx in ["baixa","media","alta"]:
    g=v[v.faixa3==fx]
    for nm,c in [("rótulo de ponto","ok_ass"),("classe dominante MapBiomas","ok_dom")]:
        lo,hi=wilson(g[c].sum(),len(g)); rows.append(dict(faixa=fx,rotulo=nm,n=len(g),acordo=g[c].mean(),ic_lo=lo,ic_hi=hi))
for nm,c in [("rótulo de ponto","ok_ass"),("classe dominante MapBiomas","ok_dom")]:
    lo,hi=wilson(v[c].sum(),len(v)); rows.append(dict(faixa="todas",rotulo=nm,n=len(v),acordo=v[c].mean(),ic_lo=lo,ic_hi=hi))
print(pd.DataFrame(rows).round(3).to_string(index=False))
# estimativa ponderada pela população (estrato = classe x faixa)
P=pd.read_csv(O+"pureza_v2.csv",usecols=["classe_nivel2_m","prop_label_m"])
P["ass"]=P.classe_nivel2_m.map(norm); P["faixa3"]=np.where(P.prop_label_m>=0.6,"alta",np.where(P.prop_label_m>=0.4,"media","baixa"))
N=P.groupby(["ass","faixa3"]).size().rename("N").reset_index()
w=v.merge(N,on=["ass","faixa3"]); w["n_s"]=w.groupby(["ass","faixa3"]).audit_id.transform("count"); w["w"]=w.N/w.n_s
rng=np.random.default_rng(0)
def west(df,col,mask=None):
    g=df if mask is None else df[mask]; return (g[col]*g.w).sum()/g.w.sum()
print("\n=== Estimativa ponderada para a população de polígonos (IC95% bootstrap estratificado) ===")
for nm,col in [("rótulo de ponto","ok_ass"),("classe dominante","ok_dom")]:
    for lab,m in [("toda a população",None),("pureza>=0,6",w.faixa3=="alta"),("pureza<0,6",w.faixa3!="alta")]:
        est=west(w,col,m); bs=[]
        for _ in range(1000):
            b=w.groupby(["ass","faixa3"],group_keys=False).sample(frac=1,replace=True,random_state=int(rng.integers(1e9)))
            bs.append(west(b,col,None if m is None else (b.faixa3=="alta") if lab=="pureza>=0,6" else (b.faixa3!="alta")))
        print(f"{nm:18s} {lab:18s} {est:.3f} [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
# acordo por classe atribuída
print("\n=== Acordo visual × rótulo de ponto, por classe atribuída (n por classe ≈ 30) ===")
print(v.groupby("ass").agg(n=("ok_ass","size"),acordo_ponto=("ok_ass","mean"),acordo_dom=("ok_dom","mean")).round(2).sort_values("acordo_ponto").to_string())
# mistura visual × pureza
print("\n=== Mistura visual × faixa de pureza (contagens) ==="); print(pd.crosstab(d.faixa3,d.mistura_visual.fillna("(vazio)")))
# mudança temporal
print("\n=== Mudança 2021–2023 ==="); print(pd.crosstab(v.mudanca_2021_2023.fillna("(vazio)"),v.ok_ass,margins=True))
# classificação das divergências (visual != rótulo de ponto)
dv=v[~v.ok_ass].copy()
def causa(r):
    if r.mudanca_2021_2023=="s": return "mudança temporal (2021–2023)"
    if r.ok_dom: return "ponto não representa o polígono (dominante correta)"
    if r.dom==r.ass: return "MapBiomas e ponto concordam; julgamento visual difere"
    return "nenhum dos rótulos MapBiomas bate com o visual"
dv["causa"]=dv.apply(causa,axis=1)
print("\n=== Causa das divergências (visual ≠ rótulo de ponto): n=%d de %d ==="%(len(dv),len(v)))
print(pd.crosstab(dv.causa,dv.faixa3,margins=True))
dv.to_csv(O+"auditoria_divergencias.csv",index=False)
d.to_csv(O+"auditoria_resultado_completo.csv",index=False)
