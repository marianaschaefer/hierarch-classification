import pandas as pd, numpy as np, unicodedata, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support
O="outputs/"; rng=np.random.default_rng(0)
f1=lambda y,p: f1_score(y,p,average="macro")
inn=pd.read_csv(O+"n300_inner.csv"); out=pd.read_csv(O+"n300_outer.csv"); man=pd.read_csv(O+"n300_manual.csv")
ro=pd.read_csv(O+"r300_outer.csv"); rm=pd.read_csv(O+"r300_manual.csv")
TAUS=[0.0,0.5,0.6,0.7,0.8,0.9]; folds=list(range(10))
sel={}
for c in ["f1_point","f1_dom"]:
    g=inn.groupby(["fold","tau"])[c].mean().reset_index(); sel[c]=g.loc[g.groupby("fold")[c].idxmax()].set_index("fold").tau
    print("seleção",c,sel[c].value_counts().to_dict())
print("\n=== Tabela de limiar ===")
rows=[]
for t in TAUS:
    o=out[out.tau==t]; m=man[man.tau==t]
    rows.append(dict(tau=t,n_train=round(o.groupby("fold").n_train.first().mean()),pt=f1(o.y_point,o.pred),dom=f1(o.y_dom,o.pred),man=f1(m.y7,m.pred7),man_ba=balanced_accuracy_score(m.y7,m.pred7)))
T=pd.DataFrame(rows); print(T.round(3).to_string(index=False))
def boot_man(fa,fb,B=1000):
    pt=fa(folds)-fb(folds); bs=[]
    for _ in range(B):
        ks=list(rng.choice(folds,10)); bs.append(fa(ks)-fb(ks))
    return pt,np.percentile(bs,2.5),np.percentile(bs,97.5)
def mf(df,ks,col_y="y7",col_p="pred7"):
    g=pd.concat([df[df.fold==k] for k in ks]); return f1(g[col_y],g[col_p])
M=lambda t: (lambda ks: mf(man[man.tau==t],ks))
for c in sel:
    s=sel[c].to_dict(); fs=lambda ks,s=s: mf(pd.concat([man[(man.fold==k)&(man.tau==s[k])] for k in range(10)]),ks)
    print(f"aninhada({c}) − bruto (manual):", np.round(boot_man(fs,M(0.0)),3))
print("τ0.6 − bruto (manual):",np.round(boot_man(M(0.6),M(0.0)),3))
print("τ0.5 − bruto (manual):",np.round(boot_man(M(0.5),M(0.0)),3))
print("\n=== Faixas de pureza e decomposição (vs ponto) ===")
bins=[0,0.4,0.6,0.8,1.0001]; lab=["<0.4","0.4-0.6","0.6-0.8",">=0.8"]
out["faixa"]=pd.cut(out.P_label,bins,labels=lab,right=False)
for t in [0.0,0.6]:
    o=out[out.tau==t]; print(t,{f:(len(g),round(f1(g.y_point,g.pred),3)) for f,g in o.groupby("faixa",observed=True)},
      "| completo",round(f1(o.y_point,o.pred),3),"| só P>=0.6",round(f1(o[o.P_label>=0.6].y_point,o[o.P_label>=0.6].pred),3))
print("\n=== Estratégias ===")
S={"Point":(out[out.tau==0],man[man.tau==0]),"Filter":(out[out.tau==0.6],man[man.tau==0.6]),
   "Relabel":(ro[ro.variant=="D_relabel_dom"],rm[rm.variant=="D_relabel_dom"]),"Relabel+drop":(ro[ro.variant=="DF_relabel_filtroDom0.5"],rm[rm.variant=="DF_relabel_filtroDom0.5"])}
for k,(o,m) in S.items(): print(f"{k:13s} ntrain {o.groupby('fold').n_train.first().mean():.0f} pt {f1(o.y_point,o.pred):.3f} dom {f1(o.y_dom,o.pred):.3f} manual {f1(m.y7,m.pred7):.3f}")
SM={k:v[1] for k,v in S.items()}
for a,b in [("Relabel","Filter"),("Relabel","Point"),("Relabel+drop","Filter"),("Filter","Point")]:
    print(f"manual {a} − {b}:",np.round(boot_man(lambda ks,a=a: mf(SM[a],ks),lambda ks,b=b: mf(SM[b],ks)),3))
# auditoria com IC
au=pd.read_csv(O+"auditoria_resultado_completo.csv"); au=au[au.vis!="indeterminado"].copy()
key=pd.read_csv("outputs/auditoria/auditoria_CHAVE_nao_abrir_antes.csv")[["audit_id","supercells_global"]]
au=au.drop(columns="supercells_global",errors="ignore").merge(key,on="audit_id")
norm=lambda s: unicodedata.normalize("NFKD",str(s).strip().lower()).encode("ascii","ignore").decode()
P=pd.read_csv(O+"pureza_v2.csv",usecols=["classe_nivel2_m","prop_label_m"]); P["ass"]=P.classe_nivel2_m.map(norm)
P["faixa3"]=np.where(P.prop_label_m>=0.6,"alta",np.where(P.prop_label_m>=0.4,"media","baixa")); N=P.groupby(["ass","faixa3"]).size().rename("N").reset_index()
R7={"formacao campestre","pastagem","solo exposto","agricultura"}; m7=lambda s:"vegetacao rasteira" if s in R7 else s
print("\n=== Auditoria (ponderada, IC95% bootstrap estratificado) ===")
AU={}
for k,(o,_) in S.items():
    g=au.merge(o[["supercells_global","pred"]].assign(p=lambda x:x.pred.map(norm)),on="supercells_global")
    g["ok10"]=g.p==g.vis; g["ok7"]=g.p.map(m7)==g.vis.map(m7); g=g.merge(N,on=["ass","faixa3"])
    g["w"]=g.N/g.groupby(["ass","faixa3"]).audit_id.transform("count"); AU[k]=g
def west(g,c): return (g[c]*g.w).sum()/g.w.sum()
strata=list(AU["Point"].groupby(["ass","faixa3"]).groups)
idx={k:AU[k].groupby(["ass","faixa3"]).indices for k in AU}
res={}
for c in ["ok10","ok7"]:
    bs={k:[] for k in AU}; dif={k:[] for k in AU}
    for _ in range(1000):
        pick={s:rng.choice(idx["Point"][s],len(idx["Point"][s])) for s in strata}
        vals={}
        for k,g in AU.items():
            ii=np.concatenate([pick[s] for s in strata]); gg=g.iloc[ii]; gg=gg.assign(w=gg.N/gg.groupby(["ass","faixa3"]).audit_id.transform("count")); vals[k]=west(gg,c)
        for k in AU: bs[k].append(vals[k]); dif[k].append(vals[k]-vals["Filter"])
    for k in AU:
        print(f"{c} {k:13s} {west(AU[k],c)*100:.1f}% [{np.percentile(bs[k],2.5)*100:.1f}, {np.percentile(bs[k],97.5)*100:.1f}]  vs Filter {np.mean(dif[k])*100:+.1f} [{np.percentile(dif[k],2.5)*100:+.1f}, {np.percentile(dif[k],97.5)*100:+.1f}]")
assert all((AU[k].audit_id.values==AU["Point"].audit_id.values).all() for k in AU)
# per-class com IC no teste completo
EN={"Afloramento Rochoso":"Rock Outcrop","Agricultura":"Agriculture","Eucalipto":"Eucalyptus","Formação Campestre":"Grassland","Mata":"Forest","Mineração":"Mining","Pastagem":"Pasture","Rios e Lagoas":"Rivers & Lakes","Solo Exposto":"Bare Soil","Área Urbana":"Urban Area"}
C10=sorted(out.y_point.unique()); order=["Forest","Eucalyptus","Pasture","Grassland","Agriculture","Bare Soil","Mining","Urban Area","Rock Outcrop","Rivers & Lakes"]
pc={}
for k in ["Point","Filter","Relabel"]:
    o=S[k][0]; base=precision_recall_fscore_support(o.y_point,o.pred,labels=C10,zero_division=0); bs=[]
    for _ in range(1000):
        ks=rng.choice(folds,10); g=pd.concat([o[o.fold==q] for q in ks]); bs.append(precision_recall_fscore_support(g.y_point,g.pred,labels=C10,zero_division=0)[2])
    bs=np.array(bs); pc[k]=pd.DataFrame({"class":[EN[c] for c in C10],"P":base[0],"R":base[1],"F1":base[2],"lo":np.percentile(bs,2.5,0),"hi":np.percentile(bs,97.5,0),"n":base[3]}).set_index("class").loc[order]
    pc[k].to_csv(O+f"perclass300_{k}.csv")
print("\n=== F1 por classe [IC] (teste completo) ===")
print(pd.DataFrame({k:pc[k].apply(lambda r:f"{r.F1:.2f} [{r.lo:.2f}, {r.hi:.2f}]",axis=1) for k in pc}).assign(n=pc["Point"].n).to_string())
T.to_csv(O+"threshold300.csv",index=False)
# figura
plt.rcParams.update({"font.size":8,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(3.45,2.3))
for col,l,mk,c in [("man","Independent manual test (7 classes)","o","#2C7FB8"),("dom","Full test, dominant-class reference","s","#41AB5D"),("pt","Full test, point-label reference","^","#E6550D")]:
    ax.plot(TAUS,T[col],marker=mk,ms=3.5,lw=1.1,label=l,c=c)
ax.set_xticks(TAUS); ax.set_xticklabels(["0\n(none)","0.5","0.6","0.7","0.8","0.9"]); ax.set_xlabel(r"Training purity threshold $\tau$"); ax.set_ylabel("Macro-F1"); ax.set_ylim(0.3,0.85)
ax.legend(frameon=False,fontsize=6.5,loc="center right",bbox_to_anchor=(1.0,0.62)); fig.tight_layout(); fig.savefig("outputs/figs/fig02_threshold_curve_rev.pdf"); plt.close()
