import pandas as pd, numpy as np, glob, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
O="outputs/"; F="outputs/figs/"; rng=np.random.default_rng(7)
EN={"Afloramento Rochoso":"Rock Outcrop","Agricultura":"Agriculture","Eucalipto":"Eucalyptus","Formação Campestre":"Grassland","Mata":"Forest",
    "Mineração":"Mining","Pastagem":"Pasture","Rios e Lagoas":"Rivers & Lakes","Solo Exposto":"Bare Soil","Área Urbana":"Urban Area",
    "Vegetação Rasteira":"Grassland Veg.","Cerrado":"Grassland"}
plt.rcParams.update({"font.size":8,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False})
W=3.45
# ---------- per-class F1 com IC (bootstrap de folds) ----------
def perclass(df,y,p,label,classes):
    folds=sorted(df.fold.unique()); out={}
    base=precision_recall_fscore_support(df[y],df[p],labels=classes,zero_division=0)
    bs=[]
    for _ in range(1000):
        ks=rng.choice(folds,len(folds)); g=pd.concat([df[df.fold==k] for k in ks])
        bs.append(precision_recall_fscore_support(g[y],g[p],labels=classes,zero_division=0)[2])
    bs=np.array(bs); lo,hi=np.percentile(bs,2.5,axis=0),np.percentile(bs,97.5,axis=0)
    return pd.DataFrame({"class":[EN[c] for c in classes],"strategy":label,"precision":base[0],"recall":base[1],"f1":base[2],"f1_lo":lo,"f1_hi":hi,"support":base[3]})
n12=pd.read_csv(O+"n12_outer.csv"); d=pd.read_csv(O+"d_outer.csv")
C10=sorted(n12.y_point.unique())
R=n12[n12.tau==0]; Fi=n12[n12.tau==0.6]
full=pd.concat([perclass(R,"y_point","pred","Point labels",C10),perclass(Fi,"y_point","pred","Filter 0.6",C10),perclass(d,"y_point","pred","Dominant relabel",C10)])
full.to_csv("outputs/manuscript/tableS_perclass_fulltest.csv",index=False)
e2=pd.concat([pd.read_csv(f) for f in glob.glob(O+"e2_pred_*.csv")]).drop_duplicates(["id_","fold","variant"])
C7=sorted(e2.y7.unique()); lab={"R_bruto":"Point labels","F_pureza0.6":"Filter 0.6","D_relabel_dom":"Dominant relabel","DF_relabel_dom_filtro0.6":"Relabel + filter","M_manual":"Manual labels"}
man=pd.concat([perclass(e2[e2.variant==v],"y7","pred7",lab[v],C7) for v in lab])
man.to_csv("outputs/manuscript/tableS_perclass_manualtest.csv",index=False)
fmt=lambda r: f"{r.f1:.2f} [{r.f1_lo:.2f}, {r.f1_hi:.2f}]"
for name,t,order in [("full",full,["Point labels","Filter 0.6","Dominant relabel"]),("manual",man,list(lab.values()))]:
    pv=t.assign(s=t.apply(fmt,axis=1)).pivot(index="class",columns="strategy",values="s")[order]
    sup=t.groupby("class").support.first()
    print(f"\n=== per-class F1 [95% CI], {name} test ==="); print(pv.assign(n=sup).to_string())
    pv.assign(n=sup).to_csv(f"outputs/manuscript/table_perclass_{name}.csv")
# ---------- figuras ----------
P=pd.read_csv(O+"pureza_v2.csv")
fig,ax=plt.subplots(figsize=(W,2.1)); ax.hist(P.prop_label_m,bins=40,color="#4C72B0",edgecolor="white",linewidth=0.3)
ax.axvline(0.6,ls="--",c="k",lw=0.8); ax.set_xlabel(r"$P_{\mathrm{label}}$"); ax.set_ylabel("Polygons")
ax.text(0.61,ax.get_ylim()[1]*0.9,r"$\tau=0.6$",fontsize=7); fig.tight_layout(); fig.savefig(F+"fig01_purity_histogram_rev.pdf"); plt.close()
print("mean purity",P.prop_label_m.mean().round(3),"median",P.prop_label_m.median().round(3),"share<0.4",(P.prop_label_m<0.4).mean().round(3))
g=P.groupby("classe_nivel2_m").agg(mean=("prop_label_m","mean"),ret=("prop_label_m",lambda s:(s>=0.6).mean()*100)).sort_values("mean")
fig,axs=plt.subplots(1,2,figsize=(W,2.4),sharey=True)
y=np.arange(len(g)); names=[EN[c] for c in g.index]
axs[0].barh(y,g["mean"],color="#4C72B0"); axs[0].set_yticks(y); axs[0].set_yticklabels(names); axs[0].set_xlabel(r"Mean $P_{\mathrm{label}}$"); axs[0].set_xlim(0,1)
axs[1].barh(y,g["ret"],color="#DD8452"); axs[1].set_xlabel(r"Retained at $\tau=0.6$ (%)"); axs[1].set_xlim(0,100)
for i,v in enumerate(g["ret"]): axs[1].text(v+2,i,f"{v:.0f}",va="center",fontsize=6.5)
fig.tight_layout(); fig.savefig(F+"fig05_purity_retention_by_class.pdf"); plt.close()
# curva de limiar
taus=[0,0.5,0.6,0.7,0.8,0.9]
nm=pd.read_csv(O+"n12_manual.csv")
man_f=[f1_score(nm[nm.tau==t].y7,nm[nm.tau==t].pred7,average="macro") for t in taus]
pt=[f1_score(n12[n12.tau==t].y_point,n12[n12.tau==t].pred,average="macro") for t in taus]
dm=[f1_score(n12[n12.tau==t].y_dom,n12[n12.tau==t].pred,average="macro") for t in taus]
fig,ax=plt.subplots(figsize=(W,2.3))
for v,l,m,c in [(man_f,"Independent manual test (7 classes)","o","#2C7FB8"),(dm,"Full test, dominant-class reference","s","#41AB5D"),(pt,"Full test, point-label reference","^","#E6550D")]:
    ax.plot(taus,v,marker=m,ms=3.5,lw=1.1,label=l,c=c)
ax.set_xticks(taus); ax.set_xticklabels(["0\n(none)","0.5","0.6","0.7","0.8","0.9"]); ax.set_xlabel(r"Training purity threshold $\tau$"); ax.set_ylabel("Macro-F1"); ax.set_ylim(0.3,0.85)
ax.legend(frameon=False,fontsize=6.5,loc="center right",bbox_to_anchor=(1.0,0.62)); fig.tight_layout(); fig.savefig(F+"fig02_threshold_curve_rev.pdf"); plt.close()
print("curve manual",np.round(man_f,3),"dom",np.round(dm,3),"pt",np.round(pt,3))
# matriz de confusão XGB hier (rodada única)
st=pd.read_csv(O+"st_pred.csv"); h=st[st.model=="XGB_hier"]
order=["Forest","Eucalyptus","Pasture","Grassland","Agriculture","Bare Soil","Mining","Urban Area","Rock Outcrop","Rivers & Lakes"]
yt=h.y.map(EN); yp=h.pred.map(EN); cm=confusion_matrix(yt,yp,labels=order,normalize="true")
fig,ax=plt.subplots(figsize=(W,3.1)); im=ax.imshow(cm,cmap="Blues",vmin=0,vmax=1)
ax.set_xticks(range(10)); ax.set_xticklabels(order,rotation=55,ha="right",fontsize=6.5); ax.set_yticks(range(10)); ax.set_yticklabels(order,fontsize=6.5)
for i in range(10):
    for j in range(10):
        if cm[i,j]>=0.01: ax.text(j,i,f"{cm[i,j]:.2f}",ha="center",va="center",fontsize=5,color="white" if cm[i,j]>0.5 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("Reference (point label)"); fig.tight_layout(); fig.savefig(F+"fig04_confusion_xgb_hier_rev.pdf"); plt.close()
pr=precision_recall_fscore_support(yt,yp,labels=order,zero_division=0)
print(pd.DataFrame({"P":pr[0],"R":pr[1],"F1":pr[2],"n":pr[3]},index=order).round(2).to_string())
print("Bare soil confusions:",dict(zip(order,np.round(cm[order.index("Bare Soil")],2))))
