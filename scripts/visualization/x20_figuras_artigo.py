import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

BASE = Path("/home/mschaefer/projetos/LULC_Hierarchical_Classification")

OUT = BASE / "reports" / "figures_article"
OUT.mkdir(parents=True, exist_ok=True)

pureza_path = BASE / "data_processed" / "pureza_poligonos_mapbiomas.csv"
threshold_path = BASE / "data_processed" / "exp08_pureza_vs_performance_summary.csv"
final_summary_path = BASE / "data_processed" / "final_pureza_0.6_summary.csv"

# Matriz de confusão do melhor modelo final
cm_path = BASE / "data_processed" / "final_pureza_0.6_XGB_flat_matriz_confusao.csv"

# Se você tiver a matriz da hierárquica filtrada, troque o caminho acima por ela.
# Exemplo sugerido:
# cm_path = BASE / "data_processed" / "hierXGB_pureza_matriz_confusao.csv"

# =========================================================
# LOAD
# =========================================================

df_p = pd.read_csv(pureza_path)
df_thr = pd.read_csv(threshold_path)
df_final = pd.read_csv(final_summary_path)

# =========================================================
# FUNÇÃO PARA SALVAR EM MÚLTIPLOS FORMATOS
# =========================================================

def salvar_figura(nome):
    for ext in ["pdf", "svg", "eps", "png"]:
        plt.savefig(OUT / f"{nome}.{ext}", dpi=300, bbox_inches="tight")


# =========================================================
# FIGURA 1 — HISTOGRAMA DE PUREZA
# =========================================================

plt.figure(figsize=(7, 5))
plt.hist(
    df_p["prop_label"],
    bins=30,
    edgecolor="black"
)

plt.xlabel("Object-label purity ($P_{label}$)")
plt.ylabel("Number of objects")
plt.title("Distribution of object-level label purity")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.axvline(0.6, linestyle="--", linewidth=2)

salvar_figura("fig01_purity_histogram")
plt.close()


# =========================================================
# FIGURA 2 — DESEMPENHO POR LIMIAR DE PUREZA
# =========================================================

plt.figure(figsize=(7, 5))

plt.plot(
    df_thr["threshold"],
    df_thr["f1_macro"],
    marker="o",
    linewidth=2,
    label="Macro-F1"
)

plt.plot(
    df_thr["threshold"],
    df_thr["balanced_accuracy"],
    marker="s",
    linewidth=2,
    label="Balanced accuracy"
)

plt.xlabel("Purity threshold ($P_{label}$)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(df_thr["threshold"])
plt.axvline(0.6, linestyle="--")
plt.title("Classification performance under purity filtering")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

salvar_figura("fig02_purity_threshold_performance")
plt.close()


# =========================================================
# FIGURA 3 — RF VS XGB APÓS FILTRO DE PUREZA
# =========================================================

metrics = ["accuracy", "balanced_accuracy", "f1_macro", "kappa"]
df_plot = df_final.set_index("modelo")[metrics]

ax = df_plot.plot(
    kind="bar",
    figsize=(8, 5),
    edgecolor="black"
)

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_title("Model performance after purity filtering ($P_{label} \\geq 0.6$)")
ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

salvar_figura("fig03_model_comparison_filtered")
plt.close()


# =========================================================
# FIGURA 4 — MATRIZ DE CONFUSÃO NORMALIZADA
# =========================================================

if cm_path.exists():
    cm = pd.read_csv(cm_path, index_col=0)

    # normalizar por linha
    cm_norm = cm.div(cm.sum(axis=1), axis=0)

    plt.figure(figsize=(10, 8))

    im = plt.imshow(
        cm_norm.values,
        interpolation="nearest",
        cmap="cividis",
        vmin=0,
        vmax=1
    )

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Proportion")

    labels = cm_norm.index.tolist()

    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    plt.xlabel("Predicted class")
    plt.ylabel("Reference class")
    plt.title("Normalized confusion matrix")

    # texto nas células
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            value = cm_norm.iloc[i, j]
            if value >= 0.05:
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.5 else "black"
                )

    plt.tight_layout()
    salvar_figura("fig04_normalized_confusion_matrix")
    plt.close()

else:
    print(f"Aviso: matriz de confusão não encontrada em: {cm_path}")


# =========================================================
# FIGURA 5 — PUREZA MÉDIA POR CLASSE
# =========================================================

df_class = (
    df_p
    .groupby("classe_label")["prop_label"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
)

plt.figure(figsize=(9, 5))

ax = df_class["mean"].plot(
    kind="bar",
    edgecolor="black"
)

ax.set_ylim(0, 1)
ax.set_ylabel("Mean object-label purity")
ax.set_xlabel("Class")
ax.set_title("Mean object-level label purity by class")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

salvar_figura("fig05_mean_purity_by_class")
plt.close()


print("Figuras salvas em:", OUT)