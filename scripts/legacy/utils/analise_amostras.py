import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# 1) CARREGAR AMOSTRAS
# =========================================================

csv_path = "data_raw/amostras_estratificadas_mapbiomas.csv"
df_amostras = pd.read_csv(csv_path)

print(f"Total de amostras: {len(df_amostras)}")


# =========================================================
# 2) CONTAGEM DE AMOSTRAS
# =========================================================

contagem_nivel2 = (
    df_amostras['classe_nivel2']
    .value_counts()
    .sort_index()
    .reset_index()
)
contagem_nivel2.columns = ['classe', 'n_amostras']

print("\n=== Contagem por classe ===")
print(contagem_nivel2)

contagem_nivel2.to_csv('data_raw/contagem_amostras.csv', index=False)


# =========================================================
# 3) MAPA DE AMOSTRAS (COLORBLIND-FRIENDLY)
# =========================================================

# Criar GeoDataFrame (sem CRS definido — só visual)
gdf = gpd.GeoDataFrame(
    df_amostras.copy(),
    geometry=gpd.points_from_xy(df_amostras['x'], df_amostras['y'])
)

# Paleta Okabe-Ito
cores = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    "#999999", "#44AA99", "#882255", "#117733"
]

classes = sorted(gdf['classe_nivel2'].unique())

mapa_cores = {
    classe: cores[i % len(cores)]
    for i, classe in enumerate(classes)
}

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

for classe in classes:
    subset = gdf[gdf['classe_nivel2'] == classe]
    ax.scatter(
        subset['x'],
        subset['y'],
        s=8,
        c=mapa_cores[classe],
        alpha=0.8,
        edgecolors='none'
    )

ax.set_title('Distribuição Espacial das Amostras', fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Legenda manual (melhor controle)
legenda = [
    Line2D([0], [0], marker='o', color='w',
           label=classe,
           markerfacecolor=mapa_cores[classe],
           markersize=6)
    for classe in classes
]

ax.legend(
    handles=legenda,
    title="Classes",
    loc='upper left',
    bbox_to_anchor=(0.01, 0.99)
)

plt.tight_layout()

plt.savefig("data_raw/mapa_amostras.png", dpi=300)
plt.savefig("data_raw/mapa_amostras.pdf", dpi=300)

plt.show()

print("\n✔️ Mapa salvo em data_raw/")