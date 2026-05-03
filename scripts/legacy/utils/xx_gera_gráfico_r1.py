import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carga de Dados
df_flat = pd.read_csv('boosting_flat_resultados_resumo.csv')
df_hiclass = pd.read_csv('boosting_hiclass_resultados_resumo.csv')
df_deep = pd.read_csv('deep_tabular_resultados_resumo.csv')
df_all = pd.concat([df_flat, df_hiclass, df_deep], ignore_index=True)

# 2. Paleta de Cores Individual por Modelo
model_palette = {
    'XGBoost_Flat': '#1f77b4', 'LightGBM_Flat': '#aec7e8', 'CatBoost_Flat': '#ff7f0e',
    'XGBoost_HiClass': '#ffbb78', 'LightGBM_HiClass': '#2ca02c', 'CatBoost_HiClass': '#98df8a',
    'DeepMLP': '#d62728', 'FTTransformer': '#ff9896'
}

def configurar_estilo():
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10, 'axes.titleweight': 'bold'})

# --- GRÁFICO 1: hF1 ---
plt.figure(figsize=(12, 6))
configurar_estilo()
df_sorted = df_all.sort_values(by='hF1_medio', ascending=False)
colors = [model_palette.get(m, '#333333') for m in df_sorted['Modelo']]
bars = plt.bar(df_sorted['Modelo'], df_sorted['hF1_medio'], color=colors, edgecolor='black', alpha=0.9)
plt.errorbar(df_sorted['Modelo'], df_sorted['hF1_medio'], yerr=df_sorted['hF1_std'], fmt='none', ecolor='black', capsize=5)
plt.title('Comparativo: Hierarchical F1-Score ($hF_1$)', pad=15)
plt.ylim(0, 1.0) # Eixo fixado em 1.0
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('grafico_hf1_final.png', dpi=300)

# --- GRÁFICO 2: Acurácia por Níveis ---
plt.figure(figsize=(14, 7))
df_acc = df_all.melt(id_vars=['Modelo'], value_vars=['Acc_N1_media', 'Acc_N2_media'], var_name='Nivel', value_name='Acuracia')
df_acc['Nivel'] = df_acc['Nivel'].map({'Acc_N1_media': 'Nível 1', 'Acc_N2_media': 'Nível 2'})
ax = sns.barplot(data=df_acc, x='Nivel', y='Acuracia', hue='Modelo', palette=model_palette, edgecolor='black')
plt.title('Acurácia por Nível Hierárquico', pad=15)
plt.ylim(0, 1.0) # Eixo fixado em 1.0
plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('grafico_acuracia_final.png', dpi=300)

# --- GRÁFICO 3: F1 Macro Nível 2 ---
plt.figure(figsize=(12, 6))
df_sorted_n2 = df_all.sort_values(by='F1_macro_N2_medio', ascending=False)
colors_n2 = [model_palette.get(m, '#333333') for m in df_sorted_n2['Modelo']]
plt.bar(df_sorted_n2['Modelo'], df_sorted_n2['F1_macro_N2_medio'], color=colors_n2, edgecolor='black', alpha=0.9)
plt.title('$F_1$-Score Macro no Nível 2 (Subclasses)', pad=15)
plt.ylim(0, 1.0) # Eixo fixado em 1.0
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('grafico_f1_macro_n2_final.png', dpi=300)