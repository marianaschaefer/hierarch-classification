import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from sklearn.metrics import confusion_matrix
import os

# Configuração de Acessibilidade: Paletas Viridis e Mako (Seguras para daltônicos)
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs("results/maps", exist_ok=True)

def gerar_tabela_geral():
    print("--- Consolidando Tabela Geral ---")
    try:
        df3 = pd.read_csv('results/boosting_hiclass_resultados_resumo.csv')
        df4 = pd.read_csv('results/boosting_flat_resultados_resumo.csv')
        df5 = pd.read_csv('results/deep_tabular_resultados_resumo.csv')
        
        # Unificar nomes de colunas para comparação
        df3 = df3.rename(columns={'hF1_medio': 'hF1', 'hF1_std': 'std', 'Acc_N2_media': 'Acc_N2'})
        df4 = df4.rename(columns={'hF1_medio': 'hF1', 'hF1_std': 'std', 'Acc_N2_media': 'Acc_N2'})
        df5 = df5.rename(columns={'hF1_medio': 'hF1', 'hF1_std': 'std', 'Acc_N2_media': 'Acc_N2'})
        
        df_final = pd.concat([df3, df4, df5])[['Modelo', 'hF1', 'std', 'Acc_N2']]
        df_final.to_csv("results/tabela_comparativa_tese.csv", index=False)
        return df_final
    except FileNotFoundError:
        print("⚠️ Erro: Alguns CSVs de resultados não foram encontrados.")
        return None

def plotar_grafico_estilo_tese(df):
    print("--- Gerando Gráfico de Barras (PDF) ---")
    plt.figure(figsize=(10, 6))
    # Paleta 'mako' é excelente para visualização sequencial e acessível
    sns.barplot(data=df, x='Modelo', y='hF1', palette='mako', hue='Modelo', legend=False)
    
    # Adicionar erro padrão (std)
    plt.errorbar(x=df['Modelo'], y=df['hF1'], yerr=df['std'], fmt='none', c='black', capsize=5)
    
    plt.ylim(0, 0.5) # Ajustado para sua escala de hF1 (~0.37)
    plt.ylabel('Hierarchical F1-Score (hF1)')
    plt.title('Performance Comparativa dos Modelos')
    plt.xticks(rotation=45, ha='right')
    
    for i, row in df.iterrows():
        plt.text(i, row['hF1'] + row['std'] + 0.01, f"{row['hF1']:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig("results/grafico_comparativo_tese.pdf", format='pdf')

def gerar_mapa_raster(caminho_tif_referencia, array_predicao, nome_arquivo, titulo, palette_name='viridis'):
    """
    Reconstrói o mapa usando o georreferenciamento do Sentinel-2/MDE.
    """
    with rasterio.open(caminho_tif_referencia) as src:
        meta = src.meta.copy()
        # [especulado] Assume-se que o array_predicao foi gerado na mesma ordem do raster
        # Se você usou supercells, precisará mapear os IDs de volta para a matriz 2D
        
        plt.figure(figsize=(10, 10))
        plt.imshow(array_predicao, cmap=palette_name)
        plt.colorbar(label='Classes')
        plt.title(titulo)
        plt.axis('off')
        plt.savefig(f"results/maps/{nome_arquivo}.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    df_resumo = gerar_tabela_geral()
    if df_resumo is not None:
        plotar_grafico_estilo_tese(df_resumo)