import pandas as pd

def limpar_conflitos_removendo_empates(input_path, output_path):
    df = pd.read_csv(input_path)
    total_original = len(df)

    print(f"Total de registros originais: {total_original}")

    # 1. Contar frequência de cada classe por polígono
    counts = (
        df.groupby(['supercells', 'classe_nivel1', 'classe_nivel2'])
        .size()
        .reset_index(name='frequencia')
    )

    # 2. Descobrir a frequência máxima por polígono
    counts['max_freq'] = counts.groupby('supercells')['frequencia'].transform('max')

    # 3. Manter apenas as classes empatadas no topo
    empatados_topo = counts[counts['frequencia'] == counts['max_freq']].copy()

    # 4. Identificar polígonos com empate real
    ids_com_empate = (
        empatados_topo.groupby('supercells')
        .size()
        .loc[lambda x: x > 1]
        .index
    )

    # 5. Remover polígonos com empate
    counts_sem_empate = counts[~counts['supercells'].isin(ids_com_empate)].copy()

    # 6. Entre os sem empate, pegar a classe majoritária
    counts_sem_empate = counts_sem_empate.sort_values(
        ['supercells', 'frequencia'],
        ascending=[True, False]
    )

    df_clean = counts_sem_empate.drop_duplicates(subset=['supercells'], keep='first')

    # 7. Remover colunas auxiliares
    df_clean = df_clean.drop(columns=['frequencia', 'max_freq'])

    # 8. Salvar
    df_clean.to_csv(output_path, index=False)

    print("-" * 40)
    print("✅ Limpeza concluída")
    print(f"Polígonos com empate removidos: {len(ids_com_empate)}")
    print(f"Polígonos finais sem ambiguidade: {len(df_clean)}")
    print(f"Registros removidos no total: {total_original - len(df_clean)}")
    print(f"Arquivo salvo em: {output_path}")


if __name__ == "__main__":
    limpar_conflitos_removendo_empates(
        'data_processed/amostras_vinculadas_poligonos.csv',
        'data_processed/dataset_FINAL_CLEAN_tese_v2.csv'
    )


df = pd.read_csv("data_processed/dataset_FINAL_CLEAN_tese_v2.csv")
print("Linhas:", len(df))
print("Supercells únicos:", df["supercells"].nunique())