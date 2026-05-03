import pandas as pd

def limpar_conflitos_removendo_empates(input_path, output_path):
    df = pd.read_csv(input_path)
    total_original = len(df)

    print(f"Total de registros originais: {total_original}")

    counts = (
        df.groupby(['supercells_global', 'classe_nivel1', 'classe_nivel2'])
        .size()
        .reset_index(name='frequencia')
    )

    counts['max_freq'] = counts.groupby('supercells_global')['frequencia'].transform('max')

    empatados_topo = counts[counts['frequencia'] == counts['max_freq']].copy()

    ids_com_empate = (
        empatados_topo.groupby('supercells_global')
        .size()
        .loc[lambda x: x > 1]
        .index
    )

    counts_sem_empate = counts[~counts['supercells_global'].isin(ids_com_empate)].copy()

    counts_sem_empate = counts_sem_empate.sort_values(
        ['supercells_global', 'frequencia'],
        ascending=[True, False]
    )

    df_clean = counts_sem_empate.drop_duplicates(subset=['supercells_global'], keep='first')
    df_clean = df_clean.drop(columns=['frequencia', 'max_freq'])

    df_clean.to_csv(output_path, index=False)

    print("-" * 40)
    print("✅ Limpeza concluída")
    print(f"Polígonos com empate removidos: {len(ids_com_empate)}")
    print(f"Polígonos finais sem ambiguidade: {len(df_clean)}")
    print(f"Registros removidos no total: {total_original - len(df_clean)}")
    print(f"Arquivo salvo em: {output_path}")


if __name__ == "__main__":
    limpar_conflitos_removendo_empates(
        '/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/amostras_vinculadas_poligonos_global.csv',
        '/home/mschaefer/projetos/LULC_Hierarchical_Classification/data_processed/dataset_FINAL_CLEAN_tese_global_v1.csv'
    )