import geopandas as gpd
import pandas as pd
import os

def vincular_amostras_aos_poligonos():
    print("--- Iniciando vínculo espacial com ID global ---")

    base_dir = "/home/mschaefer/projetos/LULC_Hierarchical_Classification"

    csv_path = os.path.join(base_dir, "data_raw", "amostras_estratificadas_mapbiomas.csv")
    shp_path = os.path.join(base_dir, "data_raw", "external", "tiles_unificados_global.shp")
    output_dir = os.path.join(base_dir, "data_processed")
    output_csv = os.path.join(output_dir, "amostras_vinculadas_poligonos_global.csv")

    # 1. Carregar pontos
    df_pontos = pd.read_csv(csv_path)

    # 2. Carregar polígonos
    gdf_poligonos = gpd.read_file(shp_path)

    print(f"Amostras carregadas: {len(df_pontos)}")
    print(f"Polígonos carregados: {len(gdf_poligonos)}")
    print(f"CRS dos polígonos: {gdf_poligonos.crs}")
    print(f"Colunas do shapefile: {list(gdf_poligonos.columns)}")

    # 3. Coluna ID global
    col_id = "supercel_1"
    if col_id not in gdf_poligonos.columns:
        raise ValueError(
            f"A coluna '{col_id}' não existe no shapefile. "
            f"Colunas disponíveis: {list(gdf_poligonos.columns)}"
        )

    # 4. Converter pontos para GeoDataFrame
    gdf_pontos = gpd.GeoDataFrame(
        df_pontos.copy(),
        geometry=gpd.points_from_xy(df_pontos["x"], df_pontos["y"]),
        crs=gdf_poligonos.crs
    )

    print(f"Cruzando {len(gdf_pontos)} pontos com {len(gdf_poligonos)} polígonos...")

    # 5. Spatial join com within
    gdf_final = gpd.sjoin(
        gdf_pontos,
        gdf_poligonos[[col_id, "geometry"]],
        how="inner",
        predicate="within"
    )

    # 6. Renomear para um nome estável
    gdf_final = gdf_final.rename(columns={col_id: "supercells_global"})

    # 7. Criar pasta de saída
    os.makedirs(output_dir, exist_ok=True)

    # 8. Salvar resultado
    gdf_final.drop(columns=["geometry", "index_right"], errors="ignore").to_csv(output_csv, index=False)

    print("-" * 40)
    print("✅ SUCESSO")
    print(f"Amostras originais: {len(df_pontos)}")
    print(f"Amostras vinculadas: {len(gdf_final)}")
    print(f"Arquivo salvo em: {output_csv}")


if __name__ == "__main__":
    vincular_amostras_aos_poligonos()