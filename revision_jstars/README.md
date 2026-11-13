# Revision analyses — JSTARS-2026-03945

Code, partitions and results for the revised manuscript *Object-Level Label Consistency in OBIA Workflows: Impact on LULC Classification Performance*. The scripts in `../scripts/` are the original pipeline; this folder contains the analyses added in the revision.

Analyses were run with Python 3.13 and the package versions in `requirements.txt`. All XGBoost models use 300 trees, maximum depth 6, learning rate 0.05, row and column subsampling of 0.8, histogram method, seed 42.

## Data

Download the QFGeo-LULC dataset from Zenodo (<https://doi.org/10.5281/zenodo.20927225>) and place it under `data/`:

| Path expected by the scripts | Content |
|---|---|
| `data/QFGeo-LULC Dataset/QFGeo-LULC Dataset/05_atributos_pipeline2/dataset_FINAL_COMPLETO_global_v1.csv` | Pipeline 2 attributes (280) |
| `data/QFGeo-LULC Dataset/QFGeo-LULC Dataset/06_atributos_pipeline1/` | Pipeline 1 attributes (manual samples) |
| `data/poligonos_finais_global_v1.gpkg` | Sampled polygons (for centroids and spatial blocks) |
| `data/mapbiomas/` | MapBiomas rasters (only for scripts 01, 09, 10) |
| `data/audit/auditoria_formulario_preenchido.xlsx` | Completed blind audit form (only for script 13) |

Run the scripts from this folder (`revision_jstars/`) after `mkdir -p outputs/figs outputs/manuscript outputs/auditoria`.

## Scripts and manuscript sections

| Script | What it does | Manuscript |
|---|---|---|
| `common.py` | Data loading, spatial folds (10-km blocks, stratified group k-fold), 2-km buffer | — |
| `01_purity_recompute.py` | Purity ($P_{label}$, $P_{dominant}$) after merging Cerrado into Grassland | III-F, IV-A |
| `02_audit_sample.py` | Stratified sample for the blind audit (10 classes × 3 purity ranges × 10) | III-I |
| `03_common_test_random_vs_spatial.py`, `04_common_test_analysis.py` | Common unfiltered test set; random vs spatial folds | III-H |
| `05_manual_test_prepare.py`, `06_manual_test_run.py`, `07_manual_test_analysis.py` | Independent manual test set and controlled comparison of label sources | IV-E, Table XI |
| `08_pipeline1_feature_selection_leakage.py` | Pipeline 1 feature selection inside vs outside training partitions | III-H |
| `09_mapbiomas_year_check.py`, `10_mapbiomas_year_check_interior.py` | Verification of MapBiomas collection and year | III-B, Table S1 |
| `11_nested_threshold_selection.py` | Nested selection of the purity threshold and threshold curve | IV-C, Tables VII–VIII |
| `12_relabeling_strategies.py` | Dominant-class relabeling (with and without filter) | IV-D, Table X |
| `13_audit_analysis.py` | Blind audit: agreement, weighting, causes of disagreement | IV-F, Tables XII–XIII |
| `14_threshold_and_strategy_analysis.py` | Tables of thresholds, purity ranges, strategies and audit agreement with CIs | IV-C, IV-D, Tables VII–X |
| `15_model_comparison_run.py`, `16_model_comparison_stats.py` | RF flat, XGBoost flat and hierarchical with identical folds; ablation; block bootstrap; Nadeau–Bengio tests with Holm | IV-G, IV-H, Tables XIV–XVI |
| `17_tables_and_figures.py` | Figures and class-wise tables | Figs. 6–9, Table IX |

## Results

- `results/folds/`: fold assignments (`folds_spatial.csv` for the full dataset, `st_folds.csv` for the filtered dataset of the model comparison).
- `results/pureza_v2.csv`: purity per polygon.
- `results/predictions/`: out-of-fold predictions of every experiment (gzip CSV).
- `results/audit/`: blind audit protocol, form and polygons, and the unblinded results.
- `results/tables/`: tables reported in the manuscript and Supplementary Material.

## Note on band names

In `../scripts/feature_engineering/x05_stats_ind_band_mde.py`, the band list names the 8th and 9th bands `nirn` and `swir1`. The deposited attribute table names them `swir1` and `swir2` (Sentinel-2 B11 and B12), which is consistent with the data: the `swir1` column has low correlation with NIR (r ≈ 0.19), as expected for B11 and not for B8A. The indices in the deposited table use B11 as SWIR-1. EBBI* is computed as (B8 − B4)/(10·√(B8 + B11)).
