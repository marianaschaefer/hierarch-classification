# Revision analyses — JSTARS-2026-03945

Code, partitions and results for the revised manuscript *Object-Level Label Consistency in OBIA Workflows: Impact on LULC Classification Performance*. The scripts in `../scripts/` are the original pipeline; this folder contains the analyses added in the revision.

Analyses were run with Python 3.13 and the package versions in `requirements.txt`. All XGBoost models use 300 trees, maximum depth 6, learning rate 0.05, row and column subsampling of 0.8, histogram method, seed 42.

## Data

Download the QFGeo-LULC dataset from Zenodo (concept DOI, all versions: <https://doi.org/10.5281/zenodo.20927224>; use version 2.0.0 or later, which includes the corrected indices and the manual-sample attributes as `v2_*` files) and place it under `data/`:

| Path expected by the scripts | Content |
|---|---|
| `data/QFGeo-LULC Dataset/QFGeo-LULC Dataset/05_atributos_pipeline2/dataset_FINAL_COMPLETO_global_v1.csv` | Pipeline 2 attributes (280) |
| `data/QFGeo-LULC Dataset/QFGeo-LULC Dataset/06_atributos_pipeline1/` | Pipeline 1 attributes (manual samples) |
| `data/poligonos_finais_global_v1.gpkg` | Sampled polygons (for centroids and spatial blocks) |
| `data/mapbiomas/` | MapBiomas rasters (only for scripts 01, 09, 10) |
| `data/audit/auditoria_formulario_preenchido.xlsx` | Completed blind audit form (only for script 13) |

Run the scripts from this folder (`revision_jstars/`) after `mkdir -p outputs/figs outputs/manuscript outputs/auditoria`.

**Corrected inputs (November 2026).** Before scripts 03–17, build the corrected input tables (see *Corrections* below):

```bash
# from the dataset root (folder with 02_raster/), on the full-resolution rasters
python scripts/20_extract_corrected_indices.py <dataset_root> data/poligonos_finais_global_v1.gpkg idx_pipeline2.csv supercells_global
python scripts/21_extract_manual_features_slic.py <dataset_root> manual_full.csv
# from revision_jstars/
python scripts/22_build_corrected_inputs.py idx_pipeline2.csv manual_full.csv
```

The outputs of scripts 20 and 21 are also deposited in the dataset (Zenodo), so steps 20–21 can be skipped.

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
| `18_gee_verification_s2_embeddings.js` | Earth Engine check of the Sentinel-2 product (L2A vs L1C, compositing alternatives) and of the embedding aggregation (mean vs median) on 33 polygons encoded in the script | III-B, Supplementary |
| `20_extract_corrected_indices.py` | Spectral indices recomputed from reflectance, (DN − 1000)/10000, removing the Sentinel-2 radiometric offset; also reproduces the original (offset) values for checking | III-B, Supplementary |
| `21_extract_manual_features_slic.py` | Attributes of the 1,395 manual samples extracted over their SLIC objects with the same procedure as Pipeline 2 | III-H, IV-E |
| `22_build_corrected_inputs.py` | Replaces the 64 index columns of the Pipeline 2 table and the attributes of the manual samples by the corrected values | III-B, III-H |
| `19_second_interpreter_and_imagery_dates.py` | Second-interpreter agreement (Cohen's kappa) and sensitivity of the audit to the availability of 2023 imagery | III-I, IV-F, V-G |

## Results

- `results/folds/`: fold assignments (`folds_spatial.csv` and `folds_random.csv` for the full dataset, `st_folds.csv` for the filtered dataset of the model comparison).
- `results/pureza_v2.csv`: purity per polygon.
- `results/predictions/`: out-of-fold predictions of every experiment (gzip CSV).
- `results/audit/`: blind audit protocol, form and polygons, the unblinded results, the second interpreter's labels (`second_interpreter_labels.csv`) and the 2023 imagery check (`imagery_2023_check.csv`).
- `results/tables/`: tables reported in the manuscript and Supplementary Material; `results/figs/`: figures.
- `results/logs/`: console output of each script in the final run (all reported numbers can be traced there).
- `results/control/`: control run of the independent manual test with consistent extraction but the original (offset) indices.
- `results/archive_v1_offset_indices/`: predictions and tables of the first revision run, before the corrections below (kept for transparency).

## Corrections (November 2026)

Two problems were found and corrected after the first revision run. All results in `results/` (except the archive) come from the corrected run.

1. **Radiometric offset in the spectral indices.** The Sentinel-2 L2A composites (processing baseline 04.00) store DN = 10000·reflectance + 1000. NDVI, NDBI, NDSI and EBBI* had been computed on DN, i.e. including the offset. They are now computed on reflectance, (DN − 1000)/10000 (script 20; the extraction reproduces the deposited values to < 1e-6 before the correction). The bands themselves keep the offset: a constant shift does not affect tree-based models. Effect on results: < 0.01 in all metrics (e.g. macro-F1 of the model comparison: RF 0.750→0.757, XGBoost flat 0.781→0.777, hierarchical 0.773→0.774).
2. **Inconsistent extraction of the manual test set.** In the independent manual test (scripts 05–07, 11, 12), the models were trained with Pipeline 2 attributes (Python, 10-m composites, pixel centres) but the manual objects were described with the Pipeline 1 attributes (R `exactextractr`, area-weighted, from a different version of the rasters). Medians agreed (r ≈ 0.998) but dispersion statistics did not (standard deviation of the visible bands about 30% lower in the R table; extremes compressed). The manual objects are now described with the same extraction as the training data (script 21). This changed the manual-test results of the automatic-label models: macro-F1 with point labels 0.727→0.857, with the purity filter 0.782→0.848; the difference filter − point changed from +0.055 [+0.031, +0.078] to −0.009 [−0.032, +0.012]. The control run in `results/control/` (consistent extraction, original offset indices extracted with the same procedure: 0.864, 0.852, −0.013 [−0.031, +0.006]) shows that this change comes from the extraction, not from the offset. Models trained on manual labels were not affected (0.920→0.922), because their training and test attributes came from the same table.

Notes on the run: scripts 06, 11, 12 and 15 resume from existing output files, and the run was restarted several times; the inputs were unchanged across restarts and an independent check retrained a subset of models with identical predictions. `results/logs/` holds the output of the final, complete execution of each script. 133 of the 1,395 manual samples share a SLIC object with another sample (the 2-km buffer prevents leakage between folds).

Pipeline 1 itself (Table VI) is internally consistent (training and test from the R table); script 08 was rerun with the corrected attributes (macro-F1 0.928 with the fixed 54 attributes, 0.941 with selection inside each training partition).

## Note on band names

In `../scripts/feature_engineering/x05_stats_ind_band_mde.py`, the band list names the 8th and 9th bands `nirn` and `swir1`. The deposited attribute table names them `swir1` and `swir2` (Sentinel-2 B11 and B12), which is consistent with the data: the `swir1` column has low correlation with NIR (r ≈ 0.19), as expected for B11 and not for B8A. The indices in the deposited table use B11 as SWIR-1. EBBI* is computed as (B8 − B4)/(10·√(B8 + B11)).
