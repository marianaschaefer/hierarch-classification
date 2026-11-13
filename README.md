# LULC Label Consistency Pipeline

This repository contains the code used for object-based land use and land cover (LULC) classification experiments, focusing on the impact of object-level label consistency in automatically generated training datasets.

## Overview

The workflow investigates how inconsistencies between polygon-level features and point/pixel-based labels affect classification performance in object-based image analysis (OBIA).

The pipeline includes:

- preprocessing of segmented polygons;
- spatial linking between samples and polygons;
- feature extraction from Sentinel-2, spectral indices, DEM, and satellite embeddings;
- object-level label purity computation;
- flat and hierarchical classification using Random Forest and XGBoost;
- ablation experiments;
- figure generation for publication.

## Repository Structure

```text
scripts/
├── preprocessing/          # spatial preprocessing and polygon/sample linking
├── feature_engineering/    # feature extraction and final dataset assembly
├── modeling/               # flat, hierarchical, and ablation experiments
├── analysis/               # object-level purity and diagnostic analyses
├── pipeline/               # final purity-filtered pipeline
├── visualization/          # figure generation scripts
└── legacy/                 # exploratory or archived scripts not used in the final pipeline


## Revision (JSTARS-2026-03945)

The analyses added in the revision of the JSTARS manuscript, with fold assignments and out-of-fold predictions, are in [`revision_jstars/`](revision_jstars/README.md).
