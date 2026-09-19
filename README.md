# Forest Covertype: End-To-End Machine Learning Pipeline

A modular, production-grade Machine Learning pipeline built to predict forest cover types from cartographic and topographical attributes across 580,000+ spatial observations.

## Project Architecture & Pipeline Design
- **Data Ingestion Module**: Automated dataset retrieval and target mapping using `sklearn.datasets.fetch_covtype`.
- **Domain Feature Engineering**: Custom `ForestFeatureEngineer` transformer executing:
  - Euclidean distance calculation to water bodies ($d = \sqrt{\Delta x^2 + \Delta y^2}$).
  - Elevation-to-hydrology vertical offset proxies.
  - Directional solar shade index variations (Morning vs. Evening hillshades).
  - Reverse mapping one-hot categorical vectors into index-based ordinal features.
- **Preprocessing**: Stratified K-Fold splitting to preserve non-uniform target distributions paired with `StandardScaler` transformations on continuous numerical dimensions.
- **Model Architecture**: Multi-class LightGBM classifier with early stopping and multi-logloss optimization.

## Key Results & Metrics
- **Weighted F1-Score**: ~0.90+
- **Macro F1-Score**: ~0.88+
- Evaluated via normalized confusion matrices and feature importance rankings.

## Repository Structure
```text
├── notebook.ipynb        # End-to-end execution notebook
├── requirements.txt      # Environment dependencies
└── README.md             # Architectural documentation
