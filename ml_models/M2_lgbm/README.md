# M2: LightGBM Gradient Boosted Decision Trees

## Model Overview
- **Name:** LightGBM Classifier (`M2_lgbm`)
- **Type:** Gradient Boosted Decision Tree (GBDT)
- **Framework:** LightGBM 4.x
- **Objective:** Binary cross-entropy with leaf-wise splitting

## Hyperparameter Grid Search
- **Max Depth:** $3, 5$
- **Learning Rate:** $0.05, 0.10$
- **Number of Estimators:** $50, 100$
- **Objective Metric:** Validation PR-AUC

## Academic & Banking Context
LightGBM is widely regarded as the industry gold-standard tabular benchmark in financial risk modeling due to its handling of non-linear feature interactions and robustness to scale. However, it requires flattening time-series trajectories into ad-hoc summary statistics, thereby losing explicit temporal ordering and inter-period covariance dynamics captured by attention mechanisms.
