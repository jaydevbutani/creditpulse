# M2: XGBoost Gradient Boosted Decision Trees

## Model Overview
- **Name:** XGBoost Classifier (`M2_xgb`)
- **Type:** Gradient Boosted Decision Tree (GBDT)
- **Framework:** XGBoost
- **Objective:** Logistic loss with regularized objective (L1 and L2 tree penalties)

## Hyperparameter Grid Search
- **Max Depth:** $3, 5$
- **Learning Rate:** $0.05, 0.10$
- **Number of Estimators:** $50, 100$
- **Evaluation Metric:** Logloss / PR-AUC

## Academic & Banking Context
XGBoost serves as an industry-standard benchmark alongside LightGBM. Its exact greedy tree search and second-order Taylor expansion of the loss function provide sharp discrimination on structured tabular tables, but it treats all 176 flattened features statically without temporal causal structure.
