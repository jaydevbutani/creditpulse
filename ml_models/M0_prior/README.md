# M0: Empirical Prior Baseline

## Model Overview
- **Name:** Empirical Base Rate Prior (`M0_prior`)
- **Type:** Non-parametric statistical baseline
- **Mathematical Form:** 
  $$\hat{P}(Y_i = 1 \mid X_i) = \bar{y}_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{j=1}^{N_{\text{train}}} y_j$$
- **Parameters:** 0

## Purpose & Banking Significance
`M0_prior` represents the null model where zero banking features or temporal sequences are observed. It assigns the historical portfolio default rate as the probability for all facilities. In academic evaluation:
- **ROC-AUC:** Always exactly $0.5000$ (random guessing).
- **PR-AUC:** Exactly equals the positive class prevalence in the test set.
- **Brier Score:** Serves as the maximum baseline calibration error for subsequent machine learning models.
