# S1: Cox Proportional Hazards Model

## Model Overview
- **Name:** Cox Proportional Hazards (`S1_cox`)
- **Type:** Semi-parametric survival analysis
- **Framework:** `lifelines` 0.30+
- **Hazard Formulation:**
  $$\lambda(t \mid \mathbf{x}) = \lambda_0(t) \exp(\boldsymbol{\beta}^T \mathbf{x})$$
  $$\hat{P}(\text{Distress before } t \mid \mathbf{x}) = 1 - S_0(t)^{\exp(\boldsymbol{\beta}^T \mathbf{x})}$$

## Hyperparameter Tuning
- **L2 Penalizer:** Swept over $\{0.01, 0.1, 1.0, 5.0\}$ to avoid ill-conditioned matrix inversion under correlated banking ratios.
- **Metric:** Validation PR-AUC on cumulative default occurrence and Concordance Index ($C$-index).

## Academic & Banking Context
Cox regression handles right-censoring naturally (loans that have not defaulted by observation end). However, standard Cox assumes the **proportional hazards assumption** (covariate effects are constant across time), which is violated in practice when loan distress dynamics evolve non-linearly over the life of a loan.
