# M1: L2-Regularized Logistic Regression Scorecard

## Model Overview
- **Name:** Logistic Regression Scorecard (`M1_logreg`)
- **Type:** Generalized Linear Model (GLM)
- **Formulation:**
  $$P(Y_i = 1 \mid \mathbf{x}_i) = \sigma(\mathbf{w}^T \mathbf{x}_i + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}_i + b)}}$$
  Objective: Minimizes Cross-Entropy with L2 regularization penalty $\frac{1}{2C} \|\mathbf{w}\|_2^2$.

## Hyperparameter Tuning
- **Search Grid:** $C \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$
- **Selection Metric:** Validation PR-AUC.
- **Preprocessing:** Standard Z-score normalization ($\mu = 0, \sigma = 1$).

## Banking & Academic Rationale
In institutional banking (e.g. Basel II / III Internal Ratings-Based models), Logistic Regression represents the historical standard for credit scorecards due to direct interpretability of log-odds coefficients. However, its major limitation in distress surveillance is the assumption of linearity and inability to model temporal trajectories without manual feature flattening.
