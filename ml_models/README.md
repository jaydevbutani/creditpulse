# CreditPulse Model Zoo: Longitudinal Distress Surveillance Benchmarks

## Overview
The CreditPulse Model Zoo provides a rigorous, inspectable, and reproducible benchmark suite of **11 distinct statistical, survival, machine learning, and deep learning architectures**. Each model is implemented with authentic training loops, hyperparameter optimization, and standardized Basel III / CECL evaluation metrics.

---

## 1. Benchmark Model Zoo Index

| Model ID | Architecture | Paradigm | Input Representation | Prediction Target | Trainable Params |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`M0_prior`** | Empirical Base Rate Prior | Non-parametric Statistical | None (constant) | Baseline Distress Rate | 0 |
| **`M1_logreg`** | L2-Regularized Logistic Regression | Generalized Linear Model | 176 Aggregated Ratios | 12m Distress Probability | 177 |
| **`M2_lgbm`** | LightGBM Decision Trees | Gradient Boosted Trees | 176 Aggregated Ratios | 12m Distress Probability | 50 trees |
| **`M2_xgb`** | XGBoost Decision Trees | Gradient Boosted Trees | 176 Aggregated Ratios | 12m Distress Probability | 50 trees |
| **`S1_cox`** | Cox Proportional Hazards | Semi-Parametric Survival | 15 Selected Ratios | Hazard Ratio / Survival Curve | 15 |
| **`S2_hazard`** | Neural Discrete Hazard Network | Discrete Survival DL | $[B, 12, 44]$ Sequence | 6-Month Hazard Curve | 2,694 |
| **`M3_gru`** | 2-Layer Gated Recurrent Unit | Recurrent Neural Net | $[B, 12, 44]$ Sequence | 12m Distress Probability | 13,857 |
| **`M4_lstm`** | 2-Layer Long Short-Term Memory | Recurrent Neural Net | $[B, 12, 44]$ Sequence | 12m Distress Probability | 18,465 |
| **`M5_transformer`** | Single-Task Pre-LN Transformer | Self-Attention Sequence DL | $[B, 12, 44]$ Sequence | 12m Distress Probability | 72,129 |
| **`M6_multitask`** | **Multi-Task Pre-LN Transformer** | **Joint Attention + Survival** | **$[B, 12, 44]$ Sequence** | **12m Distress + 6m Hazard** | **74,407** |
| **`M7_pretrained`** | Masked Reconstruction Pre-trained | Self-Supervised Transfer DL | $[B, 12, 44]$ Sequence | 12m Distress Probability | 79,261 |

---

## 2. Core Academic Novelty of Proposed M6 Multi-Task Transformer

> **"Temporal distress surveillance rather than one-shot credit scoring."**

Traditional bank underwriting uses static credit scoring models (e.g. `M1_logreg`, `M2_lgbm`) evaluated at origination. In contrast, CreditPulse formulates continuous early warning monitoring over monthly loan behavioral trajectories:
1. **Self-Attention over Behavioral Trajectories:** While RNNs (`M3_gru`, `M4_lstm`) suffer from catastrophic forgetting and gradient bottlenecks over long panels, multi-head self-attention computes direct pairwise importance across all historical payment cycles.
2. **Inductive Multi-Task Regularization:** Jointly optimizing Head A (12-month distress) and Head B (discrete monthly hazard rate) forces latent representations to encode not merely *whether* a borrower fails, but *when* distress accelerates.
3. **Regulatory Calibration:** Logits are calibrated via post-hoc Platt scaling ($a \cdot z + b$) to guarantee well-calibrated expected distress rates for CECL / IFRS-9 loan-loss provisioning.

---

## 3. Running Any Model in the Zoo

Every model is completely self-contained and runnable via standard module execution:

```bash
# Statistical & Tabular Benchmarks
python -m ml_models.M0_prior.train
python -m ml_models.M1_logreg.train
python -m ml_models.M2_lgbm.train
python -m ml_models.M2_xgb.train

# Survival Analysis Benchmarks
python -m ml_models.S1_cox.train
python -m ml_models.S2_hazard.train

# Deep Learning Benchmarks
python -m ml_models.M3_gru.train
python -m ml_models.M4_lstm.train
python -m ml_models.M5_transformer.train
python -m ml_models.M6_multitask.train
python -m ml_models.M7_pretrained.train
```

Each run executes:
1. Real dataset loading from `demo_dataset/creditpulse_teacher_demo.csv`.
2. Stratified train/val/test partitioning.
3. Hyperparameter sweep on the validation partition.
4. Evaluation on held-out test facilities with authentic PR-AUC, ROC-AUC, Brier score, ECE, and Capture@10%.
5. Metric persistence to `<model_folder>/metrics.json`.
