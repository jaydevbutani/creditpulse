# CreditPulse Model Configurations & Parameter Trace

This document details the exact configuration, architecture layers, parameter locations, and tracing flow for all benchmark models in the CreditPulse Model Zoo.

---

## 1. Core Neural Architecture Summary

| Model ID | Input Shape | Main Layers / Blocks | Hidden / Model Dim | Output Heads & Targets | Total Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`M3_gru`** | $[B, 12, 44]$ | 2-Layer Recurrent GRU | `hidden_dim = 32` | 1 Head: 12M Distress Risk | 13,857 |
| **`M4_lstm`** | $[B, 12, 44]$ | 2-Layer Recurrent LSTM with Cell Memory | `hidden_dim = 32` | 1 Head: 12M Distress Risk | 18,465 |
| **`M5_transformer`** | $[B, 12, 44]$ | 2 Pre-LN Transformer Blocks ($h=4$ heads) | `d_model = 64` | 1 Head: 12M Distress Risk | 72,129 |
| **`M6_multitask`** *(Champion)* | $[B, 12, 44]$ | 3 Pre-LN Transformer Blocks ($h=2$ heads) | `d_model = 32` | 2 Heads: 6M Distress + 6M Hazard | 33,383 |

---

## 2. Complete 11-Model Zoo Reference Table

| Model ID | Input Representation | Architecture Type | Key Hyperparameters | Loss Function | Optimizer | Epochs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`M0_prior`** | None (constant base rate) | Empirical Prior Predictor | Prior default rate $\bar{y} = 0.042$ | None | Analytical mean | — |
| **`M1_logreg`** | 176 Aggregated Ratios | L2-Regularized Logistic Scorecard | $C = 0.001$, `penalty="l2"`, `solver="lbfgs"` | Binary Cross-Entropy | L-BFGS Quasi-Newton | — |
| **`M2_lgbm`** | 176 Aggregated Ratios | LightGBM Decision Trees | `depth = 3`, `learning_rate = 0.05`, `n_est = 50` | Binary Log-loss | LightGBM GBDT | — |
| **`M2_xgb`** | 176 Aggregated Ratios | XGBoost Decision Trees | `depth = 3`, `learning_rate = 0.05`, `n_est = 50` | Binary Log-loss | XGBoost GBDT | — |
| **`S1_cox`** | 15 Informative Ratios | Cox Proportional Hazards | L2 penalizer $\alpha = 0.01$, `l1_ratio = 0.0` | Cox Partial Likelihood | Newton-Raphson | — |
| **`S2_hazard`** | $[B, 12, 44]$ Sequence | Discrete Neural Hazard Net | `hidden_dim = 32`, `lr = 0.0001`, `horizon = 6` | Multi-Period Hazard BCE | AdamW (`decay=1e-4`) | 15 |
| **`M3_gru`** | $[B, 12, 44]$ Sequence | 2-Layer Gated Recurrent Unit | `hidden_dim = 32`, `num_layers = 2`, `lr = 0.001` | BCE with Logits | AdamW (`decay=1e-4`) | 15 |
| **`M4_lstm`** | $[B, 12, 44]$ Sequence | 2-Layer Long Short-Term Memory | `hidden_dim = 32`, `num_layers = 2`, `lr = 0.001` | BCE with Logits | AdamW (`decay=1e-4`) | 15 |
| **`M5_transformer`** | $[B, 12, 44]$ Sequence | Single-Task Pre-LN Transformer | `d_model = 64`, `nhead = 4`, `num_layers = 2` | BCE with Logits | AdamW (`decay=1e-4`) | 15 |
| **`M6_multitask`** | $[B, 12, 44]$ Sequence | **Multi-Task Pre-LN Transformer** | `d_model = 32`, `nhead = 2`, `layers = 3`, `λ = 0.30` | $\mathcal{L}_{\text{distress}} + 0.30 \mathcal{L}_{\text{hazard}}$ | AdamW (`lr=0.0002`) | 8 |
| **`M7_pretrained`** | $[B, 12, 44]$ Sequence | Self-Supervised BiGRU + Fine-Tuning | `hidden_dim = 48`, `mask_ratio = 0.15` | Phase 1: MSE; Phase 2: BCE | AdamW (`decay=1e-4`) | 10 pre + 15 fine |

---

## 3. Where is Each Parameter Defined?

The project follows a strict **single source of truth** architecture:

```text
config.py  ──>  model.py  ──>  train.py
```

Every parameter is defined in `config.py` and passed directly into the model constructor `Model(cfg=cfg)` without hard-coded duplication.

### Exact File & Variable Locations:

1. **Input Dimension:**
   - Variable: `input_dim = 44` (or `INPUT_DIM = 44`)
   - Location: `ml_models/<model_id>/config.py` and `model/model_config.py`
   - Usage: First layer input dimension (`Linear(cfg.input_dim, ...)`, `GRU(input_size=cfg.input_dim, ...)`, `LSTM(input_size=cfg.input_dim, ...)`)

2. **Hidden / Model Dimension:**
   - Variable: `hidden_dim = 32` (for GRU, LSTM, Hazard) or `d_model = 32 / 64` (for Transformers)
   - Location: `ml_models/<model_id>/config.py`
   - Usage: Internal representation width across recurrent cells, self-attention keys/queries/values, and hidden states.

3. **Number of Layers / Blocks:**
   - Variable: `num_layers = 2` (GRU, LSTM, M5) or `num_layers = 3` (M6 Champion)
   - Location: `ml_models/<model_id>/config.py`
   - Usage: `nn.GRU(num_layers=cfg.num_layers)`, `nn.LSTM(num_layers=cfg.num_layers)`, `nn.TransformerEncoder(..., num_layers=cfg.num_layers)`

4. **Attention Heads:**
   - Variable: `nhead = 2` (M6 Champion, head dimension = 16) or `nhead = 4` (M5)
   - Location: `ml_models/<model_id>/config.py` and `model/model_config.py`
   - Usage: `TransformerEncoderLayer(..., nhead=cfg.nhead)`

5. **Feedforward Network (FFN) Dimension:**
   - Variable: `dim_feedforward = 64` (M6 Champion) or `dim_feedforward = 128` (M5)
   - Location: `ml_models/<model_id>/config.py`
   - Usage: Feedforward expansion layer width between attention operations (`Linear(d_model, dim_feedforward)`).

6. **Dropout Rate:**
   - Variable: `dropout = 0.10` (M3, M4, M5) or `dropout = 0.20` (M6 Champion)
   - Location: `ml_models/<model_id>/config.py`
   - Usage: Inter-layer dropout, attention weight dropout, and prediction head dropout to regularize training.

7. **Output Heads & Multi-Task Loss Weighting:**
   - Variable: `head_a_output_dim = 1`, `horizon = 6`, `loss_lambda = 0.30`
   - Location: `ml_models/M6_multitask/config.py` and `model/model_config.py`
   - Usage:
     - `self.head_a = Linear(..., 1)` (Binary distress logit)
     - `self.head_b = Linear(..., 6)` (6-month forward discrete hazard logits)
     - `total_loss = loss_a + cfg.loss_lambda * loss_b`

---

## 4. Layer Trace for Every Neural Model

### M3_gru
```text
Input [B, 12, 44]
  ↓
Layer 1: 2-Layer GRU (input_dim=44, hidden_dim=32, num_layers=2)
  ↓
Layer 2: Dropout (p=0.10)
  ↓
Layer 3: Sequence Pooling (Final timestep hidden state h_12 [B, 32])
  ↓
Layer 4: Linear Prediction Head (Linear(32, 1) producing scalar distress logit)
```

### M4_lstm
```text
Input [B, 12, 44]
  ↓
Layer 1: 2-Layer LSTM with 4 Gating Mechanisms (input_dim=44, hidden_dim=32, num_layers=2)
         (Maintains separate Cell State c_t and Hidden State h_t)
  ↓
Layer 2: Dropout (p=0.10)
  ↓
Layer 3: Sequence Pooling (Final timestep hidden state h_12 [B, 32])
  ↓
Layer 4: Linear Prediction Head (Linear(32, 1) producing scalar distress logit)
```

### M5_transformer
```text
Input [B, 12, 44]
  ↓
Layer 1: Input Linear Projection (Linear(44, 64) -> [B, 12, 64])
  ↓
Layer 2: Temporal Tokens (Prepend learnable [CLS] token + Sinusoidal Positional Encoding -> [B, 13, 64])
  ↓
Layer 3: 2 × Pre-LN Transformer Blocks (nhead=4, d_model=64, dim_feedforward=128, GELU, norm_first=True)
  ↓
Layer 4: Sequence Pooling (Extract [CLS] token at index 0 -> [B, 64])
  ↓
Layer 5: Prediction Head (Linear(64, 32) -> GELU -> Dropout(0.1) -> Linear(32, 1) producing distress logit)
```

### M6_multitask (Proposed Champion Architecture)
```text
Input [B, 12, 44]
  ↓
Layer 1: Input Linear Projection (Linear(44, 32) -> [B, 12, 32])
  ↓
Layer 2: Embeddings (Prepend learnable [CLS] token + Positional Encoding -> [B, 13, 32])
  ↓
Layer 3: 3 × Pre-LN Transformer Blocks (nhead=2, d_model=32, dim_feedforward=64, GELU, norm_first=True)
  ↓
Layer 4: Sequence Pooling (Extract [CLS] token representation at index 0 -> [B, 32])
  ↓
Layer 5: Shared Representation [B, 32]
  ├── Head A (Distress Head): Linear(32, 32) → GELU → Dropout(0.2) → Linear(32, 1) + Platt Scaling
  └── Head B (Hazard Head)  : Linear(32, 32) → GELU → Dropout(0.2) → Linear(32, 6)

Multi-Task Loss Combination:
  L_total = L_distress + 0.30 × L_hazard
```
