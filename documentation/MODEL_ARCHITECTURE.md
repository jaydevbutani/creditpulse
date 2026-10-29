# CreditPulse M6 Multi-Task Transformer

## Deep Learning Architectural Specification & Engineering Reference

This document provides a comprehensive, rigorous technical breakdown of the **CreditPulse M6_multitask Temporal Transformer** architecture, its dual-objective mathematical formulation, training dynamics, post-hoc calibration, and inference pipeline. It is structured specifically for academic thesis defense, faculty evaluation, and model governance review.

---

## 1. Architectural Overview & System Motivation

Commercial loan distress and default detection in longitudinal banking data involves two fundamental challenges:
1. **Temporal Dependency & Non-Linear Trajectories:** Credit distress does not manifest as a sudden isolated shock; it develops as a gradual sequence of declining cash inflows, escalating operating outflows, liquidity buffer exhaustion, and repeated overdraft excursions.
2. **Timing Uncertainty (The Survival Problem):** Standard binary classification predicts *if* default occurs within a fixed horizon (e.g., 6 months), but fails to inform credit analysts *when* the default is likely to happen within that window.

The **CreditPulse M6_multitask Transformer** solves both challenges simultaneously through a shared temporal encoder that feeds two specialized prediction heads:
- **Head A (Classification):** 6-month cumulative distress probability ($P(\text{Default}_{t \le 6})$).
- **Head B (Survival Analysis):** Discrete-time marginal hazard trajectory across future horizons ($h_1, h_2, \dots, h_6$).

```
Input Tokens [B, 12, 44]
          │
          ▼
   Dense Linear Projection  [B, 12, 32]
          │
          ├── + Positional Embedding (Temporal Recency) [B, 12, 32]
          └── + Loan Age Seasoning Embedding           [B, 12, 32]
          │
          ▼
   Prepend Learnable [CLS] Token                       [B, 13, 32]
          │
          ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 3x Pre-LayerNorm Transformer Blocks                         │
   │                                                             │
   │  LayerNorm ──► Multi-Head Self-Attention (2 Heads) ──► Add  │
   │  LayerNorm ──► Feed-Forward Network (GELU, 2x)     ──► Add  │
   └─────────────────────────────────────────────────────────────┘
          │
          ▼
   Final LayerNorm                                     [B, 13, 32]
          │
          ▼
   Extract [CLS] Representation Vector                 [B, 32]
          │
          ├────────────────────────────────┬───────────────────────────────┐
          ▼                                                                ▼
   Head A: 6-Month Distress                                         Head B: Discrete Hazard
   Dense(32, 32) -> GELU -> Dropout -> Dense(32, 1)                 Dense(32, 32) -> GELU -> Dropout -> Dense(32, 6)
          │                                                                │
          ▼                                                                ▼
   Scalar Logit z_A ∈ ℝ                                             Hazard Logits z_B ∈ ℝ⁶
          │                                                                │
   Platt Sigmoid Calibration                                        Sigmoid Activation
          │                                                                │
          ▼                                                                ▼
   P(Distress) ∈ [0, 1]                                             h_1, ..., h_6 ∈ [0, 1]⁶
```

---

## 2. Input Representation & Sequence Tokenization

### 2.1 Sequence Geometry
Each borrower account is observed across a sliding window of $W = 12$ monthly observation tokens:
$$\mathbf{X} \in \mathbb{R}^{B \times 12 \times 44}$$
where $B$ is the mini-batch size, $W = 12$ denotes temporal steps from $t-11$ to $t$ (the current cutoff date), and $D = 44$ is the dimension of the monthly behavioral feature vector.

### 2.2 Feature Vector Composition ($D = 44$)
The 44 monthly features capture longitudinal banking signals partitioned into 4 functional blocks:

| Feature Subspace | Dimensions | Representative Indicators | Economic Motivation |
| :--- | :---: | :--- | :--- |
| **Liquidity & Balance Scale** | 4 | Log end-of-month balance, minimum monthly balance, log inflows, log outflows | Establishes absolute operating scale and reserve buffer size. |
| **Cash Flow Velocity** | 3 | Net cash flow ratio $\frac{\text{Inflow} - \text{Outflow}}{\text{Inflow} + \text{Outflow}}$, Debt-service-to-income $\frac{\text{Payment}}{\text{Inflow}}$, Payment-to-balance ratio | Measures debt-servicing strain and cash burn rates. |
| **Overdraft & Stress Dynamics**| 4 | Days below zero balance, overdraft occurrence indicator, trough sub-zero severity, monthly transaction count | Measures acute liquidity crises and reliance on short-term overdraft credit. |
| **Temporal Deltas & Momentum** | 33 | 1-month and 3-month percentage shifts in liquidity, rolling standard deviations, payment stability | Captures second-order velocity (acceleration of financial deterioration). |

---

## 3. Embedding Pipeline & Shared Encoder

### 3.1 Input Token Linear Projection
The raw 44-dimensional feature vector is mapped into the model's hidden representation subspace ($d_{\text{model}} = 32$):
$$\mathbf{h}_t = \mathbf{W}_{\text{in}} \mathbf{x}_t + \mathbf{b}_{\text{in}}, \quad \mathbf{W}_{\text{in}} \in \mathbb{R}^{32 \times 44}, \quad \mathbf{b}_{\text{in}} \in \mathbb{R}^{32}$$

### 3.2 Dual Temporal Embeddings
Longitudinal credit risk depends both on **relative recency** within the observation window and **overall loan seasoning** from origination:
1. **Positional Encoding ($\mathbf{E}_{\text{pos}}$):** Learned relative positional embedding indicating proximity to the cutoff date ($t-0$ has distance 0, $t-11$ has distance 11).
2. **Loan Age Seasoning Encoding ($\mathbf{E}_{\text{age}}$):** Learned embedding of months elapsed since loan origination ($t_{\text{origination}}$). Banking facilities exhibit empirical hazard curves where default risk peaks around months 18–30 (the "seasoning hump").

$$\mathbf{z}_t^{(0)} = \mathbf{h}_t + \mathbf{E}_{\text{pos}}(t) + \mathbf{E}_{\text{age}}(\text{age}_t)$$

### 3.3 Learnable `[CLS]` Pooling Token
Unlike natural language processing with variable sentence lengths, CreditPulse prepends a learnable classification parameter $\mathbf{z}_{\text{cls}} \in \mathbb{R}^{1 \times 32}$:
$$\mathbf{Z}^{(0)} = \left[ \mathbf{z}_{\text{cls}} \;;\; \mathbf{z}_1^{(0)} \;;\; \dots \;;\; \mathbf{z}_{12}^{(0)} \right] \in \mathbb{R}^{B \times 13 \times 32}$$
The `[CLS]` token interacts with all 12 monthly tokens via self-attention, learning to aggregate global sequence distress without recency bias or the dilution inherent in arithmetic average pooling.

---

## 4. Pre-LayerNorm Transformer Encoder Blocks

The backbone comprises $N = 3$ stacked Transformer encoder layers using **Pre-LayerNorm** (Pre-LN) formulation:

$$\mathbf{Z}'^{(l)} = \mathbf{Z}^{(l-1)} + \text{Dropout}\left( \text{MHA}\left( \text{LayerNorm}(\mathbf{Z}^{(l-1)}) \right) \right)$$
$$\mathbf{Z}^{(l)} = \mathbf{Z}'^{(l)} + \text{Dropout}\left( \text{FFN}\left( \text{LayerNorm}(\mathbf{Z}'^{(l)}) \right) \right)$$

### Why Pre-LayerNorm?
In standard Post-LN Transformers, gradients passing through the residual connection can explode or vanish near the output layer, requiring delicate learning rate warmup schedules. Pre-LayerNorm maintains an unobstructed identity gradient path through the residual stream:
$$\frac{\partial \mathbf{Z}^{(L)}}{\partial \mathbf{Z}^{(0)}} = \mathbf{I} + \sum_{l=1}^L \frac{\partial \mathbf{f}_l}{\partial \mathbf{Z}^{(l-1)}}$$
This ensures uniform gradient propagation and robust convergence even with small batch sizes and tabular sequences.

### Multi-Head Self-Attention (MHA)
- **Attention Heads:** $H = 2$
- **Per-Head Dimension:** $d_k = d_{\text{model}} / H = 32 / 2 = 16$
- **Scaled Dot-Product Attention:**
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}$$
  where $\mathbf{M}$ is a key padding mask preventing attention into inactive or padded pre-origination months.

### Feed-Forward Network (FFN)
The feed-forward sub-layer expands $d_{\text{model}}$ by a multiplier of 2 ($d_{\text{ff}} = 64$):
$$\text{FFN}(\mathbf{u}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{u} + \mathbf{b}_1) + \mathbf{b}_2$$
- **Activation:** **Gaussian Error Linear Unit (GELU)**:
  $$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[ 1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right) \right]$$
  GELU provides smoother non-linear gradient flow than ReLU, avoiding dead neuron saturation on normalized financial tokens.

---

## 5. Dual Multi-Task Output Heads

After the final LayerNorm, the pooled representation vector is extracted from the `[CLS]` index:
$$\mathbf{r} = \mathbf{Z}_{[:, 0]}^{(3)} \in \mathbb{R}^{B \times 32}$$

### Head A: 6-Month Binary Distress Classification
Predicts whether the facility will enter default or critical distress within 6 months:
$$z_A = \mathbf{W}_{A2} \cdot \text{GELU}(\mathbf{W}_{A1} \mathbf{r} + \mathbf{b}_{A1}) + b_{A2} \quad \in \mathbb{R}$$

### Head B: Discrete-Time Marginal Hazard Trajectory
Predicts marginal conditional failure hazard across horizons $m \in \{1, 2, 3, 4, 5, 6\}$:
$$\mathbf{z}_B = \mathbf{W}_{B2} \cdot \text{GELU}(\mathbf{W}_{B1} \mathbf{r} + \mathbf{b}_{B1}) + \mathbf{b}_{B2} \quad \in \mathbb{R}^6$$
Marginal hazard probabilities are computed via sigmoid activation:
$$h_m = \sigma(z_{B, m}) = \frac{1}{1 + e^{-z_{B, m}}}$$
Cumulative incidence of distress across the 6-month horizon is:
$$S(m) = \prod_{k=1}^m (1 - h_k), \quad F(m) = 1 - S(m)$$

---

## 6. Multi-Task Loss Formulation

Training jointly optimizes both heads via a weighted multi-task objective:
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{HeadA}} + \lambda \times \mathcal{L}_{\text{HeadB}}$$

### 6.1 Head A Loss with Class Imbalance Weighting
Because default events in banking sequence datasets are rare (~4.2% prevalence), standard cross-entropy suffers from extreme majority-class bias. Head A uses weighted Binary Cross-Entropy:
$$\mathcal{L}_{\text{HeadA}} = - \frac{1}{B} \sum_{i=1}^B \left[ w_{\text{pos}} y_i \log \sigma(z_{A, i}) + (1 - y_i) \log(1 - \sigma(z_{A, i})) \right]$$
where the positive class weight is set to the empirical imbalance ratio:
$$w_{\text{pos}} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = 8.6221$$

### 6.2 Head B Discrete Hazard Loss
Head B is supervised using multi-label binary cross-entropy on observed event timings:
$$\mathcal{L}_{\text{HeadB}} = - \frac{1}{B} \sum_{i=1}^B \sum_{m=1}^6 \left[ y_{i, m}^{(B)} \log \sigma(z_{B, i, m}) + (1 - y_{i, m}^{(B)}) \log(1 - \sigma(z_{B, i, m})) \right]$$

### 6.3 Weighting Hyperparameter $\lambda = 0.30$
$\lambda$ balances gradient magnitudes between the single scalar classification task (Head A) and the 6-horizon hazard task (Head B), preventing hazard gradients from overpowering the primary distress classification objective.

---

## 7. Post-Hoc Platt Probability Calibration

Deep neural networks trained with positive class weights produce calibrated rankings (high PR-AUC) but uncalibrated absolute probabilities (logits are shifted towards the positive class). 

CreditPulse applies **Platt Scaling** (monotonic logistic regression over raw logits):
$$P(\text{Distress} \mid z_A) = \frac{1}{1 + \exp(-(a \cdot z_A + b))}$$
- **Slope ($a$):** $0.145613$
- **Intercept ($b$):** $-3.371945$

This maps unconstrained neural logits into empirical default frequencies, reducing the Brier score to $0.0194$ and Expected Calibration Error (ECE) to $0.0124$.

### Decision Threshold at Top 10% Queue
Credit review teams have fixed investigative capacity. The decision threshold is configured on the validation set to flag the top 10% highest-risk facilities:
$$\tau_{10\%} = 0.02847 \quad (\approx 2.85\% \text{ calibrated risk})$$
Facilities with $P(\text{Distress}) \ge 2.85\%$ trigger an automatic **MODEL ALERT**. On out-of-time test data, this threshold captures **77.8% of all defaulting loans** (Capture@10%).

---

## 8. Summary Model Specification Table

| Architectural Hyperparameter | Exact Configured Value | Source in Code |
| :--- | :--- | :--- |
| **Model Name** | `M6_multitask` | `model/model_config.py` |
| **Model Family** | Temporal Multi-Task Transformer | `model/model_config.py` |
| **Total Parameters** | `33,383` parameters | `model/architecture.py` |
| **Trainable Parameters** | `33,383` (100% trainable) | `model/architecture.py` |
| **Input Feature Dimension ($D$)**| `44` features / monthly token | `model/model_config.py:INPUT_DIM` |
| **Sequence Length ($W$)** | `12` monthly tokens | `model/model_config.py:SEQUENCE_LENGTH` |
| **Hidden Embedding ($d_{\text{model}}$)** | `32` dimensions | `model/model_config.py:D_MODEL` |
| **Transformer Layers ($N$)** | `3` Pre-LayerNorm blocks | `model/model_config.py:N_LAYERS` |
| **Attention Heads ($H$)** | `2` heads ($d_k = 16$) | `model/model_config.py:N_HEADS` |
| **FFN Expansion Multiplier** | `2x` ($d_{\text{ff}} = 64$) | `model/model_config.py:FEEDFORWARD_MULT` |
| **Dropout Rate** | `0.2616` (26.16%) | `model/model_config.py:DROPOUT` |
| **Non-Linear Activation** | `GELU` (Gaussian Error Linear Unit) | `model/model_config.py:ACTIVATION` |
| **Sequence Pooling Strategy** | `[CLS]` Token Representation | `model/model_config.py:POOLING` |
| **Head A Output** | 6-Month Binary Distress Logit ($1$ unit) | `model/architecture.py:head_a` |
| **Head B Output** | Discrete Hazard Logits ($6$ units) | `model/architecture.py:head_b` |
| **Optimizer** | `AdamW` ($\text{lr} = 2.18 \times 10^{-4}, \text{decay} = 1.43 \times 10^{-4}$) | `model/model_config.py:TRAINING_CONFIG` |
| **Loss Formulation** | $\mathcal{L}_{\text{HeadA}} + 0.30 \times \mathcal{L}_{\text{HeadB}}$ | `model/train.py` |
| **Positive Class Weight** | $w_{\text{pos}} = 8.6221$ | `model/model_config.py:TRAINING_CONFIG` |
| **Platt Scaling Slope ($a$)**| $0.145613$ | `model/model_config.py:PLATT_CALIBRATION` |
| **Platt Intercept ($b$)** | $-3.371945$ | `model/model_config.py:PLATT_CALIBRATION` |
| **Alert Threshold** | $0.02847$ ($2.85\%$ probability) | `model/model_config.py:ALERT_THRESHOLD_10PCT` |
| **Random Seed / Ensemble** | Seeds `[0, 1, 2, 3, 4]` (5-fold ensemble) | `model/model_config.py:TRAINING_CONFIG` |

---

## 9. Feature Attribution via Integrated Gradients

To explain individual facility predictions to credit examiners without black-box opacity, CreditPulse implements **Path-Integrated Gradients** (Sundararajan et al., 2017).

Attribution for monthly feature $j$ across the sequence is computed by integrating gradients along the straight-line interpolation path from a neutral zero baseline $\mathbf{x}^{(0)}$ to the borrower's observed sequence $\mathbf{x}$:
$$\text{Attr}_j(\mathbf{x}) = (x_j - x_j^{(0)}) \times \int_{0}^1 \frac{\partial F(\mathbf{x}^{(0)} + \alpha (\mathbf{x} - \mathbf{x}^{(0)}))}{\partial x_j} d\alpha$$

This satisfies two fundamental axiomatic properties required by banking supervisory guidelines:
1. **Completeness:** $\sum_j \text{Attr}_j(\mathbf{x}) = F(\mathbf{x}) - F(\mathbf{x}^{(0)})$, ensuring the sum of feature attributions equals the model's total logit departure from baseline.
2. **Implementation Invariance:** Attributions depend strictly on the functional mapping of the neural network, invariant to internal implementation details.

These raw attribution scores are aggregated across the 12-month tokens into human-understandable risk drivers (e.g. *Liquidity Buffer Deterioration*, *Outflow Acceleration*, *Overdraft Frequency*).
