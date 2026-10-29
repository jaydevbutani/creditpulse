# CreditPulse Deep Learning Architecture & Training Documentation

This directory exposes the core Deep Learning model configuration, multi-task Transformer architecture, loss formulation, calibration methodology, and evaluation metrics for academic evaluation and faculty review.

---

## 1. Model Architecture (`M6_multitask`)

The CreditPulse model is a **Temporal Multi-Task Transformer** with Pre-Layer Normalization operating on 12-month behavioral token sequences.

```text
12-Month Behavioral Sequence [B, 12, 44]
               │
               ▼
   Linear Token Projection (44 → 32)
               │
               ▼
   + Positional Distance Embedding (0..23)
   + Loan Origination Age Embedding (-24..72)
   + [CLS] Token Prepending
               │
               ▼
   ┌─────────────────────────────────────────┐
   │ 3 × Pre-LN Transformer Blocks           │
   │  • LayerNorm(d_model=32)                │
   │  • Multi-Head Self-Attention (2 heads)  │
   │  • Residual Connection & Dropout (0.26) │
   │  • LayerNorm(d_model=32)                │
   │  • Feedforward (32 → 64 → 32) + GELU    │
   │  • Residual Connection                  │
   └─────────────────────────────────────────┘
               │
               ▼
   LayerNorm & [CLS] Token Pooling [B, 32]
               │
       ┌───────┴────────────────────────┐
       ▼                                ▼
Head A (Binary Distress)       Head B (Discrete Survival)
Linear(32 → 1)                 Linear(32 → 6)
Calibrated 6M Distress Risk    Monthly Hazard Shape (+1m..+6m)
```

---

## 2. Multi-Task Configuration & Objectives

| Component | Head A | Head B |
| :--- | :--- | :--- |
| **Prediction Target** | 6-Month Binary Financial Distress | Discrete Survival Hazards across Horizons +1m, +2m, +3m, +4m, +5m, +6m |
| **Output Dimension** | 1 (Scalar Logit) | 6 (Vector of Interval Hazards) |
| **Loss Function** | Weighted Binary Cross-Entropy with `pos_weight = 8.622` | Discrete-Time Survival Negative Log-Likelihood (`hazard_nll`) |
| **Loss Weight ($\lambda$)** | $1.0$ | $0.30$ |

### Joint Multi-Task Loss Formulation
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{HeadA}}(\hat{y}, y; w_{\text{pos}}) + \lambda \cdot \mathcal{L}_{\text{HeadB}}(\hat{h}, \text{event}_k, \text{follow}_k)$$

Where:
$$\mathcal{L}_{\text{HeadB}} = -\sum_{i=1}^B \left[ \sum_{u < \text{event}_i} \log(1 - h_u) + \mathbf{1}_{[\text{event}_i > 0]} \log(h_{\text{event}_i}) \right]$$

---

## 3. Training & Optimization Hyperparameters

All hyperparameters are declared in [`ml/config.py`](file:///c:/Users/jayde/Downloads/creditpulse/creditpulse/ml/config.py):

- **Model Dimensions**: $d_{\text{model}} = 32$, Layers $= 3$, Attention Heads $= 2$, Feedforward $= 64$
- **Dropout Rate**: $0.2616$ (applied to attention weights and residual branches)
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$)
- **Learning Rate**: $2.1775 \times 10^{-4}$ with Cosine Annealing and 10% Warmup
- **Weight Decay**: $1.4255 \times 10^{-4}$
- **Batch Size**: 1,024 stratified sequences
- **Training Epochs**: 8 epochs with early stopping patience = 5
- **Ensemble**: 5 random seeds ($s_0, s_1, s_2, s_3, s_4$) for epistemic variance mitigation

---

## 4. Post-Hoc Calibration & Decision Thresholding

To ensure predicted probabilities match empirical default frequencies, ensemble raw logits $z$ are calibrated via Platt Scaling:
$$P(\text{Distress} \mid z) = \frac{1}{1 + \exp(-(a \cdot z + b))}$$
- Parameter $a = 0.14561$
- Parameter $b = -3.37194$
- **10% Queue Alert Threshold**: $0.02847$ (fills top 10% queue capacity on validation data)

---

## 5. Reproducible Execution

Run the end-to-end forward/backward pipeline test:
```powershell
python -m ml.train
```
