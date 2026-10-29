# CreditPulse Deep Learning Architecture & Training Documentation

This directory houses the PyTorch implementation of the **M6 Multi-Task Distress Transformer (`M6_multitask`)**.

---

## 1. Directory Structure

```text
model/
├── model_config.py     # Central hyperparameters, loss weights, calibration values
├── architecture.py     # Pre-LN MultiTask Transformer neural network
├── dataset.py          # PyTorch Sequence Dataset & DataLoader
├── preprocessing.py    # 44-dimensional temporal token construction
├── train.py            # Runnable multi-task training script
├── evaluate.py         # Out-of-time test evaluation and metrics
├── inference.py        # Single-facility & batch scoring engine with Platt calibration
├── explainability.py   # Path-integrated gradients feature attribution
└── README.md           # This academic documentation
```

---

## 2. Model Architecture Specifications

* **Topology**: Pre-LayerNorm Multi-Head Self-Attention Transformer Encoder.
* **Input Window**: 12 monthly tokens ($t-11$ to $t$).
* **Features per Step**: 44 behavioral variables covering liquidity ratios, cash burn velocity, overdraft occurrences, and debt-service-to-income bounds.
* **Embedding Dimension ($d_{\text{model}}$)**: 32 dense projection units.
* **Depth**: 3 stacked Pre-LN Transformer encoder blocks.
* **Attention Heads**: 2 self-attention heads ($d_k = 16$).
* **Dropout**: 0.2616 (attention and residual paths).
* **Pooling**: Learnable prefix `[CLS]` token representation.

### Multi-Task Dual Output Heads

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{HeadA}} + 0.30 \times \mathcal{L}_{\text{HeadB}}$$

1. **Head A (6-Month Distress Classification)**:
   Scalar logit trained with positive class weight $w = 8.622$ to penalize false negatives under class imbalance.
2. **Head B (Discrete-Time Hazard Trajectory)**:
   6-dimensional logit vector modeling monthly default timing probability over future horizons $+1\text{m}$ to $+6\text{m}$.

---

## 3. Post-Hoc Probability Calibration

Raw Transformer logits $z$ are calibrated via Platt monotonic logistic scaling:

$$P(\text{Distress} \mid z) = \frac{1}{1 + \exp(-(0.145613 \cdot z - 3.371945))}$$

* **Alert Threshold**: $0.02847$ ($2.85\%$) calibrated probability, tuned on the validation split to saturate the top 10% review queue capacity.
* **Target Brier Score**: $0.0194$.
* **Expected Calibration Error (ECE)**: $0.0124$.

---

## 4. How to Run the Training Script

Run directly from the VS Code terminal or PowerShell:

```bash
# Run top-level entry point
python train.py

# Or as module
python -m model.train
```
