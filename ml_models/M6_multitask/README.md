# M6: Multi-Task Pre-LN Temporal Transformer (Proposed Champion)

## Model Overview
- **Name:** Multi-Task Pre-LN Temporal Distress Transformer (`M6_multitask`)
- **Type:** Multi-Task Self-Attention Sequence Architecture
- **Novelty:** Formulates loan monitoring as **continuous temporal distress surveillance rather than one-shot credit scoring**. Jointly predicts:
  1. **Head A:** 12-month forward cumulative distress probability $P(\text{Distress} \le 12 \text{m})$.
  2. **Head B:** Discrete hazard trajectory across future months $[h_1, h_2, \dots, h_6]$.
- **Calibration:** Post-hoc Platt Scaling ($a \cdot z + b$) fitted on validation logits to minimize Expected Calibration Error (ECE) under CECL / Basel III guidelines.
- **Trainable Parameters:** ~74,297

## Layer Architecture & Academic Sections
1. **Section 1 (Input Projection):** `Linear(44, 64)` projects monthly behavioral vectors into continuous embedding space.
2. **Section 2 (Embeddings):** Learnable `[CLS]` token + Sinusoidal Positional Encoding $\in \mathbb{R}^{13 \times 64}$.
3. **Section 3 (Pre-LN Encoder):** 2 Transformer Encoder blocks with Multi-Head Self-Attention ($h=4, d_k=16$), GELU feedforward ($d_{\text{ff}}=128$), Pre-LayerNorm (`norm_first=True`).
4. **Section 4 (Head A - Distress):** `Linear(64, 32)` $\to$ `GELU` $\to$ `Dropout(0.1)` $\to$ `Linear(32, 1)` + Platt scaling.
5. **Section 5 (Head B - Hazard):** `Linear(64, 32)` $\to$ `GELU` $\to$ `Dropout(0.1)` $\to$ `Linear(32, 6)`.

## Multi-Task Optimization Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{distress}}(y, \hat{y}) + \lambda \mathcal{L}_{\text{hazard}}(\mathbf{h}, \hat{\mathbf{h}})$$
where $\lambda = 0.30$. Auxiliary hazard loss acts as an inductive regularizer, constraining latent representations to capture the temporal velocity of distress progression.

## Keras Equivalent Specification
```python
# Conceptual Keras equivalent for viva explanation
inputs = keras.Input(shape=(12, 44), name="monthly_features")
x = keras.layers.Dense(64)(inputs)
attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
x = keras.layers.LayerNormalization()(x + attn)
ffn = keras.layers.Dense(128, activation="gelu")(x)
ffn = keras.layers.Dense(64)(ffn)
x = keras.layers.LayerNormalization()(x + ffn)
cls_rep = keras.layers.GlobalAveragePooling1D()(x)

# Head A: 12-month Distress
head_a = keras.layers.Dense(32, activation="gelu")(cls_rep)
out_distress = keras.layers.Dense(1, activation="sigmoid", name="distress_prob")(head_a)

# Head B: 6-month Hazard
head_b = keras.layers.Dense(32, activation="gelu")(cls_rep)
out_hazard = keras.layers.Dense(6, activation="sigmoid", name="hazard_probs")(head_b)

model = keras.Model(inputs=inputs, outputs=[out_distress, out_hazard])
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
    loss={"distress_prob": "binary_crossentropy", "hazard_probs": "binary_crossentropy"},
    loss_weights={"distress_prob": 1.0, "hazard_probs": 0.30}
)
```
