# M5: Single-Task Temporal Transformer

## Model Overview
- **Name:** Single-Task Temporal Transformer (`M5_transformer`)
- **Type:** Pre-LN Self-Attention Sequence Classifier
- **Target:** 12-month binary distress classification (Head A only, no survival hazard loss)
- **Trainable Parameters:** ~37,857

## Layer Architecture
1. **Input Projection:** `Linear(44, 64)` projects monthly behavioral vectors into latent embedding space.
2. **Tokens & Position:** Prepend learnable `[CLS]` token $\in \mathbb{R}^{1 \times 64}$ + Sinusoidal positional encoding.
3. **Transformer Layers (Pre-LN):**
   - 2 Transformer Encoder layers
   - Multi-head self-attention: $h = 4$ attention heads, $d_{\text{k}} = 16$
   - Feedforward expansion: $d_{\text{ff}} = 128$ with `GELU`
   - Pre-LayerNorm architecture (`norm_first=True`)
4. **Classification Head:** `LayerNorm(64)` $\to$ `Linear(64, 32)` $\to$ `GELU` $\to$ `Linear(32, 1)`

## Hyperparameter Grid Search
- **Learning Rates:** $10^{-3}, 5 \times 10^{-4}, 10^{-4}$
- **Optimizer:** `AdamW(weight_decay=1e-4)`
- **Loss:** `BCEWithLogitsLoss`

## Keras Equivalent Specification
```python
# Conceptual Keras equivalent for viva explanation
inputs = keras.Input(shape=(12, 44), name="monthly_features")
x = keras.layers.Dense(64)(inputs)
# Positional encoding + multi-head attention
attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
x = keras.layers.LayerNormalization()(x + attn)
ffn = keras.layers.Dense(128, activation="gelu")(x)
ffn = keras.layers.Dense(64)(ffn)
x = keras.layers.LayerNormalization()(x + ffn)
cls_rep = keras.layers.GlobalAveragePooling1D()(x)
logits = keras.layers.Dense(1, activation=None)(cls_rep)
model = keras.Model(inputs=inputs, outputs=logits)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
              loss=keras.losses.BinaryCrossentropy(from_logits=True))
```
