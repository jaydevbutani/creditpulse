# S2: Discrete-Time Neural Hazard Survival Model

## Model Overview
- **Name:** Discrete-Time Neural Hazard (`S2_hazard`)
- **Type:** Multi-period discrete survival neural network
- **Target:** 6 forward monthly hazards $[h_1, h_2, \dots, h_6]$
- **Cumulative Distress Probability:**
  $$P(T \le H \mid \mathbf{X}) = 1 - \prod_{t=1}^H (1 - h_t)$$

## Layer Architecture & Parameter Count
1. **Input Projection:** `Linear(44, 32)` + `ReLU` + `Dropout(0.1)`
2. **Temporal Aggregation:** Sequence pooling across 12 monthly steps
3. **Hazard MLP Head:** `Linear(32, 32)` $\to$ `ReLU` $\to$ `Linear(32, 6)`
- **Total Parameters:** ~2,726

## Hyperparameter Grid Search
- **Learning Rates:** $10^{-3}, 5 \times 10^{-4}, 10^{-4}$
- **Optimizer:** `AdamW(weight_decay=1e-4)`
- **Criterion:** `BCEWithLogitsLoss`

## Keras Equivalent Specification
```python
# Conceptual Keras equivalent for viva explanation
inputs = keras.Input(shape=(12, 44), name="monthly_features")
x = keras.layers.Dense(32, activation="relu")(inputs)
pooled = keras.layers.GlobalAveragePooling1D()(x)
hidden = keras.layers.Dense(32, activation="relu")(pooled)
hazard_logits = keras.layers.Dense(6, name="hazard_logits")(hidden)
model = keras.Model(inputs=inputs, outputs=hazard_logits)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
              loss=keras.losses.BinaryCrossentropy(from_logits=True))
```
