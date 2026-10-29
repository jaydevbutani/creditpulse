# M3: 2-Layer Gated Recurrent Unit (GRU)

## Model Overview
- **Name:** Gated Recurrent Unit (`M3_gru`)
- **Type:** Recurrent Neural Network (RNN)
- **Input:** 12-month panel trajectory $[B, 12, 44]$
- **Hidden Dimension:** 32 (2 stacked recurrent layers)
- **Trainable Parameters:** ~13,921

## Layer Architecture
1. **Recurrent Stack:** `nn.GRU(input_size=44, hidden_size=32, num_layers=2, batch_first=True, dropout=0.1)`
2. **Pooling / Terminal State:** Final step hidden state $\mathbf{h}_{12}$
3. **Classification Head:** `Dropout(0.1)` $\to$ `Linear(32, 1)`

## Hyperparameter Grid Search
- **Learning Rates:** $10^{-3}, 5 \times 10^{-4}, 10^{-4}$
- **Optimizer:** `AdamW(weight_decay=1e-4)`
- **Loss:** `BCEWithLogitsLoss`

## Keras Equivalent Specification
```python
# Conceptual Keras equivalent for viva explanation
inputs = keras.Input(shape=(12, 44), name="monthly_sequences")
x = keras.layers.GRU(32, return_sequences=True, dropout=0.1)(inputs)
x = keras.layers.GRU(32, return_sequences=False, dropout=0.1)(x)
x = keras.layers.Dropout(0.1)(x)
logits = keras.layers.Dense(1, activation=None, name="distress_logits")(x)
model = keras.Model(inputs=inputs, outputs=logits)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
              loss=keras.losses.BinaryCrossentropy(from_logits=True))
```
