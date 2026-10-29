# M7: Pre-trained Sequence Model (Self-Supervised Denoising Fine-Tuned)

## 1. What is this model?
`M7_pretrained` is an in-domain self-supervised benchmark (not an external billion-parameter foundation model). It demonstrates the transfer learning paradigm for longitudinal banking panel data by pretraining a bidirectional temporal backbone and fine-tuning it for distress discrimination.

## 2. Answers to Viva Questions
- **What was pretrained?**
  A 2-layer Bidirectional GRU (`hidden_dim=48`, combined latent representation = 96) and a reconstruction linear projection head.
- **How was it pretrained?**
  Via masked sequence denoising: 15% of feature values across the 12-month sequence $[B, 12, 44]$ are randomly masked out.
- **What was the input?**
  12-month panel trajectory $[B, 12, 44]$ corrupted with random zeros.
- **What was the pretraining objective?**
  Mean Squared Error ($\text{MSELoss}$) between the uncorrupted input sequence and reconstructed sequence.
- **How was it fine-tuned?**
  The reconstruction head was removed and replaced with a classification head (`Linear(96, 32) -> ReLU -> Dropout -> Linear(32, 1)`). The entire model was fine-tuned on the supervised distress labels using binary cross-entropy.
- **What parameters were frozen/unfrozen?**
  All 79,261 parameters were unfrozen for end-to-end gradient updates with AdamW.
- **What learning rate was used?**
  Pretraining used $\text{lr} = 10^{-3}$; fine-tuning was swept across $\{10^{-3}, 5 \times 10^{-4}, 10^{-4}\}$ with validation selection.

## 3. Conceptual Keras Equivalent (For Viva Reference Only)
```python
# Conceptual Keras equivalent for viva explanation (PyTorch was used for execution)
inputs = keras.Input(shape=(12, 44), name="monthly_sequences")
x = keras.layers.Bidirectional(keras.layers.GRU(48, return_sequences=True))(inputs)
x = keras.layers.LayerNormalization()(x)
pooled = keras.layers.GlobalAveragePooling1D()(x)
dense = keras.layers.Dense(32, activation="relu")(pooled)
logits = keras.layers.Dense(1, activation=None)(dense)
model = keras.Model(inputs=inputs, outputs=logits)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
              loss=keras.losses.BinaryCrossentropy(from_logits=True))
```
