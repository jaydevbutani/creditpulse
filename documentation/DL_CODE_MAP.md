# CreditPulse — Deep Learning Code Map

> Quick-reference guide for teacher and examiner code navigation during viva defense.

---

## 1. Quick File Locator

| Question | File to Open in VS Code | Key Components / Symbols |
| :--- | :--- | :--- |
| **Where is the model architecture?** | [`model/architecture.py`](file:///c:/Users/jayde/Downloads/creditpulse/creditpulse/model/architecture.py) | Class `MultiTaskDistressTransformer` (Pre-LN, `[CLS]` pooling, Dual Heads A & B) |
| **Where are the hyperparameters?** | [`model/model_config.py`](file:///c:/Users/jayde/Downloads/creditpulse/creditpulse/model/model_config.py) | `INPUT_DIM=44`, `SEQUENCE_LENGTH=12`, `D_MODEL=32`, `N_LAYERS=3`, `N_HEADS=2`, `LEARNING_RATE=0.0002` |
| **Where are features created?** | [`model/preprocessing.py`](file:///c:/Users/jayde/Downloads/creditpulse/creditpulse/model/preprocessing.py) | Function `extract_monthly_features()`, `FEATURE_REGISTRY` (44 documented features) |
| **Where is the dataset loaded?** | [`model/dataset.py`](file:///c:/Users/jayde/Downloads/creditpulse/creditpulse/model/dataset.py) | `LoanSequenceDataset`, `create_dataloaders()` |
| **Where is the training loop?** | [`model/train.py`](file:///c:/Users/jayde/Downloads/creditpulse/creditpulse/model/train.py) | `run_training_pipeline()`, AdamW loop, multi-task loss calculation |
| **Top-level training runner?** | [`train.py`](file:///c:/Users/jayde/Downloads/creditpulse/creditpulse/train.py) | Standalone script (`python train.py`) |
| **Where is evaluation?** | [`model/evaluate.py`](file:///c:/Users/jayde/Downloads/creditpulse/creditpulse/model/evaluate.py) | PR-AUC, ROC-AUC, Brier score, Expected Calibration Error (ECE) |
| **Where is production inference?** | [`model/inference.py`](file:///c:/Users/jayde/Downloads/creditpulse/creditpulse/model/inference.py) | `DistressScorer`, Platt calibration forward pass |
| **Where is explainability?** | [`model/explainability.py`](file:///c:/Users/jayde/Downloads/creditpulse/creditpulse/model/explainability.py) | `compute_integrated_gradients()`, `extract_top_drivers()` |

---

## 2. End-to-End Data & Model Pipeline Flow

```
                      Raw Monthly Banking Records
            (Balances, Inflows, Outflows, Overdraft Days, Loan Terms)
                                  │
                                  ▼
                     Feature Preprocessing Pipeline
                     (model/preprocessing.py)
                                  │
                                  ▼
                12-Month Temporal Sequence Matrix
                     Tensor Shape: [Batch, 12, 44]
                                  │
                                  ▼
                   M6 Multi-Task Transformer
                     (model/architecture.py)
         • Input Projection: [44 -> 32]
         • Positional & Loan Age Embeddings
         • 3 Pre-LayerNorm Transformer Blocks (2 Heads, GELU)
         • Learnable [CLS] Sequence Pooling
                                  │
                                  ▼
                       Dual Prediction Heads
                                  ├──► Head A: 6-Month Binary Distress Logit
                                  └──► Head B: Monthly Hazard Horizons (+1m to +6m)
                                  │
                                  ▼
                     Platt Scaling Calibration
                 P = 1 / (1 + exp(-(0.1456 * z - 3.3719)))
                                  │
                                  ▼
                  Calibrated Distress Probability (%)
                   (Risk Tier: Normal / Watch / Alert)
                                  │
                                  ▼
                   Human-in-the-Loop Review & Flag
                  (Independent Examiner Discretion)
                                  │
                                  ▼
                     Portfolio Surveillance Queue
                         & Audit Case History
```

---

## 3. Standalone Verification Commands

```powershell
# 1. Verify Model Parameter Count (33,383) & Tensor Flow
python -m model.architecture

# 2. Run Supervised PyTorch Training on Dataset
python train.py

# 3. Verify Out-of-Time Test Evaluation Benchmarks
python -m model.evaluate

# 4. Verify Platt-Calibrated Production Inference Forward Pass
python -m model.inference

# 5. Verify Path-Integrated Gradients Explainability
python -m model.explainability
```
