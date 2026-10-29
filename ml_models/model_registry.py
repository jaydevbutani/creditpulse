"""CreditPulse Model Registry.
Contains the official benchmark model registry of the 11 compared models.
"""
from __future__ import annotations

MODELS = [
    "M0_prior",
    "M1_logreg",
    "M2_lgbm",
    "M2_xgb",
    "S1_cox",
    "S2_hazard",
    "M3_gru",
    "M4_lstm",
    "M5_transformer",
    "M6_multitask",
    "M7_pretrained",
]

CHAMPION_MODEL = "M6_multitask"

MODEL_DESCRIPTIONS = {
    "M0_prior": "Empirical prior baseline predicting historical default prevalence.",
    "M1_logreg": "L2-regularized Logistic Regression on 12-month aggregated tabular features.",
    "M2_lgbm": "LightGBM Gradient Boosted Decision Trees on tabular feature aggregations.",
    "M2_xgb": "XGBoost Gradient Boosted Decision Trees with histogram split binning.",
    "S1_cox": "Semi-parametric Cox Proportional Hazards regression for time-to-distress.",
    "S2_hazard": "Discrete-time multi-period logistic hazard survival model (+1m to +6m).",
    "M3_gru": "2-layer Gated Recurrent Unit (GRU) processing longitudinal sequence tokens.",
    "M4_lstm": "2-layer Long Short-Term Memory (LSTM) with sequence representation pooling.",
    "M5_transformer": "Single-task temporal Transformer encoder with self-attention on Head A only.",
    "M6_multitask": "Proposed multi-task Pre-LN Transformer with [CLS] pooling and dual heads (Distress + Hazard).",
    "M7_pretrained": "Transfer-learning sequence backbone fine-tuned for distress classification.",
}
