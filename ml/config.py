"""CreditPulse Deep Learning Model & Training Configuration.
Provides a single, transparent, and presentable location for faculty review
exposing the multi-task Transformer architecture (M6_multitask), hyperparameters,
loss functions, and calibration parameters.
"""
from __future__ import annotations

# ==============================================================================
# 1. ARCHITECTURE HYPERPARAMETERS (Multi-Task Pre-LN Transformer)
# ==============================================================================
MODEL_NAME = "M6_multitask"
MODEL_FAMILY = "Temporal Multi-Task Transformer"
MODEL_VERSION = "M6_multitask-ens5-cfg29476ad6e2-data1a2c09-f49070-9277f1"

# Input sequence dimensions
SEQUENCE_LENGTH = 12       # 12 historical monthly observation tokens (t-11 to t)
INPUT_DIM = 44             # Number of behavioral features per monthly token (cash flow, balances, debt service)
MAX_POS_WINDOW = 24        # Maximum positional window supported for token embeddings

# Neural network dimensions
D_MODEL = 32               # Hidden embedding dimension (d_model)
N_LAYERS = 3               # Number of stacked Pre-LayerNorm Transformer blocks
N_HEADS = 2                # Number of multi-head self-attention heads
FEEDFORWARD_MULT = 2       # Feedforward expansion multiplier (ff_dim = 2 * d_model = 64)
DROPOUT = 0.2616           # Attention & residual dropout rate (26.16%)
ACTIVATION = "GELU"        # Non-linear activation in feedforward sub-layers
POOLING = "cls"            # Learnable [CLS] token pooling for sequence representation

# Multi-Task Output Heads
HORIZON = 6                # Future evaluation horizon (6 months)
HEAD_A_NAME = "6-Month Calibrated Distress Probability"
HEAD_A_OUTPUT_DIM = 1      # Scalar distress logit
HEAD_B_NAME = "Discrete-Time Marginal Hazard Trajectory (+1m to +6m)"
HEAD_B_OUTPUT_DIM = 6      # 6 discrete interval hazard logits

# ==============================================================================
# 2. TRAINING CONFIGURATION & OPTIMIZATION
# ==============================================================================
TRAINING_CONFIG = {
    "epochs": 8,
    "batch_size": 1024,
    "learning_rate": 0.00021775,
    "weight_decay": 0.00014255,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingWithWarmup",
    "warmup_fraction": 0.10,
    "pos_weight": 8.6221,        # Positive class weight addressing low default prevalence (~4%)
    "loss_lambda": 0.30,         # Multi-task loss weight: Total Loss = Loss_HeadA + lambda * Loss_HeadB
    "seed_ensemble": [0, 1, 2, 3, 4], # 5-seed deep ensemble for epistemic variance reduction
    "patience": 5,
    "train_split": "1993-01 to 1997-12 (Czech Banking Cohort)",
    "test_split": "P1 Out-of-Time Test Set (1998-01 to 1998-12)"
}

# ==============================================================================
# 3. POST-HOC CALIBRATION & DECISION BOUNDARIES
# ==============================================================================
# Platt scaling (logistic sigmoid regression over raw ensemble logits)
# P(Distress | z) = 1 / (1 + exp(-(a * z + b)))
PLATT_CALIBRATION = {
    "method": "Platt Scaling (Monotonic Logistic Calibration)",
    "a": 0.1456126868724823,
    "b": -3.3719449043273926,
    "target_brier_score": 0.0194,
    "target_ece": 0.0124
}

# Decision threshold configured on validation set to fill 10% queue capacity
ALERT_THRESHOLD_10PCT = 0.028466200456023216 # ~2.85% calibrated probability

# ==============================================================================
# 4. BENCHMARK PERFORMANCE METRICS (P1 Out-of-Time Test)
# ==============================================================================
PERFORMANCE_BENCHMARKS = {
    "pr_auc": 0.714,
    "pr_auc_95_ci": "[0.582, 0.824]",
    "roc_auc": 0.892,
    "brier_score": 0.0194,
    "expected_calibration_error": 0.0124,
    "capture_at_top_10pct": 0.778,      # Captures 77.8% of all future defaulting facilities
    "median_lead_time_months": 4.0,     # 4.0 months advance warning before distress event
    "event_loans_test": 9
}
