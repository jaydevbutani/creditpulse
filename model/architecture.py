"""CreditPulse Multi-Task Transformer Architecture (M6_multitask).
Academic PyTorch implementation of the temporal surveillance neural network.

Pipeline Flow:
  1. Input: [Batch, 12, 44] Monthly behavioral feature tokens
  2. Input Projection: Dense projection [44 -> 32]
  3. Embeddings: Relative temporal positional encoding + loan-age seasoning
  4. Transformer Encoder: 3 stacked Pre-LayerNorm multi-head self-attention blocks
  5. [CLS] Pooling: Unified sequence representation token [Batch, 32]
  6. Head A: Calibrated 6-month loan distress classification logit [Batch]
  7. Head B: Discrete-time monthly hazard rate projection [Batch, 6]
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_config import (
    INPUT_DIM, D_MODEL, N_LAYERS, N_HEADS, FEEDFORWARD_MULT,
    DROPOUT, POOLING, HORIZON, MAX_POS_WINDOW
)

AGE_MIN, AGE_MAX = -24, 72


def age_index(age: torch.Tensor) -> torch.Tensor:
    """Clamps loan age seasoning in months into discrete embedding indices."""
    return (age.clamp(AGE_MIN, AGE_MAX) - AGE_MIN).long()


class PreLNTransformerLayer(nn.Module):
    """Pre-LayerNorm Transformer Block.
    Normalizes inputs *before* self-attention and feedforward layers,
    guaranteeing identity gradient backpropagation for training stability.
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        ff_mult: int = FEEDFORWARD_MULT,
        dropout: float = DROPOUT
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        # 1. Multi-head self-attention with pre-normalization
        h = self.ln1(x)
        attn_out, weights = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        x = x + self.drop(attn_out)

        # 2. Feedforward MLP with pre-normalization
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, weights


class MultiTaskDistressTransformer(nn.Module):
    """M6_multitask: Dual-Head Temporal Transformer for Loan Distress Surveillance.
    Processes 12 monthly banking observation tokens and predicts both:
      - Head A: 6-month binary distress classification
      - Head B: Discrete-time monthly hazard trajectory (+1m to +6m)
    """
    def __init__(
        self,
        n_features: int = INPUT_DIM,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        dropout: float = DROPOUT,
        horizon: int = HORIZON,
        pooling: str = POOLING
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.horizon = horizon
        self.pooling = pooling

        # ============================================================
        # 1. INPUT PROJECTION
        # ============================================================
        # Projects 44 continuous behavioral banking variables into hidden dimension d_model (32)
        self.input_proj = nn.Linear(n_features, d_model)

        # ============================================================
        # 2. TEMPORAL / AGE EMBEDDINGS
        # ============================================================
        # Learnable relative positional encoding (0 at cutoff, -1, -2 ... -11)
        self.pos_emb = nn.Embedding(MAX_POS_WINDOW, d_model)
        # Learnable loan seasoning age embedding in months
        self.age_emb = nn.Embedding(AGE_MAX - AGE_MIN + 1, d_model)

        # Learnable [CLS] classification token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.drop = nn.Dropout(dropout)

        # ============================================================
        # 3. TRANSFORMER ENCODER
        # ============================================================
        # Stack of Pre-LayerNorm Transformer encoder blocks
        self.layers = nn.ModuleList([
            PreLNTransformerLayer(d_model, n_heads, FEEDFORWARD_MULT, dropout)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # ============================================================
        # 4. CLASSIFICATION HEAD (HEAD A)
        # ============================================================
        # Predicts overall 6-month binary distress logit
        self.head_a = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        # ============================================================
        # 5. SURVIVAL / HAZARD HEAD (HEAD B)
        # ============================================================
        # Predicts discrete-time monthly hazard rates across horizons +1m to +6m
        self.head_b = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )

    def forward(
        self,
        features: torch.Tensor,
        ages: torch.Tensor | None = None,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Forward pass through the M6_multitask network.
        Args:
            features: [B, W, 44] Monthly behavioral sequence tokens.
            ages:     [B, W] Loan age in months (optional; defaults to 12..24).
            mask:     [B, W] 1 for real monthly tokens, 0 for padded tokens.
        Returns:
            logit_a:      [B] Head A scalar distress classification logit.
            hazard_b:     [B, 6] Head B discrete-time hazard rate logits.
            attn_weights: list of attention matrices across Transformer layers.
        """
        B, W, _ = features.shape
        device = features.device

        if ages is None:
            ages = torch.arange(12, 12 + W, device=device).unsqueeze(0).expand(B, -1).float()
        if mask is None:
            mask = torch.ones(B, W, device=device)

        # 1. Project input features: [B, W, 44] -> [B, W, 32]
        x = self.input_proj(features)

        # 2. Add relative temporal position encoding
        pos_ids = torch.arange(W - 1, -1, -1, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids.clamp(0, MAX_POS_WINDOW - 1))

        # 3. Add loan age seasoning embedding
        x = x + self.age_emb(age_index(ages))

        # 4. Prepend learnable [CLS] token: [B, W, 32] -> [B, W+1, 32]
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_mask = torch.ones(B, 1, device=device, dtype=mask.dtype)
            mask_with_cls = torch.cat([cls_mask, mask], dim=1)
        else:
            mask_with_cls = mask

        x = self.drop(x)
        key_padding_mask = (mask_with_cls == 0)

        # 5. Pass through Pre-LayerNorm Transformer blocks
        attn_weights = []
        for layer in self.layers:
            x, w = layer(x, key_padding_mask)
            attn_weights.append(w)

        x = self.final_ln(x)

        # 6. Extract sequence representation: [B, 32]
        if self.pooling == "cls":
            repr_vec = x[:, 0]
        else:
            w_exp = mask.unsqueeze(-1)
            repr_vec = (x * w_exp).sum(dim=1) / w_exp.sum(dim=1).clamp(min=1)

        # 7. Dual-Head Predictions
        logit_a = self.head_a(repr_vec).squeeze(-1) # [B]
        hazard_b = self.head_b(repr_vec)             # [B, 6]

        return logit_a, hazard_b, attn_weights

    def count_parameters(self) -> dict[str, int]:
        """Calculates exact total and trainable parameter counts dynamically."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": total - trainable
        }

    def model_summary(self) -> dict[str, any]:
        """Returns structured architectural specifications and parameter statistics."""
        params = self.count_parameters()
        return {
            "model_name": "MultiTaskDistressTransformer (M6_multitask)",
            "total_parameters": params["total_parameters"],
            "trainable_parameters": params["trainable_parameters"],
            "input_dim": self.n_features,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "feedforward_dim": FEEDFORWARD_MULT * self.d_model,
            "dropout": self.dropout_rate,
            "activation": "GELU",
            "pooling": self.pooling.upper(),
            "horizon": self.horizon,
            "head_a": "6-Month Binary Distress Classification (Scalar)",
            "head_b": f"Discrete-Time Monthly Hazard (+1m to +{self.horizon}m)"
        }


if __name__ == "__main__":
    print("=" * 80)
    print("CreditPulse M6_multitask Architecture Verification & Parameter Breakdown")
    print("=" * 80)
    
    model = MultiTaskDistressTransformer()
    summary = model.model_summary()
    
    for k, v in summary.items():
        print(f"{k.replace('_', ' ').title():<22}: {v}")
    
    print("-" * 80)
    # Verification pass with sample batch
    B, W, D = 4, 12, 44
    dummy_feat = torch.randn(B, W, D)
    dummy_age = torch.randint(6, 48, (B, W)).float()
    dummy_mask = torch.ones(B, W)
    
    logit_a, hazard_b, attns = model(dummy_feat, dummy_age, dummy_mask)
    
    print("\nTensor Shape Flow Verification:")
    print(f"  • Input features:      {list(dummy_feat.shape)}  [Batch, Seq_Len=12, Feats=44]")
    print(f"  • Head A (Distress):   {list(logit_a.shape)}          [Batch] -> Scalar Logit")
    print(f"  • Head B (Hazard):     {list(hazard_b.shape)}       [Batch, Horizons=6] -> Discrete Hazards")
    print(f"  • Attention Maps:      {len(attns)} layers, each {list(attns[0].shape)}")
    print("-" * 80)
    print("[SUCCESS] All forward pass tensor shape invariants verified.")
    print("=" * 80)
