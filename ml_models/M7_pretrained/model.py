"""=============================================================================
CREDITPULSE BENCHMARK M7: SELF-SUPERVISED PRE-TRAINED SEQUENCE MODEL
=============================================================================
Architecture Specification for Viva / Examiner Review:
  1. Input: 12 months × 44 behavioral features [B, 12, 44]
  2. Sequential Backbone: 2-Layer Bidirectional GRU (hidden_size = cfg.hidden_dim = 48)
     - Bidirectional concatenated output dimension: 48 * 2 = 96
     - LayerNorm(96)
  3. Pre-training Head (Phase 1):
     - Linear(96, 48) -> ReLU -> Linear(48, 44)
     - Objective: Denoising reconstruction (MSE) of 15% randomly masked tokens
  4. Fine-Tuning Head (Phase 2):
     - Sequence Mean Pooling -> Linear(96, 32) -> ReLU -> Dropout(0.1) -> Linear(32, 1)
     - Entire backbone is fine-tuned end-to-end on distress labels via AdamW
=============================================================================
"""
from __future__ import annotations
import torch
import torch.nn as nn
from ml_models.M7_pretrained.config import M7PretrainedConfig


class PretrainedSequenceBackbone(nn.Module):
    """Bidirectional sequential encoder strictly configured via M7PretrainedConfig."""
    def __init__(self, cfg: M7PretrainedConfig | None = None):
        super().__init__()
        self.cfg = cfg or M7PretrainedConfig()

        # Step 1: Bidirectional GRU Backbone
        self.backbone = nn.GRU(
            input_size=self.cfg.input_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=self.cfg.dropout
        )
        self.norm = nn.LayerNorm(self.cfg.hidden_dim * 2)

        # Step 2: Phase 1 Self-supervised Reconstruction Head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim * 2, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.input_dim)
        )

        # Step 3: Phase 2 Downstream Fine-Tuning Classification Head
        self.classifier_head = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(32, self.cfg.output_dim)
        )

    def forward_pretrain(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs reconstructed feature sequence [B, 12, 44]."""
        out, _ = self.backbone(x)
        out = self.norm(out)
        recon = self.reconstruction_head(out)
        return recon

    def forward_finetune(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs distress classification logits [B]."""
        out, _ = self.backbone(x)
        out = self.norm(out)
        pooled = out.mean(dim=1)
        logits = self.classifier_head(pooled).squeeze(-1)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_finetune(x)
