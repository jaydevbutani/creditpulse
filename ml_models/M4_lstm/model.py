"""=============================================================================
CREDITPULSE BENCHMARK M4: 2-LAYER LONG SHORT-TERM MEMORY (LSTM)
=============================================================================
Layer Trace:
  Input: 12 monthly observations × 44 behavioral features [B, 12, 44]
  → 1. Sequence Encoder: 2-Layer LSTM with Cell Gates (input_dim=44, hidden_dim=32, num_layers=2)
  → 2. Dropout: rate=0.1
  → 3. Sequence Pooling: Final timestep hidden state h_12 [B, 32] (distinct from cell memory c_n)
  → 4. Prediction Head: Linear(32, 1) producing scalar distress logit
=============================================================================
"""
from __future__ import annotations
import torch
import torch.nn as nn
from ml_models.M4_lstm.config import M4LSTMConfig


class LSTMDistressClassifier(nn.Module):
    """2-Layer LSTM sequence model strictly configured via M4LSTMConfig."""
    def __init__(self, cfg: M4LSTMConfig | None = None):
        super().__init__()
        self.cfg = cfg or M4LSTMConfig()

        # 1. Sequence encoder: 2-layer LSTM maintaining separate cell memory c_t
        self.lstm = nn.LSTM(
            input_size=self.cfg.input_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            batch_first=True,
            dropout=self.cfg.dropout if self.cfg.num_layers > 1 else 0.0
        )

        # 2. Dropout
        self.dropout = nn.Dropout(self.cfg.dropout)

        # 3. Final prediction head
        self.classifier = nn.Linear(self.cfg.hidden_dim, self.cfg.output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Input: [Batch, 12, 44]
        # 1. Sequence encoding across 12 monthly steps
        lstm_out, (h_n, c_n) = self.lstm(x)  # [Batch, 12, hidden_dim]

        # 2. Pooling: terminal timestep hidden state h_12
        final_h = lstm_out[:, -1, :]  # [Batch, hidden_dim]

        # 3. Prediction head: linear projection to scalar logit
        logits = self.classifier(self.dropout(final_h)).squeeze(-1)  # [Batch]
        return logits
