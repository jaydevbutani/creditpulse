"""Deep sequence models (report §8.1–8.3): GRU, LSTM, pre-LN Transformer, multi-task heads, pretraining."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_POS = 24          # positions counted back from the cutoff month (supports windows up to 24)
AGE_MIN, AGE_MAX = -24, 72


def age_index(age: torch.Tensor) -> torch.Tensor:
    return (age.clamp(AGE_MIN, AGE_MAX) - AGE_MIN).long()


class PreLNLayer(nn.Module):
    """Pre-layer-norm Transformer block that also returns attention weights (for explainability)."""

    def __init__(self, d, heads, ff_mult, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad):
        h = self.ln1(x)
        a, w = self.attn(h, h, h, key_padding_mask=key_pad, need_weights=True, average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, w


class SeqModel(nn.Module):
    def __init__(self, n_feat: int, kind: str = "transformer", d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.2, pooling: str = "cls", head_b: bool = False,
                 derive_a: bool = False, horizon: int = 6):
        super().__init__()
        self.kind, self.pooling, self.head_b, self.derive_a, self.H = kind, pooling, head_b, derive_a, horizon
        self.inp = nn.Sequential(nn.Linear(n_feat + 1, d_model), nn.Dropout(dropout))  # +1: is-masked flag
        self.mask_emb = nn.Parameter(torch.zeros(d_model))
        if kind == "transformer":
            self.pos = nn.Embedding(MAX_POS, d_model)
            self.age = nn.Embedding(AGE_MAX - AGE_MIN + 1, d_model)
            self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
            self.layers = nn.ModuleList([PreLNLayer(d_model, n_heads, 2, dropout) for _ in range(n_layers)])
            self.ln = nn.LayerNorm(d_model)
        elif kind in ("gru", "lstm"):
            rnn = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = rnn(d_model, d_model, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        else:
            raise ValueError(kind)
        self.head_a = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
        if head_b:
            self.head_hz = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, horizon))
        self.recon = nn.Linear(d_model, n_feat)  # used only by self-supervised pretraining
        self.last_attn = None

    # ------------------------------------------------------------------ encoders
    def embed(self, x, pad, age, mlm_mask=None):
        flag = torch.zeros_like(pad) if mlm_mask is None else mlm_mask
        if mlm_mask is not None:
            x = x.masked_fill(mlm_mask.bool().unsqueeze(-1), 0.0)
        h = self.inp(torch.cat([x, flag.unsqueeze(-1)], -1))
        if mlm_mask is not None:
            h = h + mlm_mask.unsqueeze(-1) * self.mask_emb
        return h

    def encode_tokens(self, x, pad, age, mlm_mask=None):
        """Per-position representations [B, W, d] (transformer only)."""
        B, W, _ = x.shape
        h = self.embed(x, pad, age, mlm_mask)
        back = torch.arange(W - 1, -1, -1, device=x.device)          # distance from the cutoff month
        h = h + self.pos(back.clamp(max=MAX_POS - 1)).unsqueeze(0) + self.age(age_index(age))
        key_pad = pad.bool()
        cls = self.cls.expand(B, 1, -1)
        h = torch.cat([cls, h], 1)
        kp = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), key_pad], 1)
        attn = []
        for layer in self.layers:
            h, w = layer(h, kp)
            attn.append(w)
        self.last_attn = attn
        return self.ln(h), kp

    def encode(self, x, pad, age):
        if self.kind == "transformer":
            h, kp = self.encode_tokens(x, pad, age)
            if self.pooling == "cls":
                return h[:, 0]
            valid = (~kp[:, 1:]).float().unsqueeze(-1)
            return (h[:, 1:] * valid).sum(1) / valid.sum(1).clamp(min=1)
        # RNN: shift left-padded sequences to start at t=0 and pack, so padding never enters the state
        h = self.embed(x, pad, age)
        lengths = (pad == 0).sum(1)
        W = x.shape[1]
        idx = (torch.arange(W, device=x.device).unsqueeze(0) + (W - lengths).unsqueeze(1)) % W
        h = torch.gather(h, 1, idx.unsqueeze(-1).expand_as(h))
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.rnn(packed)
        hn = state[0] if self.kind == "lstm" else state
        return hn[-1]

    def forward(self, x, pad, age):
        z = self.encode(x, pad, age)
        out = {}
        if self.head_b:
            out["hz_logits"] = self.head_hz(z)
        if self.derive_a and self.head_b:
            log_surv = F.logsigmoid(-out["hz_logits"]).sum(1)          # log prod(1 - h)
            p = (1 - torch.exp(log_surv)).clamp(1e-6, 1 - 1e-6)
            out["logit_a"] = torch.log(p) - torch.log1p(-p)
        else:
            out["logit_a"] = self.head_a(z).squeeze(-1)
        return out

    def pretrain_loss(self, x, pad, age, mask_frac=0.15):
        valid = pad == 0
        mlm = (torch.rand_like(pad) < mask_frac) & valid
        mlm[mlm.sum(1) == 0, -1] = True  # at least one masked token per sequence
        mlm = mlm.float()
        h, _ = self.encode_tokens(x, pad, age, mlm_mask=mlm)
        rec = self.recon(h[:, 1:])
        err = ((rec - x) ** 2).mean(-1)
        return (err * mlm).sum() / mlm.sum()


# ---------------------------------------------------------------------- losses
def hazard_nll(hz_logits: torch.Tensor, event_k: torch.Tensor, follow_k: torch.Tensor) -> torch.Tensor:
    """Discrete-time NLL (report §8.3). Event in interval j: -[sum_{u<j} log(1-h_u) + log h_j].
    Censored after interval j: -sum_{u<=j} log(1-h_u). Returns per-sample loss."""
    H = hz_logits.shape[1]
    u = torch.arange(1, H + 1, device=hz_logits.device).unsqueeze(0)
    log_h = F.logsigmoid(hz_logits)
    log_1mh = F.logsigmoid(-hz_logits)
    ev = event_k.unsqueeze(1)
    fo = follow_k.unsqueeze(1)
    is_ev = (event_k > 0).unsqueeze(1)
    surv_terms = torch.where(is_ev, (u < ev).float(), (u <= fo).float())
    loss = -(surv_terms * log_1mh).sum(1) - (is_ev & (u == ev)).float().mul(log_h).sum(1)
    return loss


def focal_bce(logits, y, pos_weight, gamma=2.0):
    p = torch.sigmoid(logits)
    pt = torch.where(y == 1, p, 1 - p)
    w = torch.where(y == 1, pos_weight, torch.ones_like(y))
    return (w * (1 - pt) ** gamma * F.binary_cross_entropy_with_logits(logits, y, reduction="none")).mean()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def cosine_warmup(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * prog))
