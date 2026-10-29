"""R-31 overfit one batch, R-32 shapes/params, R-33 mask invariance, R-37 hazard loss, R-38 finite grads."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from creditpulse.models.deep import SeqModel, count_params, hazard_nll

torch.set_num_threads(1)
NF = 20


def batch(n=32, W=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, W, NF, generator=g)
    pad = torch.zeros(n, W)
    pad[: n // 2, :4] = 1  # half the sequences are left-padded
    x[pad.bool()] = 0
    age = torch.arange(W).float().expand(n, W) - 3
    y = (torch.rand(n, generator=g) < 0.3).float()
    return x, pad, age, y


@pytest.mark.parametrize("kind", ["gru", "lstm", "transformer"])
def test_shapes_and_params(kind):
    m = SeqModel(NF, kind=kind, d_model=32, n_layers=2, head_b=True)
    x, pad, age, _ = batch()
    o = m(x, pad, age)
    assert o["logit_a"].shape == (32,)
    assert o["hz_logits"].shape == (32, 6)
    assert 5_000 < count_params(m) < 300_000


@pytest.mark.parametrize("kind", ["gru", "lstm", "transformer"])
def test_overfit_one_batch(kind):
    torch.manual_seed(0)
    m = SeqModel(NF, kind=kind, d_model=32, n_layers=1, dropout=0.0)
    x, pad, age, y = batch()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(m(x, pad, age)["logit_a"], y)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        assert torch.isfinite(gn)
        opt.step()
    assert loss.item() < 0.05


@pytest.mark.parametrize("kind", ["gru", "lstm", "transformer"])
def test_mask_invariance(kind):
    """Adding extra padded months on the left must not change the output (R-33)."""
    torch.manual_seed(0)
    m = SeqModel(NF, kind=kind, d_model=32, n_layers=2, head_b=True).eval()
    x, pad, age, _ = batch()
    extra = 5
    x2 = torch.cat([torch.randn(32, extra, NF), x], 1)  # garbage in padded slots
    pad2 = torch.cat([torch.ones(32, extra), pad], 1)
    age2 = torch.cat([age[:, :1] - torch.arange(extra, 0, -1).float(), age], 1)
    with torch.no_grad():
        a, b = m(x, pad, age), m(x2, pad2, age2)
    assert torch.allclose(a["logit_a"], b["logit_a"], atol=1e-5)
    assert torch.allclose(a["hz_logits"], b["hz_logits"], atol=1e-5)


def test_hazard_loss_hand_computed():
    """R-37: event, censored and early-event toy cases equal the hand formula."""
    z = torch.tensor([[0.0, 1.0, -1.0, 0.5, 0.0, 0.0]] * 3)
    h = torch.sigmoid(z[0]).tolist()
    ev = torch.tensor([3, 0, 1])
    fo = torch.tensor([3, 4, 1])
    got = hazard_nll(z, ev, fo).tolist()
    exp_event3 = -(math.log(1 - h[0]) + math.log(1 - h[1]) + math.log(h[2]))
    exp_cens4 = -sum(math.log(1 - h[u]) for u in range(4))
    exp_event1 = -math.log(h[0])
    assert np.allclose(got, [exp_event3, exp_cens4, exp_event1], atol=1e-6)


def test_hazard_censored_at_zero_contributes_nothing():
    z = torch.randn(2, 6)
    assert torch.allclose(hazard_nll(z, torch.tensor([0, 0]), torch.tensor([0, 0])), torch.zeros(2))
