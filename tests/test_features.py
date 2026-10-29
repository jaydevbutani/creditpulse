"""R-18 causality (future perturbation, T1) incl. an injected-leak check (R-03), R-19, R-20, R-23."""
import json

import numpy as np
import pandas as pd
import pytest

from creditpulse.features.panel import FEATURES, build_panel
from creditpulse.features.tokens import PanelIndex, Preprocessor
from creditpulse.labels.events import month_index
from creditpulse.utils import ROOT, load_config

CFG = load_config()
INTERIM = ROOT / "data" / "interim"
PROC = ROOT / "data" / "processed"
needs_data = pytest.mark.skipif(not (INTERIM / "trans.parquet").exists(), reason="run `make data` first")


def _tables(account_ids):
    t = {k: pd.read_parquet(INTERIM / f"{k}.parquet") for k in ["trans", "account", "loan", "district", "card", "disp"]}
    t["trans"] = t["trans"][t["trans"].account_id.isin(account_ids)]
    t["account"] = t["account"][t["account"].account_id.isin(account_ids)]
    t["loan"] = t["loan"][t["loan"].account_id.isin(account_ids)]
    return t


def _panel(t, trans=None):
    return build_panel(trans if trans is not None else t["trans"], t["account"], t["loan"], t["district"], t["card"],
                       t["disp"], CFG)


def assert_causal(panel_fn, t, cutoff_mi):
    """Tokens for months <= cutoff must be bit-identical after shuffling, scaling and deleting the future."""
    base = panel_fn(t)
    fut = t["trans"].date.dt.year * 12 + t["trans"].date.dt.month - 1 > cutoff_mi
    variants = []
    tr = t["trans"].copy()
    tr.loc[fut, ["amount", "balance", "signed_amount"]] *= 10
    variants.append(tr)
    tr = t["trans"].copy()
    tr.loc[fut, "amount"] = tr.loc[fut, "amount"].sample(frac=1, random_state=0).values
    variants.append(tr)
    variants.append(t["trans"][~fut])
    for v in variants:
        alt = panel_fn({**t, "trans": v})
        a = base[base.mi <= cutoff_mi][FEATURES].to_numpy()
        b = alt[alt.mi <= cutoff_mi][FEATURES].to_numpy()
        if a.shape != b.shape or not np.array_equal(a, b):
            raise AssertionError("future data changed past tokens")


@needs_data
def test_future_perturbation_does_not_change_past_tokens():
    loan = pd.read_parquet(INTERIM / "loan.parquet")
    accs = loan.account_id.iloc[[0, 5, 50]].tolist()
    t = _tables(accs)
    assert_causal(lambda tt: _panel(tt), t, month_index(pd.Timestamp("1996-06-30")))


@needs_data
def test_injected_leak_is_detected():
    """A deliberately leaky builder (uses next month's inflow) must fail the causality check (R-03)."""
    loan = pd.read_parquet(INTERIM / "loan.parquet")
    t = _tables(loan.account_id.iloc[[0]].tolist())

    def leaky(tt):
        p = _panel(tt)
        p["inflow"] = p.groupby("account_id")["inflow"].shift(-1).fillna(0)
        return p

    with pytest.raises(AssertionError):
        assert_causal(leaky, t, month_index(pd.Timestamp("1996-06-30")))


def test_gather_left_padding_and_shapes():
    rows = []
    for m in range(5):
        r = {f: float(m + 1) for f in FEATURES}
        r.update({"account_id": 7, "mi": 100 + m})
        rows.append(r)
    idx = PanelIndex(pd.DataFrame(rows))
    X, pad = idx.gather([7, 7], [104, 102], 12)
    assert X.shape == (2, 12, len(FEATURES)) and pad.shape == (2, 12)
    assert pad[0].tolist() == [1] * 7 + [0] * 5          # 5 months of history, left-padded
    assert pad[1].tolist() == [1] * 9 + [0] * 3
    assert np.all(X[0, :7] == 0) and X[0, -1, 0] == 5.0 and X[1, -1, 0] == 3.0


def test_preprocessor_zeroes_padding_and_is_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1000, (50, 12, len(FEATURES)))
    pad = np.zeros((50, 12), dtype=np.float32)
    pad[:10, :3] = 1
    pre = Preprocessor().fit(X, pad)
    Z = pre.transform(X, pad)
    assert np.isfinite(Z).all() and np.all(Z[pad == 1] == 0)
    rt = Preprocessor.from_dict(json.loads(json.dumps(pre.to_dict())))
    assert np.array_equal(rt.transform(X, pad), Z)


@pytest.mark.skipif(not (PROC / "tokens_p1.npz").exists(), reason="run `make features` first")
def test_token_hygiene_and_train_only_fit():
    d = np.load(PROC / "tokens_p1.npz")
    assert np.isfinite(d["Z"]).all() and np.isfinite(d["A"]).all()            # R-19
    assert set(np.unique(d["pad"])) <= {0.0, 1.0}
    assert np.all(d["Z"][d["pad"] == 1] == 0)
    chk = json.loads((ROOT / "reports" / "tables" / "feature_checks.json").read_text())
    assert chk["R-20"]["pass"] and chk["R-24"]["pass"]
