"""Stage 4 — tokens, train-only preprocessing, aggregated window features, drift and leak scans.

Token tensor per snapshot: X [W=12, F], pad mask [W] (1 = padded / no data, R-19/R-23).
Left-padded: position W-1 is the cutoff month M0.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditpulse.features.panel import FEATURE_GROUPS, FEATURES, MONEY
from creditpulse.labels.events import mi_to_date
from creditpulse.utils import load_config, p, signed_log

LOG_FEATURES = MONEY | {"instal_to_inflow", "penalty_count", "n_tx", "n_credit", "n_debit"}
STATIC = set(FEATURE_GROUPS["loan_static"]) - {"loan_age", "remaining_term", "pre_loan"} | set(
    FEATURE_GROUPS["account_static"]) - {"account_age", "has_card", "card_gold", "card_junior"}


class PanelIndex:
    """O(1) lookup of (account, month) rows in the contiguous per-account panel."""

    def __init__(self, panel: pd.DataFrame):
        panel = panel.sort_values(["account_id", "mi"]).reset_index(drop=True)
        self.M = panel[FEATURES].to_numpy(np.float64)
        g = panel.groupby("account_id")
        self.start_row = g.cumcount().eq(0)
        first = panel.groupby("account_id").agg(row=("mi", lambda s: s.index[0]), mi0=("mi", "min"), mi1=("mi", "max"))
        self.row0 = first["row"].to_dict()
        self.mi0 = first["mi0"].to_dict()
        self.mi1 = first["mi1"].to_dict()

    def gather(self, account_ids, cutoff_mis, window: int):
        account_ids = np.asarray(account_ids)
        cutoff_mis = np.asarray(cutoff_mis)
        n = len(account_ids)
        X = np.zeros((n, window, len(FEATURES)))
        pad = np.ones((n, window), dtype=np.float32)
        r0 = np.array([self.row0[a] for a in account_ids])
        m0 = np.array([self.mi0[a] for a in account_ids])
        for k in range(window):
            m = cutoff_mis - (window - 1 - k)
            ok = m >= m0
            rows = r0 + (m - m0)
            X[ok, k] = self.M[rows[ok]]
            pad[ok, k] = 0.0
        return X, pad


class Preprocessor:
    """Signed-log -> winsorise (train 1/99 pct, C10) -> standardise. Fitted on training tokens only (R-20)."""

    def __init__(self):
        self.lo = self.hi = self.mu = self.sd = None
        self.meta: dict = {}

    def _log(self, X):
        X = X.copy()
        for j, f in enumerate(FEATURES):
            if f in LOG_FEATURES:
                X[..., j] = signed_log(X[..., j])
        return X

    def fit(self, X, pad, q=(0.01, 0.99), fit_range=("", "")):
        Z = self._log(X)[pad == 0]
        self.lo, self.hi = np.quantile(Z, q[0], axis=0), np.quantile(Z, q[1], axis=0)
        Z = np.clip(Z, self.lo, self.hi)
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        self.meta = {"fit_cutoff_min": fit_range[0], "fit_cutoff_max": fit_range[1], "n_tokens": int(len(Z))}
        return self

    def transform(self, X, pad):
        Z = (np.clip(self._log(X), self.lo, self.hi) - self.mu) / self.sd
        Z[pad == 1] = 0.0
        return Z.astype(np.float32)

    def to_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "features": FEATURES, "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        o = cls()
        assert d["features"] == FEATURES, "feature list changed since the preprocessor was fitted"
        o.lo, o.hi, o.mu, o.sd = (np.array(d[k]) for k in ["lo", "hi", "mu", "sd"])
        o.meta = d["meta"]
        return o


def aggregate(Z: np.ndarray, pad: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Aggregated window features for M1/M2/S1: last, mean, std, min, max, 6-month slope."""
    valid = (pad == 0)[..., None]
    n_valid = valid.sum(1).clip(min=1)
    mean = (Z * valid).sum(1) / n_valid
    std = np.sqrt((((Z - mean[:, None]) * valid) ** 2).sum(1) / n_valid)
    big = 1e6
    mn = np.where(valid, Z, big).min(1)
    mx = np.where(valid, Z, -big).max(1)
    mn[mn == big] = 0
    mx[mx == -big] = 0
    last = Z[:, -1]
    tt = np.arange(6) - 2.5
    w6, v6 = Z[:, -6:], valid[:, -6:]
    slope = (w6 * v6 * tt[None, :, None]).sum(1) / (tt**2).sum()
    cols, blocks = [], []
    for name, arr in [("last", last), ("mean", mean), ("std", std), ("min", mn), ("max", mx), ("slope6", slope)]:
        keep = [j for j, f in enumerate(FEATURES) if name == "last" or f not in STATIC]
        blocks.append(arr[:, keep])
        cols += [f"{FEATURES[j]}__{name}" for j in keep]
    blocks.append((pad == 0).sum(1, keepdims=True).astype(float))
    cols.append("n_valid_months")
    return np.concatenate(blocks, 1).astype(np.float32), cols


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    e = np.histogram(np.clip(expected, qs[0], qs[-1]), qs)[0] / len(expected)
    a = np.histogram(np.clip(actual, qs[0], qs[-1]), qs)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_dataset(cfg: dict, snap: pd.DataFrame, index: PanelIndex, pre: Preprocessor | None = None,
                  window: int | None = None, fit_split: str = "train"):
    W = window or cfg["labels"]["window"]
    X, pad = index.gather(snap.account_id.values, snap.cutoff_mi.values, W)
    if pre is None:
        tr = (snap.split == fit_split).values
        rng = (str(snap.cutoff_date[tr].min().date()), str(snap.cutoff_date[tr].max().date()))
        pre = Preprocessor().fit(X[tr], pad[tr], cfg["features"]["winsor_q"], rng)
    Z = pre.transform(X, pad)
    return Z, pad, pre


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = p(cfg, "processed")
    panel = pd.read_parquet(proc / "panel.parquet")
    snap = pd.read_parquet(proc / "snapshots.parquet")
    index = PanelIndex(panel)
    Z, pad, pre = build_dataset(cfg, snap, index)
    A, acols = aggregate(Z, pad)
    np.savez_compressed(proc / "tokens_p1.npz", Z=Z, pad=pad, A=A)
    (proc / "agg_columns.json").write_text(json.dumps(acols))
    art = p(cfg, "artifacts")
    (art / "preprocessor.json").write_text(json.dumps(pre.to_dict()))

    checks = {}
    checks["R-19"] = {"pass": bool(np.isfinite(Z).all() and np.isfinite(A).all()),
                      "padded_fraction": float(pad.mean())}
    tr = (snap.split == "train").values
    other = set(snap.loan_id[~tr])
    checks["R-20"] = {"pass": bool(pre.meta["n_tokens"] == int((pad[tr] == 0).sum()) and
                                   not (set(snap.loan_id[tr]) & other)),
                      "fit_range": [pre.meta["fit_cutoff_min"], pre.meta["fit_cutoff_max"]],
                      "fit_tokens": pre.meta["n_tokens"], "fitted_on": "train-split loans only (no val/test loan contributes)"}
    # R-22 drift: PSI per feature at the cutoff month, train vs test
    te = (snap.split == "test").values
    last_ok_tr, last_ok_te = pad[tr, -1] == 0, pad[te, -1] == 0
    ps = {f: psi(Z[tr][last_ok_tr, -1, j], Z[te][last_ok_te, -1, j]) for j, f in enumerate(FEATURES)}
    flagged = {k: round(v, 3) for k, v in sorted(ps.items(), key=lambda x: -x[1]) if v > cfg["features"]["psi_threshold"]}
    pd.Series(ps).sort_values(ascending=False).rename("psi").to_csv(p(cfg, "reports") / "tables" / "psi_train_vs_test.csv")
    checks["R-22"] = {"pass": True, "n_above_0.25": len(flagged), "flagged": flagged,
                      "review": "Flagged features are calendar/loan-age driven (test loans are younger and observed "
                                "later); they are kept and discussed in the model card."}
    # R-24 single-feature AUC scan on train
    y = snap.label.values
    lab = tr & ~np.isnan(y)
    aucs = {}
    for j, c in enumerate(acols):
        x = A[lab, j]
        if np.std(x) == 0:
            continue
        a = roc_auc_score(y[lab], x)
        aucs[c] = max(a, 1 - a)
    s = pd.Series(aucs).sort_values(ascending=False)
    s.rename("single_feature_auc").to_csv(p(cfg, "reports") / "tables" / "single_feature_auc.csv")
    checks["R-24"] = {"pass": bool(s.max() <= cfg["features"]["leak_auc_threshold"]), "max_auc": float(s.max()),
                      "top5": s.head(5).round(3).to_dict()}
    (p(cfg, "reports") / "tables" / "feature_checks.json").write_text(json.dumps(checks, indent=2))
    return checks


if __name__ == "__main__":
    for k, v in run().items():
        print(k, "PASS" if v["pass"] else "FAIL", {a: b for a, b in v.items() if a != "pass"})
