"""Metric suite fixed before any model is trained (report §9.1) + loan-clustered bootstrap (§9.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# --------------------------------------------------------------- probability metrics
def pr_auc(y, s):
    return float(average_precision_score(y, s)) if 0 < np.sum(y) < len(y) else np.nan


def roc_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else np.nan


def brier(y, s):
    return float(brier_score_loss(y, np.clip(s, 0, 1)))


def brier_skill(y, s, base_rate):
    ref = brier(y, np.full(len(y), base_rate))
    return float(1 - brier(y, s) / ref) if ref > 0 else np.nan


def reliability(y, s, n_bins=5):
    """Equal-mass bins (few positives). Returns DataFrame with mean predicted / observed per bin."""
    order = np.argsort(s, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = [(float(np.mean(s[b])), float(np.mean(y[b])), len(b)) for b in bins if len(b)]
    return pd.DataFrame(rows, columns=["mean_pred", "obs_rate", "n"])


def ece(y, s, n_bins=5):
    r = reliability(np.asarray(y), np.asarray(s), n_bins)
    return float(np.sum(r.n * (r.mean_pred - r.obs_rate).abs()) / r.n.sum())


# --------------------------------------------------------------- monitoring metrics
def _flags(df: pd.DataFrame, k_pct: float | None = None, threshold: float | None = None) -> pd.Series:
    """Flag snapshots: top-K% of each cutoff month's at-risk queue, or score >= threshold."""
    if threshold is not None:
        return df["score"] >= threshold
    rank = df.groupby("cutoff_mi")["score"].rank(ascending=False, method="first")
    n = df.groupby("cutoff_mi")["score"].transform("size")
    return rank <= np.ceil(n * k_pct / 100.0)


def capture_table(df: pd.DataFrame, k_pct=None, threshold=None, horizon=6) -> pd.DataFrame:
    """Per event loan: captured (flagged at least once in the H months before the event) and lead time.

    df columns: loan_id, cutoff_mi, score, event_mi (NaN if no event). All at-risk snapshots, including
    right-censored ones, take part in the monthly ranking (that is what an analyst's queue sees).
    """
    d = df.copy()
    d["flag"] = _flags(d, k_pct, threshold).values
    ev = d[d.event_mi.notna() & (d.cutoff_mi >= d.event_mi - horizon) & (d.cutoff_mi < d.event_mi)]
    rows = []
    for lid, g in ev.groupby("loan_id"):
        fl = g[g.flag]
        first = fl.cutoff_mi.min() if len(fl) else np.nan
        rows.append((lid, int(len(fl) > 0), g.event_mi.iloc[0] - first if len(fl) else np.nan))
    return pd.DataFrame(rows, columns=["loan_id", "captured", "lead"])


def event_capture(df, k_pct, horizon=6) -> float:
    t = capture_table(df, k_pct=k_pct, horizon=horizon)
    return float(t.captured.mean()) if len(t) else np.nan


def lead_time_stats(df, threshold, horizon=6) -> dict:
    t = capture_table(df, threshold=threshold, horizon=horizon)
    lead = t.lead.dropna()
    return {
        "event_loans": int(len(t)),
        "flagged_share": float(t.captured.mean()) if len(t) else np.nan,
        "median_lead": float(lead.median()) if len(lead) else np.nan,
        "share_lead_ge3": float((t.lead >= 3).mean()) if len(t) else np.nan,
    }


def false_alarm_load(df, threshold) -> float:
    """Mean flags per month on loans that never have an event."""
    d = df[df.event_mi.isna()]
    return float((d.score >= threshold).groupby(d.cutoff_mi).sum().mean()) if len(d) else np.nan


# --------------------------------------------------------------- survival metrics (Head B)
def cumulative_incidence(h: np.ndarray) -> np.ndarray:
    """F(k) = 1 - prod_{u<=k} (1 - h_u), for each k."""
    return 1 - np.cumprod(1 - h, axis=1)


def c_index(event_k, follow_k, risk) -> float:
    """Harrell's concordance for discrete times with right censoring."""
    t = np.where(event_k > 0, event_k, follow_k).astype(float)
    e = (event_k > 0).astype(int)
    num = den = 0.0
    idx = np.flatnonzero(e)
    for i in idx:
        comp = (t > t[i]) | ((t == t[i]) & (e == 0))  # censored at the same month was still at risk
        den += comp.sum()
        num += (risk[i] > risk[comp]).sum() + 0.5 * (risk[i] == risk[comp]).sum()
    return float(num / den) if den else np.nan


def integrated_brier(event_k, follow_k, h: np.ndarray, horizon=6) -> float:
    """IPCW integrated Brier score over months 1..H using Kaplan–Meier censoring weights."""
    t = np.where(event_k > 0, event_k, follow_k).astype(float)
    e = (event_k > 0).astype(int)
    surv = np.cumprod(1 - h, axis=1)
    # censoring KM G(t)
    times = np.arange(0, horizon + 1)
    G = np.ones(horizon + 1)
    g = 1.0
    for k in range(1, horizon + 1):
        at_risk = (t >= k).sum()
        cens = ((t == k) & (e == 0)).sum()
        g *= 1 - cens / at_risk if at_risk else 1
        G[k] = max(g, 1e-3)
    scores = []
    for k in range(1, horizon + 1):
        s_k = surv[:, k - 1]
        died = (t <= k) & (e == 1)
        alive = t > k
        w_d = 1 / G[np.clip(t.astype(int) - 1, 0, horizon)]
        bs = np.where(died, (0 - s_k) ** 2 * w_d, 0) + np.where(alive, (1 - s_k) ** 2 / G[k], 0)
        scores.append(bs.mean())
    return float(np.mean(scores)) if times.size else np.nan


# --------------------------------------------------------------- calibration
class Platt:
    """Platt scaling on the logit of the score (fitted on validation)."""

    def fit(self, s, y):
        """Calibration must be monotone increasing. If the fitted slope is not positive (a validation set with
        almost no signal), fall back to a pure intercept shift with slope 1 and flag it."""
        from scipy.optimize import minimize_scalar

        z = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
        lr = LogisticRegression(C=1e4).fit(z[:, None], y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        self.degenerate = self.a <= 0
        if self.degenerate:
            y = np.asarray(y, float)

            def nll(b):
                q = np.clip(1 / (1 + np.exp(-(z + b))), 1e-9, 1 - 1e-9)
                return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

            self.a, self.b = 1.0, float(minimize_scalar(nll, bounds=(-20, 20), method="bounded").x)
        return self

    def __call__(self, s):
        return apply_platt(s, self.a, self.b)

    def params(self):
        return {"a": self.a, "b": self.b, "slope_fallback": bool(self.degenerate)}


def apply_platt(s, a, b):
    z = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(a * z + b)))


# --------------------------------------------------------------- bootstrap
def cluster_bootstrap(df: pd.DataFrame, fn, n_boot=2000, seed=0, cluster="loan_id") -> np.ndarray:
    """Resample loans (clusters) with replacement; all snapshots of a drawn loan come along."""
    rng = np.random.default_rng(seed)
    groups = df.groupby(cluster).indices
    keys = list(groups)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        v = fn(df.iloc[idx])
        if v is not None and not np.isnan(v):
            vals.append(v)
    return np.array(vals)


def naive_bootstrap(df, fn, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(df)
    vals = [fn(df.iloc[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return np.array([v for v in vals if not np.isnan(v)])


def ci(vals, alpha=0.05):
    if len(vals) == 0:
        return (np.nan, np.nan)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_diff(df, col_a, col_b, metric=pr_auc, n_boot=2000, seed=0):
    """Paired loan-clustered bootstrap of metric(col_a) - metric(col_b)."""
    fn = lambda d: metric(d.y.values, d[col_a].values) - metric(d.y.values, d[col_b].values)  # noqa: E731
    vals = cluster_bootstrap(df, fn, n_boot, seed)
    point = fn(df)
    lo, hi = ci(vals)
    return {"diff": float(point), "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}


def headline(y, s, base_rate, n_bins=5) -> dict:
    return {
        "pr_auc": pr_auc(y, s), "roc_auc": roc_auc(y, s), "brier": brier(y, s),
        "brier_skill": brier_skill(y, s, base_rate), "ece": ece(y, s, n_bins),
        "prevalence": float(np.mean(y)), "n": int(len(y)), "positives": int(np.sum(y)),
    }
