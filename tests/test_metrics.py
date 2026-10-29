"""R-39 metric correctness, R-40 clustered bootstrap, R-42 capture and lead-time toy examples."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from creditpulse.eval.metrics import (
    brier, c_index, capture_table, ci, cluster_bootstrap, cumulative_incidence, ece, event_capture,
    lead_time_stats, naive_bootstrap, pr_auc, roc_auc,
)


def test_metrics_match_sklearn_and_hand():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    assert np.isclose(pr_auc(y, s), average_precision_score(y, s))
    assert np.isclose(roc_auc(y, s), roc_auc_score(y, s))
    assert np.isclose(brier([1, 0], [0.8, 0.4]), ((0.2 ** 2) + (0.4 ** 2)) / 2)
    assert np.isclose(ece([0, 0, 1, 1, 1], [0.0, 0.0, 1.0, 1.0, 1.0], 5), 0.0)


def test_c_index_matches_sksurv():
    from sksurv.metrics import concordance_index_censored

    rng = np.random.default_rng(1)
    t = rng.integers(1, 7, 80)
    e = rng.random(80) < 0.4
    risk = -t + rng.normal(0, 2, 80)
    ek = np.where(e, t, 0)
    ours = c_index(ek, t, risk)
    ref = concordance_index_censored(e, t.astype(float), risk)[0]
    assert abs(ours - ref) < 1e-9


def test_cumulative_incidence():
    h = np.array([[0.1, 0.2, 0.0, 0.0, 0.0, 0.5]])
    F = cumulative_incidence(h)
    assert np.isclose(F[0, 1], 1 - 0.9 * 0.8) and np.isclose(F[0, -1], 1 - 0.9 * 0.8 * 0.5)


def test_capture_and_lead_time_toy():
    """Loan flagged at month 2, event at month 6 -> lead time 4 (R-42)."""
    rows = []
    for c in range(0, 6):
        rows.append({"loan_id": 1, "cutoff_mi": c, "score": 0.9 if c >= 2 else 0.01, "event_mi": 6.0})
        rows.append({"loan_id": 2, "cutoff_mi": c, "score": 0.5, "event_mi": np.nan})
        for j in range(3, 11):
            rows.append({"loan_id": j, "cutoff_mi": c, "score": 0.02, "event_mi": np.nan})
    df = pd.DataFrame(rows)
    t = capture_table(df, threshold=0.8)
    assert t.lead.iloc[0] == 4
    st = lead_time_stats(df, 0.8)
    assert st["median_lead"] == 4 and st["share_lead_ge3"] == 1.0
    assert event_capture(df, 10) == 1.0        # loan 1 is top-1 of 10 at months 2..5
    df.loc[df.loan_id == 1, "score"] = 0.01
    assert event_capture(df, 10) == 0.0


def test_clustered_bootstrap_wider_than_naive():
    """R-40: snapshots of one loan are correlated; resampling them individually understates uncertainty."""
    rng = np.random.default_rng(0)
    rows = []
    for lid in range(60):
        risky = rng.random() < 0.25
        base = rng.random() * 0.6 + (0.3 if risky else 0)   # loan-level score, outcome shared by all snapshots
        for c in range(15):
            rows.append({"loan_id": lid, "y": int(risky), "score": base + rng.normal(0, 0.01)})
    df = pd.DataFrame(rows)
    f = lambda d: pr_auc(d.y.values, d.score.values)  # noqa: E731
    cw = np.diff(ci(cluster_bootstrap(df, f, 400, 0)))[0]
    nw = np.diff(ci(naive_bootstrap(df, f, 400, 0)))[0]
    assert cw > nw


def test_survival_baselines_toy_curve():
    """R-30: on a toy dataset where a high x means earlier events, both survival baselines give a higher
    hazard for high x and a C-index above 0.5 on training data."""
    from creditpulse.models.baselines import Cox, DiscreteHazard

    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 2))
    h = 1 / (1 + np.exp(-(-2.5 + 1.2 * x[:, 0])))
    t = np.minimum(rng.geometric(h), 7)
    ek = np.where(t <= 6, t, 0)
    fk = np.where(t <= 6, t, 6)
    for m in (DiscreteHazard(C=1.0).fit(x, ek, fk), Cox(alpha=0.01).fit(x, ek, fk)):
        hz = m.hazard(np.array([[2.0, 0.0], [-2.0, 0.0]]))
        assert (hz[0] > hz[1]).all()
        F6 = cumulative_incidence(m.hazard(x))[:, -1]
        assert c_index(ek, fk, F6) > 0.6
