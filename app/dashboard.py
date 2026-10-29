"""CreditPulse dashboard (report §11): Portfolio monitor · Loan detail · Model health.

Every number is read from saved pipeline artefacts (R-53); nothing is typed in by hand.
Run:  streamlit run app/dashboard.py
Set CREDITPULSE_API=http://localhost:8000 to fetch loan detail from the live scoring service.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from creditpulse.eval.metrics import capture_table, reliability  # noqa: E402
from creditpulse.labels.events import mi_to_date, month_index  # noqa: E402

st.set_page_config(page_title="CreditPulse", page_icon="📈", layout="wide")

INK, ACCENT, WARN, MUTED = "#1d3557", "#2a6f97", "#c8553d", "#8d99ae"
st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 1.6rem; max-width: 1300px;}}
      h1, h2, h3 {{color: {INK}; letter-spacing: -0.01em;}}
      div[data-testid="stMetricValue"] {{font-size: 1.6rem; color: {INK};}}
      .cp-note {{color: {MUTED}; font-size: 0.82rem;}}
      .cp-pill {{display:inline-block; padding:2px 8px; border-radius:10px; background:#eef4f8; color:{ACCENT};
                 font-size:0.75rem; margin-right:6px;}}
      .cp-reason {{border-left: 3px solid {WARN}; padding: 6px 10px; margin: 6px 0; background: #fbf4f2;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    a = ROOT / "artifacts"
    t = ROOT / "reports" / "tables"
    scores = pd.read_parquet(a / "scores" / "monthly_scores.parquet")
    res = pd.read_csv(t / "results_test_p1.csv")
    psi = pd.read_csv(t / "psi_train_vs_test.csv", index_col=0)
    ens = {m: pd.read_parquet(a / "preds" / f"ENS_{m}.parquet") for m in
           ["M1_logreg", "M2_lgbm", "M5_transformer", "M6_multitask", "M7_pretrained"]
           if (a / "preds" / f"ENS_{m}.parquet").exists()}
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet",
                            columns=["account_id", "mi", "inflow", "outflow", "bal_eom", "bal_min", "days_below_zero",
                                     "uver_paid", "loan_payment", "n_tx"])
    serving = json.loads((a / "serving.json").read_text())
    loans = pd.read_parquet(ROOT / "data" / "processed" / "loans_events.parquet")
    return scores, res, psi, ens, panel, serving, loans


scores, res, psi, ens, panel, serving, loans = load()
R = res.set_index("model")
served = serving["model"]

st.sidebar.markdown("## CreditPulse")
st.sidebar.caption("Early-warning monitor for loan distress · Berka 1993–1998")
page = st.sidebar.radio("Page", ["Portfolio monitor", "Loan detail", "Model health"], label_visibility="collapsed")
months = sorted(scores.cutoff_mi.unique())
labels = {m: mi_to_date(m).strftime("%Y-%m") for m in months}
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: `{serving['model_version']}`")
st.sidebar.caption("Decision support for human analysts only. Not for automated approval, rejection, pricing or "
                   "intervention.")


def fmt_ci(m, k):
    return f"{R.loc[m, k]:.3f}  [{R.loc[m, k + '_lo']:.3f}, {R.loc[m, k + '_hi']:.3f}]"


def drivers_of(row):
    try:
        return json.loads(row) if isinstance(row, str) else []
    except json.JSONDecodeError:
        return []


# ------------------------------------------------------------------ page 1
if page == "Portfolio monitor":
    st.title("Portfolio monitor")
    c = st.select_slider("Month-end cutoff", options=months, value=months[-1], format_func=lambda m: labels[m])
    q = scores[scores.cutoff_mi == c].sort_values("queue_rank")
    prev = scores[scores.cutoff_mi == c - 1].set_index("loan_id")
    n = len(q)
    thr = serving["threshold_10pct_queue"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("At-risk loans", f"{n}")
    k2.metric("Top 5% / 10% / 20%", f"{int(np.ceil(n * .05))} / {int(np.ceil(n * .10))} / {int(np.ceil(n * .20))}")
    k3.metric("Above alert threshold", f"{int((q.risk_6m >= thr).sum())}",
              help=f"Threshold {thr:.3f} fills a 10% queue on validation")
    k4.metric("Mean 6-month risk", f"{q.risk_6m.mean():.2%}")

    if len(prev):
        top_now = set(q[q.queue_percentile < 10].loan_id)
        top_prev = set(prev[prev.queue_percentile < 10].index)
        st.markdown(f"<span class='cp-pill'>month-on-month</span> {len(top_now - top_prev)} loans entered the top 10%, "
                    f"{len(top_prev - top_now)} left it.", unsafe_allow_html=True)

    st.subheader("Headline evaluation (P1 out-of-time test)")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("PR-AUC", f"{R.loc[served, 'pr_auc']:.3f}", help=f"95% CI {fmt_ci(served, 'pr_auc')}; prevalence "
              f"{R.loc['M0_prior', 'pr_auc']:.3f}")
    h2.metric("Event capture @ top 10%", f"{R.loc[served, 'capture@10']:.0%}")
    h3.metric("Median lead time", f"{R.loc[served, 'median_lead_months']:.0f} months"
              if pd.notna(R.loc[served, 'median_lead_months']) else "n/a")
    h4.metric("Brier score", f"{R.loc[served, 'brier']:.4f}", help=f"Brier skill {R.loc[served, 'brier_skill']:.2f}")
    st.markdown(f"<p class='cp-note'>Test set: {int(R.loc[served, 'event_loans_test'])} event loans. Intervals are "
                f"95% loan-clustered bootstrap. Berka is 1993–1998 Czech data; treat point values with caution.</p>",
                unsafe_allow_html=True)

    st.subheader(f"Review queue · {labels[c]}")
    q = q.copy()
    q["Δ vs last month"] = q.loan_id.map(prev.risk_6m) if len(prev) else np.nan
    q["Δ vs last month"] = q.risk_6m - q["Δ vs last month"]  # blank = newly in the at-risk queue
    q["top driver"] = q.drivers.map(lambda d: (drivers_of(d) or [{"text": "—"}])[0]["text"])
    show = q[["queue_rank", "loan_id", "risk_6m", "Δ vs last month", "queue_percentile", "loan_age", "top driver"]].head(60)
    st.dataframe(
        show, hide_index=True, use_container_width=True, height=520,
        column_config={
            "risk_6m": st.column_config.ProgressColumn("6-month risk", format="%.3f", min_value=0, max_value=float(max(q.risk_6m.max(), 0.01))),
            "Δ vs last month": st.column_config.NumberColumn(format="%+.3f"),
            "queue_percentile": st.column_config.NumberColumn("percentile", format="%.1f"),
            "loan_age": st.column_config.NumberColumn("loan age (m)"),
        },
    )

# ------------------------------------------------------------------ page 2
elif page == "Loan detail":
    st.title("Loan detail")
    c1, c2 = st.columns([1, 2])
    c = c1.selectbox("Month-end cutoff", months[::-1], format_func=lambda m: labels[m])
    q = scores[scores.cutoff_mi == c].sort_values("queue_rank")
    lid = c2.selectbox("Loan (sorted by queue rank)", q.loan_id.tolist(),
                       format_func=lambda l: f"#{l} · rank {int(q.set_index('loan_id').loc[l, 'queue_rank'])}")
    row = q.set_index("loan_id").loc[lid]
    api = os.environ.get("CREDITPULSE_API")
    live = None
    if api:
        try:
            import httpx

            live = httpx.post(f"{api}/score", json={"loan_id": int(lid), "as_of_date": str(mi_to_date(c).date())},
                              timeout=30).json()
        except Exception as e:  # noqa: BLE001
            st.warning(f"Live API unavailable ({e}); showing batch results.")
    risk = live["risk_6m"] if live else row.risk_6m
    hz = live["hazard_scaled"] if live else [row[f"hs{k}"] for k in range(1, 7)]
    drv = live["drivers"] if live else drivers_of(row.drivers)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("6-month risk (calibrated)", f"{risk:.2%}")
    m2.metric("Queue rank", f"{int(row.queue_rank)} / {len(q)}")
    m3.metric("Percentile", f"top {row.queue_percentile + 100 / len(q):.1f}%")
    m4.metric("Loan age", f"{int(row.loan_age)} months")
    if live:
        st.caption(f"Served live by {api} · {live['model_version']}")

    hist = scores[(scores.loan_id == lid) & (scores.cutoff_mi <= c)].sort_values("cutoff_mi").tail(8)
    g1, g2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[labels[m] for m in hist.cutoff_mi], y=hist.risk_6m, mode="lines+markers",
                             line=dict(color=ACCENT, width=3), name="6-month risk"))
    fig.add_hline(y=serving["threshold_10pct_queue"], line_dash="dot", line_color=WARN,
                  annotation_text="10% queue threshold")
    fig.update_layout(title="Risk trajectory (last 8 months)", height=330, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_tickformat=".1%", xaxis_type="category")
    g1.plotly_chart(fig, use_container_width=True)
    cum = 1 - np.cumprod(1 - np.array(hz))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"+{k}m" for k in range(1, 7)], y=hz, name="monthly hazard", marker_color=ACCENT))
    fig.add_trace(go.Scatter(x=[f"+{k}m" for k in range(1, 7)], y=cum, name="cumulative", line=dict(color=WARN)))
    fig.update_layout(title="When the risk falls: next 6 months", height=330, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_tickformat=".1%", legend=dict(orientation="h"))
    g2.plotly_chart(fig, use_container_width=True)
    st.markdown("<p class='cp-note'>Timing shape from Head B (exploratory, report K2), rescaled so the 6-month "
                "cumulative equals the calibrated Head A risk. The queue ranks on Head A.</p>", unsafe_allow_html=True)

    st.subheader("Why this loan is ranked here")
    if drv:
        for d in drv:
            st.markdown(f"<div class='cp-reason'>{d['text']} <span class='cp-note'>· {d['feature']}</span></div>",
                        unsafe_allow_html=True)
    else:
        st.write("No risk-increasing drivers above zero.")

    st.subheader("Monthly behaviour behind the score")
    acc = int(loans.set_index("loan_id").loc[lid, "account_id"])
    beh = panel[(panel.account_id == acc) & (panel.mi <= c) & (panel.mi > c - 12)].copy()
    beh["month"] = beh.mi.map(lambda m: mi_to_date(m).strftime("%Y-%m"))
    beh = beh[["month", "inflow", "outflow", "bal_eom", "bal_min", "days_below_zero", "uver_paid", "loan_payment", "n_tx"]]
    st.dataframe(beh.iloc[::-1], hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.NumberColumn(format="%.0f") for k in
                                ["inflow", "outflow", "bal_eom", "bal_min", "uver_paid", "loan_payment"]})

# ------------------------------------------------------------------ page 3
else:
    st.title("Model health")
    st.subheader("Out-of-time test results (P1)")
    cols = ["model", "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", "brier", "ece", "capture@5", "capture@10",
            "capture@20", "median_lead_months", "share_lead_ge3", "false_alarms_per_month", "pr_auc_seed_std"]
    st.dataframe(res[[c for c in cols if c in res]].round(3), hide_index=True, use_container_width=True)
    st.markdown("<p class='cp-note'>No model is claimed to outperform another unless its paired-bootstrap interval "
                "excludes zero (R-47). See reports/tables/ranking_claims.csv.</p>", unsafe_allow_html=True)

    a, b = st.columns(2)
    m = a.selectbox("Reliability curve for", list(ens), index=list(ens).index(served) if served in ens else 0)
    split = b.radio("Split", ["val", "test"], horizontal=True)
    d = ens[m][(ens[m].split == split) & ens[m].label.notna()]
    fig = go.Figure()
    for col, nm, colr in [("score_raw", "before calibration", MUTED), ("score", "after Platt", ACCENT)]:
        rr = reliability(d.label.values, d[col].values, 5)
        fig.add_trace(go.Scatter(x=rr.mean_pred, y=rr.obs_rate, mode="lines+markers", name=nm, line=dict(color=colr)))
    lim = float(max(d.score.max(), d.label.mean() * 3, 0.05))
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color="#ccc", dash="dot"), name="perfect"))
    fig.update_layout(height=360, xaxis_title="mean predicted", yaxis_title="observed rate",
                      margin=dict(l=10, r=10, t=10, b=10))
    a.plotly_chart(fig, use_container_width=True)

    ladder = ["M1_logreg", "M2_lgbm", "M2_xgb", "S1_cox", "S2_hazard", "M3_gru", "M4_lstm", "M5_transformer",
              "M6_multitask", "M7_pretrained"]
    cap = res.set_index("model").reindex([m for m in ladder if m in R.index])[["capture@5", "capture@10", "capture@20"]]
    fig = go.Figure()
    for k, colr in zip(cap.columns, ["#a9cce3", ACCENT, INK]):
        fig.add_trace(go.Bar(y=cap.index, x=cap[k], name=k.replace("capture@", "top "), marker_color=colr,
                             orientation="h"))
    fig.update_layout(title="Backtest: event loans flagged before their event", height=420, barmode="group",
                      xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"),
                      yaxis=dict(autorange="reversed"))
    b.plotly_chart(fig, use_container_width=True)

    e = ens[served]
    te = e[e.split == "test"]
    ct = capture_table(te.rename(columns={}), threshold=serving["threshold_10pct_queue"])
    lead_counts = ct.lead.dropna().astype(int).value_counts().reindex(range(1, 7), fill_value=0)
    fig = go.Figure(go.Bar(x=[f"{k} m" for k in lead_counts.index], y=lead_counts.values, marker_color=ACCENT))
    fig.update_layout(title=f"Lead time at the 10% queue threshold ({served}, test)", xaxis_title="months before event",
                      yaxis_title="event loans", height=320, margin=dict(l=10, r=10, t=40, b=10), bargap=0.3)
    a2, b2 = st.columns(2)
    a2.plotly_chart(fig, use_container_width=True)
    b2.markdown("**Feature drift (PSI, train vs test, cutoff month)**")
    b2.dataframe(psi.head(15).round(3), use_container_width=True, height=300)
    st.markdown("<p class='cp-note'>Monitoring ideas (documented, not built): PSI on inputs each month, calibration "
                "over time, alert volume per month, retraining triggers.</p>", unsafe_allow_html=True)
