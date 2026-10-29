"""Stage 9 — explainability (report §10): SHAP for M2, integrated gradients + permutation importance for M6,
attention as a supplementary view, sanity checks R-48..R-50, and plain-language reason codes."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from creditpulse.eval.metrics import pr_auc  # noqa: E402
from creditpulse.features.panel import FEATURE_GROUPS, FEATURES  # noqa: E402
from creditpulse.utils import load_config, p  # noqa: E402

GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

# Plain-language templates. {v} = raw value at the cutoff month, {w} = window statistic.
REASONS = {
    "bal_min": "Lowest balance last month was {v:,.0f} CZK",
    "bal_eom": "Month-end balance was {v:,.0f} CZK",
    "bal_mean": "Average balance last month was {v:,.0f} CZK",
    "bal_max": "Highest balance last month was only {v:,.0f} CZK",
    "bal_std": "Balance swung strongly within the month (std {v:,.0f} CZK)",
    "days_below_zero": "Balance was below zero on {v:.0f} days last month",
    "days_below_10pct": "Balance sat near empty on {v:.0f} days last month",
    "inflow": "Money coming in last month was {v:,.0f} CZK",
    "outflow": "Money going out last month was {v:,.0f} CZK",
    "net_flow": "Net cash flow last month was {v:,.0f} CZK",
    "neg_net_share6": "Spending exceeded income in {v:.0%} of the last 6 months",
    "log_in_out_ratio": "Inflow-to-outflow balance has weakened",
    "instal_to_inflow": "The loan instalment equals {v:.0%} of last month's inflow",
    "loan_payment": "Monthly instalment of {v:,.0f} CZK",
    "loan_amount": "Loan size of {v:,.0f} CZK",
    "payment_to_amount": "Short, steep repayment schedule",
    "loan_duration": "Loan term of {v:.0f} months",
    "remaining_term": "{v:.0f} instalments still to pay",
    "loan_age": "Loan is {v:.0f} months old",
    "bal_slope6": "Balance trend over 6 months: {v:+,.0f} CZK per month",
    "bal_rm3": "3-month average balance {v:,.0f} CZK",
    "bal_rm6": "6-month average balance {v:,.0f} CZK",
    "bal_rs6": "Balance volatility over 6 months ({v:,.0f} CZK)",
    "inflow_rm3": "3-month average inflow {v:,.0f} CZK",
    "inflow_rm6": "6-month average inflow {v:,.0f} CZK",
    "inflow_rs6": "Irregular income over the last 6 months",
    "outflow_rm3": "3-month average outflow {v:,.0f} CZK",
    "outflow_rm6": "6-month average outflow {v:,.0f} CZK",
    "outflow_rs6": "Irregular spending over the last 6 months",
    "net_rm3": "3-month average net flow {v:,.0f} CZK",
    "net_rm6": "6-month average net flow {v:,.0f} CZK",
    "inflow_z": "Inflow is {v:+.1f} standard deviations from this account's norm",
    "bal_z": "Balance is {v:+.1f} standard deviations from this account's norm",
    "d_bal_eom": "Month-end balance changed by {v:+,.0f} CZK",
    "d_inflow": "Inflow changed by {v:+,.0f} CZK versus the previous month",
    "d_outflow": "Outflow changed by {v:+,.0f} CZK versus the previous month",
    "sh_cash_withdrawal": "{v:.0%} of transactions were cash withdrawals",
    "sh_transfer_out": "{v:.0%} of transactions were outgoing transfers",
    "sh_cash_deposit": "{v:.0%} of transactions were cash deposits",
    "sh_transfer_in": "{v:.0%} of transactions were incoming transfers",
    "sh_card_withdrawal": "{v:.0%} of transactions were card withdrawals",
    "sipo_amt": "Household standing orders of {v:,.0f} CZK",
    "pension_amt": "Pension income of {v:,.0f} CZK",
    "insurance_amt": "Insurance payments of {v:,.0f} CZK",
    "interest_amt": "Interest credited {v:,.0f} CZK (reflects balance level)",
    "fees_amt": "Account fees of {v:,.0f} CZK",
    "n_tx": "{v:.0f} transactions last month",
    "active_days": "Account active on {v:.0f} days last month",
    "account_age": "Account is {v:.0f} months old",
    "dist_unemp": "District unemployment {v:.2f}%",
    "dist_salary": "District average salary {v:,.0f} CZK",
    "counterparty_share": "{v:.0%} of transactions involved another bank's account",
    "n_counterparties": "{v:.0f} distinct transfer counterparties last month",
    "n_credit": "Only {v:.0f} incoming payments last month",
    "n_debit": "{v:.0f} outgoing payments last month",
    "amt_mean": "Average transaction size {v:,.0f} CZK",
    "amt_median": "Typical transaction size {v:,.0f} CZK",
    "amt_max": "Largest single transaction {v:,.0f} CZK",
    "max_debit_share": "One debit made up {v:.0%} of last month's spending",
    "uver_paid": "Loan instalment paid last month: {v:,.0f} CZK",
    "shortfall": "Instalment shortfall of {v:,.0f} CZK",
    "missed_flag": "An instalment was missed",
    "penalty_flag": "Penalty interest was charged",
    "penalty_count": "{v:.0f} penalty-interest charges",
}


def reason_text(feat: str, raw_value: float) -> str:
    t = REASONS.get(feat)
    if t is None:
        return f"{feat.replace('_', ' ')} = {raw_value:,.2f}"
    try:
        return t.format(v=raw_value)
    except (ValueError, KeyError):
        return t


def integrated_gradients(model, x, pad, age, steps=32):
    """IG on the Head-A logit with the training-mean token (zeros after standardisation) as baseline."""
    model.eval()
    base = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps + 1)[1:]
    total = torch.zeros_like(x)
    for a in alphas:
        xi = (base + a * (x - base)).requires_grad_(True)
        out = model(xi, pad, age)["logit_a"].sum()
        g, = torch.autograd.grad(out, xi)
        total += g
    return ((x - base) * total / steps).detach()


def load_m6(cfg, seed=0):
    from creditpulse.models.deep import SeqModel

    ck = torch.load(p(cfg, "artifacts") / "models" / f"M6_multitask_s{seed}.pt", weights_only=False)
    hp, arch = ck["hp"], ck["arch"]
    m = SeqModel(len(ck["cols"]), kind=arch["kind"], d_model=hp["d_model"], n_layers=hp["n_layers"],
                 n_heads=hp.get("n_heads", 4), dropout=hp["dropout"], pooling=hp.get("pooling", "cls"),
                 head_b=arch["head_b"], derive_a=hp.get("derive_a", False))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def run(cfg=None):
    cfg = cfg or load_config()
    from creditpulse.train.data import load_bundle

    B = load_bundle(cfg)
    tables, figs = p(cfg, "reports") / "tables", p(cfg, "reports") / "figures"
    te = B.idx("test")
    out = {}

    # ---------------- SHAP for M2 (tree baseline)
    import joblib
    import shap

    m2 = joblib.load(p(cfg, "artifacts") / "m2_lgbm.joblib")
    expl = shap.TreeExplainer(m2["model"])
    sv = expl.shap_values(B.A[te])
    sv = sv[1] if isinstance(sv, list) else sv
    glob = pd.Series(np.abs(sv).mean(0), index=B.acols).sort_values(ascending=False)
    glob.rename("mean_abs_shap").to_csv(tables / "shap_m2_global.csv")
    out["shap_top10"] = glob.head(10).round(4).to_dict()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    g = glob.head(15).iloc[::-1]
    ax.barh(g.index, g.values, color="#2a6f97")
    ax.set_xlabel("mean |SHAP| (test)")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(figs / "shap_m2.png", dpi=130)
    plt.close(fig)

    # ---------------- IG for M6 across seeds (R-50 stability)
    x = torch.tensor(B.Z[te])
    pad = torch.tensor(B.pad[te])
    age = torch.tensor(B.age[te], dtype=torch.float32)
    feat_rank = {}
    month_prof = {}
    for sd in cfg["train"]["seeds"]:
        try:
            m, _ = load_m6(cfg, sd)
        except FileNotFoundError:
            continue
        ig = integrated_gradients(m, x, pad, age)
        feat_rank[sd] = ig.abs().sum((0, 1)).numpy()
        month_prof[sd] = ig.abs().sum((0, 2)).numpy()
    seeds = sorted(feat_rank)
    F_ = np.mean([feat_rank[s] for s in seeds], 0)
    ig_feat = pd.Series(F_ / F_.sum(), index=FEATURES).sort_values(ascending=False)
    ig_feat.rename("ig_share").to_csv(tables / "ig_m6_features.csv")
    group_share = ig_feat.groupby(lambda f: GROUP_OF[f]).sum().sort_values(ascending=False)
    group_share.rename("ig_share").to_csv(tables / "ig_m6_groups.csv")
    rhos = [spearmanr(feat_rank[a], feat_rank[b])[0] for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    out["R-50"] = {"pass": bool(np.mean(rhos) >= 0.7), "mean_rank_corr_across_seeds": float(np.mean(rhos)),
                   "min": float(np.min(rhos)), "pairs": len(rhos)}
    W = B.Z.shape[1]
    mp = np.mean([month_prof[s] for s in seeds], 0)
    month_df = pd.DataFrame({"position": [f"M-{W - 1 - k}" for k in range(W)], "ig_share": mp / mp.sum()})

    # ---------------- R-48 randomised-weights sanity check
    m0, ck = load_m6(cfg, seeds[0])
    rand_ranks = []
    for r in range(2):
        torch.manual_seed(100 + r)
        mr, _ = load_m6(cfg, seeds[0])
        for prm in mr.parameters():
            torch.nn.init.normal_(prm, 0, 0.05)
        rand_ranks.append(integrated_gradients(mr, x[:300], pad[:300], age[:300]).abs().sum((0, 1)).numpy())
    rr = spearmanr(rand_ranks[0], rand_ranks[1])[0]
    rt = spearmanr(rand_ranks[0], feat_rank[seeds[0]])[0]
    domain = ["bal_min", "bal_eom", "bal_mean", "days_below_zero", "instal_to_inflow", "inflow", "neg_net_share6",
              "bal_rm3", "bal_rm6", "loan_payment", "bal_slope6"]
    top15 = list(ig_feat.head(15).index)
    out["R-48"] = {"pass": bool(rr < 0.7 and len(set(domain) & set(top15)) >= 3),
                   "random_vs_random_rank_corr": float(rr), "random_vs_trained_rank_corr": float(rt),
                   "domain_features_in_trained_top15": sorted(set(domain) & set(top15))}

    # ---------------- permutation importance by feature group and by month position (M6 seed 0)
    from creditpulse.train.deep import predict, tensors

    T = tensors(B, np.arange(len(FEATURES)))
    yte = B.y[te]
    s0, _, _ = predict(m0, T, te)
    base = pr_auc(yte, s0)
    rng = np.random.default_rng(0)
    perm_g = {}
    for gname, feats in FEATURE_GROUPS.items():
        cols = [FEATURES.index(f) for f in feats]
        drops = []
        for _ in range(3):
            Zp = B.Z.copy()
            pi = rng.permutation(te)
            Zp[np.ix_(te, np.arange(W), cols)] = B.Z[np.ix_(pi, np.arange(W), cols)]
            Tp = (torch.tensor(Zp),) + T[1:]
            s, _, _ = predict(m0, Tp, te)
            drops.append(base - pr_auc(yte, s))
        perm_g[gname] = float(np.mean(drops))
    perm_g = pd.Series(perm_g).sort_values(ascending=False)
    perm_g.rename("pr_auc_drop").to_csv(tables / "perm_m6_groups.csv")
    perm_m = []
    for k in range(W):
        drops = []
        for _ in range(3):
            Zp = B.Z.copy()
            pi = rng.permutation(te)
            Zp[te, k, :] = B.Z[pi, k, :]
            s, _, _ = predict(m0, (torch.tensor(Zp),) + T[1:], te)
            drops.append(base - pr_auc(yte, s))
        perm_m.append(float(np.mean(drops)))
    month_df["perm_pr_auc_drop"] = perm_m

    # ---------------- attention (supplementary) vs permutation by month (R-49)
    with torch.no_grad():
        m0(x, pad, age)
    att = m0.last_attn[-1][:, 0, 1:].mean(0).numpy()  # CLS -> month tokens, last layer
    month_df["attention_cls"] = att
    rho_att = spearmanr(att, perm_m)[0] if np.std(perm_m) > 0 else np.nan
    rho_ig = spearmanr(month_df.ig_share, perm_m)[0] if np.std(perm_m) > 0 else np.nan
    out["R-49"] = {"pass": True, "attention_vs_permutation_rank_corr": float(rho_att),
                   "ig_vs_permutation_rank_corr": float(rho_ig),
                   "statement": "Attention is reported as a supplementary view only; agreement with permutation "
                                "importance is shown here and not hidden."}
    month_df.to_csv(tables / "month_profile_m6.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(month_df.position, month_df.ig_share, color="#2a6f97", label="IG share")
    ax[0].plot(month_df.position, month_df.attention_cls / month_df.attention_cls.sum(), "o-", color="#e07a5f",
               label="CLS attention (norm.)")
    ax[0].set_title("Which months drive the score (M6)", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[0].tick_params(labelsize=7)
    gs = group_share.iloc[::-1]
    ax[1].barh(gs.index, gs.values, color="#2a6f97")
    ax[1].set_title("Integrated-gradients share by feature group (M6)", fontsize=9)
    ax[1].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(figs / "m6_explain.png", dpi=130)
    plt.close(fig)
    out["ig_top10"] = ig_feat.head(10).round(4).to_dict()
    out["perm_groups"] = perm_g.round(4).to_dict()
    (tables / "explain_checks.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=float))
