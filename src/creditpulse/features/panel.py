"""Stage 3 — monthly account panel (report §7.2) and the per-month feature catalogue (§7.3).

Every value in the row for month m uses only transactions dated in or before month m, so
tokens for a cutoff never see the future (R-18). Built for all 4,500 accounts (pretraining).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from creditpulse.labels.events import loan_schedule, month_index
from creditpulse.utils import load_config, p

MONEY = {  # money-valued features get the signed-log transform before scaling
    "inflow", "outflow", "net_flow", "bal_eom", "bal_min", "bal_max", "bal_mean", "bal_std",
    "amt_mean", "amt_median", "amt_max", "sipo_amt", "insurance_amt", "fees_amt", "pension_amt",
    "interest_amt", "uver_paid", "shortfall", "loan_amount", "loan_payment", "instal_to_inflow",
    "d_bal_eom", "d_inflow", "d_outflow", "inflow_rm3", "inflow_rm6", "outflow_rm3", "outflow_rm6",
    "net_rm3", "net_rm6", "bal_rm3", "bal_rm6", "inflow_rs6", "outflow_rs6", "bal_rs6", "bal_slope6",
}

FEATURE_GROUPS = {
    "cash_flow": ["inflow", "outflow", "net_flow", "log_in_out_ratio", "neg_net_share6"],
    "balance": ["bal_eom", "bal_min", "bal_max", "bal_mean", "bal_std", "days_below_zero", "days_below_10pct"],
    "activity": ["n_tx", "n_credit", "n_debit", "amt_mean", "amt_median", "amt_max", "active_days",
                 "max_debit_share", "n_counterparties", "counterparty_share"],
    "channel_mix": ["sh_cash_withdrawal", "sh_card_withdrawal", "sh_transfer_out", "sh_transfer_in", "sh_cash_deposit"],
    "obligations": ["sipo_amt", "insurance_amt", "fees_amt", "pension_amt", "interest_amt"],
    "loan_servicing": ["uver_paid", "uver_flag", "due_flag", "shortfall", "instal_to_inflow"],
    "distress": ["missed_flag", "penalty_flag", "penalty_count"],
    "dynamics": ["d_bal_eom", "d_inflow", "d_outflow", "inflow_rm3", "inflow_rm6", "outflow_rm3", "outflow_rm6",
                 "net_rm3", "net_rm6", "bal_rm3", "bal_rm6", "inflow_rs6", "outflow_rs6", "bal_rs6",
                 "bal_slope6", "inflow_z", "bal_z", "hist_short"],
    "loan_static": ["has_loan", "loan_amount", "loan_duration", "loan_payment", "payment_to_amount",
                    "loan_age", "remaining_term", "pre_loan"],
    "account_static": ["account_age", "freq_monthly", "freq_weekly", "freq_after_tx", "has_card",
                       "card_gold", "card_junior", "n_disponents", "dist_log_pop", "dist_urban",
                       "dist_salary", "dist_unemp", "dist_entrepreneurs", "dist_crime"],
    "time": ["month_sin", "month_cos"],
    "masks": ["inactive"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def daily_balance_month_stats(tr: pd.DataFrame, accounts: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """End-of-day balance carried forward over every calendar day, then summarised per month."""
    eod = tr.groupby(["account_id", "date"], sort=True)["balance"].last().reset_index()
    first = eod.groupby("account_id")["date"].min()
    base = pd.Timestamp("1993-01-01")
    eod["d"] = (eod["date"] - base).dt.days
    end_d = (end - base).days
    pieces = []
    for aid, g in eod.groupby("account_id", sort=False):
        d0 = int(g["d"].iloc[0])
        n = end_d - d0 + 1
        vals = np.full(n, np.nan)
        vals[g["d"].to_numpy() - d0] = g["balance"].to_numpy()
        idx = np.where(~np.isnan(vals), np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        vals = vals[idx]
        days = base + pd.to_timedelta(np.arange(d0, end_d + 1), unit="D")
        pieces.append(pd.DataFrame({"account_id": aid, "mi": days.year * 12 + days.month - 1, "bal": vals}))
    daily = pd.concat(pieces, ignore_index=True)
    g = daily.groupby(["account_id", "mi"])["bal"]
    out = pd.DataFrame({
        "bal_eom": g.last(), "bal_min": g.min(), "bal_max": g.max(), "bal_mean": g.mean(),
        "bal_std": g.std(ddof=0),
        "days_below_zero": (daily["bal"] < 0).groupby([daily.account_id, daily.mi]).sum(),
    })
    daily = daily.join(out["bal_mean"], on=["account_id", "mi"])
    thr = 0.1 * daily["bal_mean"].clip(lower=0)
    out["days_below_10pct"] = (daily["bal"] < thr).groupby([daily.account_id, daily.mi]).sum()
    out = out.reset_index()
    out["first_mi"] = out.account_id.map(month_index(first))
    return out


def build_panel(
    trans: pd.DataFrame,
    accounts: pd.DataFrame,
    loans: pd.DataFrame,
    district: pd.DataFrame,
    card: pd.DataFrame,
    disp: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    end = pd.Timestamp(cfg["data"]["data_end"])
    thr = cfg["labels"]["missed_threshold"]
    tr = trans[trans.is_disbursement == 0].copy()  # C9
    tr["mi"] = month_index(tr["date"])
    accs = accounts[accounts.account_id.isin(tr.account_id.unique())]

    bal = daily_balance_month_stats(tr, accs, end)

    # ---- transaction aggregates per account-month
    tr["credit"] = (tr.type == "credit").astype(int)
    tr["debit_amt"] = np.where(tr.type == "debit", tr.amount, 0.0)
    g = tr.groupby(["account_id", "mi"])
    agg = pd.DataFrame({
        "inflow": tr.amount.where(tr.type == "credit", 0.0).groupby([tr.account_id, tr.mi]).sum(),
        "outflow": g["debit_amt"].sum(),
        "n_tx": g.size(),
        "n_credit": g["credit"].sum(),
        "amt_mean": g["amount"].mean(),
        "amt_median": g["amount"].median(),
        "amt_max": g["amount"].max(),
        "active_days": g["date"].nunique(),
        "max_debit": g["debit_amt"].max(),
        "counterparty_share": g["has_counterparty"].mean(),
    })
    agg["n_counterparties"] = tr[tr.counterparty != ""].groupby(["account_id", "mi"])["counterparty"].nunique()
    agg["n_counterparties"] = agg["n_counterparties"].fillna(0)
    agg["n_debit"] = agg["n_tx"] - agg["n_credit"]
    for op, col in [("cash_withdrawal", "sh_cash_withdrawal"), ("card_withdrawal", "sh_card_withdrawal"),
                    ("transfer_out", "sh_transfer_out"), ("transfer_in", "sh_transfer_in"),
                    ("cash_deposit", "sh_cash_deposit")]:
        agg[col] = (tr.operation == op).groupby([tr.account_id, tr.mi]).mean()
    for ks, col in [("household", "sipo_amt"), ("insurance", "insurance_amt"), ("fees", "fees_amt"),
                    ("pension", "pension_amt"), ("interest", "interest_amt"), ("loan_payment", "uver_paid")]:
        agg[col] = tr.amount.where(tr.k_symbol == ks, 0.0).groupby([tr.account_id, tr.mi]).sum()
    agg["penalty_count"] = (tr.k_symbol == "penalty_interest").groupby([tr.account_id, tr.mi]).sum()
    agg = agg.reset_index()

    # ---- full month grid, explicit zero-activity rows (R-21)
    panel = bal.merge(agg, on=["account_id", "mi"], how="left")
    zero_cols = [c for c in agg.columns if c not in ("account_id", "mi")]
    panel["inactive"] = panel["n_tx"].isna().astype(int)
    panel[zero_cols] = panel[zero_cols].fillna(0.0)
    panel["net_flow"] = panel["inflow"] - panel["outflow"]
    panel["log_in_out_ratio"] = np.log1p(panel["inflow"]) - np.log1p(panel["outflow"])
    panel["max_debit_share"] = np.where(panel.outflow > 0, panel.max_debit / panel.outflow.replace(0, 1), 0.0)
    panel["penalty_flag"] = (panel["penalty_count"] > 0).astype(int)
    panel = panel.drop(columns=["max_debit"])

    # ---- loan servicing (causal: schedule is fixed at origination)
    lo = loan_schedule(loans, trans).set_index("account_id")
    li = panel.account_id.map
    panel["has_loan"] = panel.account_id.isin(lo.index).astype(int)
    start = li(lo["start_mi"])
    first_due, last_due = li(lo["first_due_mi"]), li(lo["last_due_mi"])
    pay = li(lo["payments"])
    panel["loan_amount"] = li(lo["amount"]).fillna(0.0)
    panel["loan_duration"] = li(lo["duration"]).fillna(0.0)
    panel["loan_payment"] = pay.fillna(0.0)
    panel["payment_to_amount"] = (pay / li(lo["amount"])).fillna(0.0)
    panel["loan_age"] = (panel.mi - start).fillna(-99.0)
    panel["pre_loan"] = ((panel.has_loan == 1) & (panel.mi < start)).astype(int)
    panel["remaining_term"] = (last_due - panel.mi).clip(lower=0).fillna(0.0)
    panel.loc[panel.pre_loan == 1, "remaining_term"] = panel.loc[panel.pre_loan == 1, "loan_duration"]
    due = (panel.mi >= first_due) & (panel.mi <= last_due)
    panel["due_flag"] = due.astype(int)
    panel["uver_flag"] = (panel.uver_paid > 0).astype(int)
    panel["shortfall"] = np.where(due, (pay - panel.uver_paid).clip(lower=0), 0.0)
    panel["missed_flag"] = (due & (panel.uver_paid < thr * pay)).astype(int)
    panel["instal_to_inflow"] = np.where(due, pay / (panel.inflow + 1.0), 0.0)
    panel["loan_age"] = np.where(panel.has_loan == 1, panel.loan_age, 0.0)
    # penalty interest before origination is a legitimate input; after origination it IS the event
    # and the at-risk filter (D2) guarantees it is zero before any cutoff.

    # ---- dynamics (causal rolling windows over the account's own past)
    panel = panel.sort_values(["account_id", "mi"]).reset_index(drop=True)
    ga = panel.groupby("account_id")
    panel["d_bal_eom"] = ga["bal_eom"].diff().fillna(0.0)
    panel["d_inflow"] = ga["inflow"].diff().fillna(0.0)
    panel["d_outflow"] = ga["outflow"].diff().fillna(0.0)
    def roll(col, w, fn, mp=1):
        r = getattr(ga[col].rolling(w, min_periods=mp), fn)(**({"ddof": 0} if fn == "std" else {}))
        return r.reset_index(level=0, drop=True)

    for col, nm in [("inflow", "inflow"), ("outflow", "outflow"), ("net_flow", "net"), ("bal_eom", "bal")]:
        for w in (3, 6):
            panel[f"{nm}_rm{w}"] = roll(col, w, "mean")
    for col, nm in [("inflow", "inflow"), ("outflow", "outflow"), ("bal_eom", "bal")]:
        panel[f"{nm}_rs6"] = roll(col, 6, "std", 2).fillna(0.0)
    panel["_neg"] = (panel.net_flow < 0).astype(float)
    ga = panel.groupby("account_id")
    panel["neg_net_share6"] = roll("_neg", 6, "mean")
    tt = np.arange(6) - 2.5  # least-squares slope over the last 6 month-ends
    sl = sum(tt[5 - j] * ga["bal_eom"].shift(j) for j in range(6)) / float(np.dot(tt, tt))
    panel["bal_slope6"] = sl.fillna(0.0)
    for col, nm in [("inflow", "inflow_z"), ("bal_eom", "bal_z")]:
        sh = ga[col].shift(1)
        cnt = ga.cumcount()
        csum = sh.fillna(0).groupby(panel.account_id).cumsum()
        csq = (sh.fillna(0) ** 2).groupby(panel.account_id).cumsum()
        mu = csum / cnt.replace(0, np.nan)
        sd = np.sqrt((csq / cnt.replace(0, np.nan) - mu**2).clip(lower=0))
        panel[nm] = ((panel[col] - mu) / (sd + 1.0)).fillna(0.0).clip(-10, 10)
    panel = panel.drop(columns="_neg")
    panel["hist_short"] = (ga.cumcount() < 5).astype(int)

    # ---- account static
    a = accs.set_index("account_id")
    panel["account_age"] = panel.mi - panel.account_id.map(month_index(a["date"]))
    fq = panel.account_id.map(a["frequency"])
    panel["freq_monthly"] = (fq == "monthly").astype(int)
    panel["freq_weekly"] = (fq == "weekly").astype(int)
    panel["freq_after_tx"] = (fq == "after_transaction").astype(int)
    d = district.set_index("district_id")
    did = panel.account_id.map(a["district_id"])
    panel["district_id"] = did
    panel["dist_log_pop"] = np.log(did.map(d["population"]))
    panel["dist_urban"] = did.map(d["urban_ratio"])
    panel["dist_salary"] = did.map(d["avg_salary"])
    panel["dist_unemp"] = did.map(d["unemp_96"])
    panel["dist_entrepreneurs"] = did.map(d["entrepreneurs_per_1k"])
    panel["dist_crime"] = did.map(d["crimes_96"]) / did.map(d["population"]) * 1000
    # card: causal (issued date known); disponents from disp
    cd = card.merge(disp[["disp_id", "account_id"]], on="disp_id")
    cd["issued_mi"] = month_index(cd["issued"])
    cd = cd.sort_values("issued_mi").drop_duplicates("account_id").set_index("account_id")
    iss = panel.account_id.map(cd["issued_mi"])
    has = panel.mi >= iss
    panel["has_card"] = has.astype(int)
    ct = panel.account_id.map(cd["type"])
    panel["card_gold"] = (has & (ct == "gold")).astype(int)
    panel["card_junior"] = (has & (ct == "junior")).astype(int)
    panel["n_disponents"] = panel.account_id.map(disp[disp.type == "DISPONENT"].groupby("account_id").size()).fillna(0)

    m = panel.mi % 12
    panel["month_sin"] = np.sin(2 * np.pi * m / 12)
    panel["month_cos"] = np.cos(2 * np.pi * m / 12)
    panel = panel.sort_values(["account_id", "mi"]).reset_index(drop=True)
    return panel[["account_id", "mi", "first_mi", "district_id"] + FEATURES]


def load_interim(cfg: dict) -> dict[str, pd.DataFrame]:
    interim = p(cfg, "interim")
    return {k: pd.read_parquet(interim / f"{k}.parquet") for k in
            ["account", "loan", "trans", "district", "card", "disp", "client"]}


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    t = load_interim(cfg)
    panel = build_panel(t["trans"], t["account"], t["loan"], t["district"], t["card"], t["disp"], cfg)
    panel.to_parquet(p(cfg, "processed") / "panel.parquet", index=False)
    # R-21 completeness: no gaps, flows reconcile with raw transactions
    span = panel.groupby("account_id").mi.agg(["min", "max", "size"])
    gaps = int(((span["max"] - span["min"] + 1) != span["size"]).sum())
    tr = t["trans"][t["trans"].is_disbursement == 0]
    raw_in = tr.loc[tr.type == "credit", "amount"].sum()
    raw_out = tr.loc[tr.type == "debit", "amount"].sum()
    chk = {
        "R-21": {
            "pass": bool(gaps == 0 and abs(panel.inflow.sum() - raw_in) < 1 and abs(panel.outflow.sum() - raw_out) < 1),
            "accounts": int(panel.account_id.nunique()), "rows": len(panel), "accounts_with_gaps": gaps,
            "inflow_diff": float(panel.inflow.sum() - raw_in), "outflow_diff": float(panel.outflow.sum() - raw_out),
            "n_features": len(FEATURES),
        }
    }
    (p(cfg, "reports") / "tables" / "panel_checks.json").write_text(json.dumps(chk, indent=2))
    return chk


if __name__ == "__main__":
    print(run())
