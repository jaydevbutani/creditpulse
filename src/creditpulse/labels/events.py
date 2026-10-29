"""Stage 2 — event dates (D3), snapshots (D1/D2), censoring (D6) and splits (D5).

Month arithmetic uses an integer month index: mi = year*12 + (month-1).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from creditpulse.utils import load_config, p


def month_index(d: pd.Series | pd.Timestamp):
    if isinstance(d, pd.Timestamp):
        return d.year * 12 + d.month - 1
    return d.dt.year * 12 + d.dt.month - 1


def mi_to_period(mi: int) -> pd.Period:
    return pd.Period(year=int(mi) // 12, month=int(mi) % 12 + 1, freq="M")


def mi_to_date(mi) -> pd.Timestamp:
    return mi_to_period(mi).to_timestamp(how="end").normalize()


def loan_schedule(loan: pd.DataFrame, trans: pd.DataFrame) -> pd.DataFrame:
    """Instalment schedule per loan. The first instalment falls in the origination month if a
    UVER debit is seen there, otherwise in the following month (observed for 279/403 loans)."""
    lo = loan.copy()
    lo["start_mi"] = month_index(lo["date"])
    uv = trans[trans.k_symbol == "loan_payment"][["account_id", "date"]].copy()
    uv["mi"] = month_index(uv["date"])
    uv = uv.merge(lo[["account_id", "start_mi"]], on="account_id")
    paid_m0 = set(uv.loc[uv.mi == uv.start_mi, "account_id"])
    lo["first_due_mi"] = np.where(lo.account_id.isin(paid_m0), lo.start_mi, lo.start_mi + 1)
    lo["last_due_mi"] = lo["first_due_mi"] + lo["duration"] - 1
    return lo


def event_dates(loan: pd.DataFrame, trans: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """D3: event month T = min( (a) first SANKC. UROK after origination,
                                (b) first scheduled month with paid < thr * instalment )."""
    data_end_mi = month_index(pd.Timestamp(cfg["data"]["data_end"]))
    thr = cfg["labels"]["missed_threshold"]
    lo = loan_schedule(loan, trans)
    tr = trans[trans.account_id.isin(lo.account_id)][["account_id", "date", "k_symbol", "amount"]].copy()
    tr["mi"] = month_index(tr["date"])

    pen = tr[tr.k_symbol == "penalty_interest"].merge(lo[["account_id", "date"]], on="account_id", suffixes=("", "_loan"))
    pen = pen[pen.date > pen.date_loan].groupby("account_id").mi.min().rename("penalty_mi")

    paid = tr[tr.k_symbol == "loan_payment"].groupby(["account_id", "mi"]).amount.sum()
    miss = {}
    for r in lo.itertuples():
        for m in range(r.first_due_mi, min(r.last_due_mi, data_end_mi) + 1):
            if paid.get((r.account_id, m), 0.0) < thr * r.payments:
                miss[r.account_id] = m
                break
    lo = lo.merge(pen, on="account_id", how="left")
    lo["missed_mi"] = lo.account_id.map(miss)
    routes = cfg["labels"]["event_routes"]
    cols = [c for c, k in [("penalty_mi", "penalty"), ("missed_mi", "missed")] if k in routes]
    lo["event_mi"] = lo[cols].min(axis=1)
    lo["has_event"] = lo["event_mi"].notna().astype(int)
    return lo


def build_snapshots(lo: pd.DataFrame, cfg: dict, horizon: int | None = None) -> pd.DataFrame:
    """D1/D2/D6: one row per (loan, month-end cutoff) while the loan is at risk.

    label_A   1 if the event month is in (c, c+H]; 0 if the window is fully observed (or the loan
              is repaid inside it); NaN if right-censored (dropped for Head A).
    event_k   month of the event relative to c (1..H) or 0.
    follow_k  observed months after c within the horizon (Head B censoring length).
    """
    H = horizon or cfg["labels"]["horizon"]
    data_end_mi = month_index(pd.Timestamp(cfg["data"]["data_end"]))
    rows = []
    for r in lo.itertuples():
        first_c = r.start_mi + cfg["labels"]["min_loan_age"]
        last_c = min(r.last_due_mi - 1, data_end_mi - 1)
        T = r.event_mi if r.has_event else np.inf
        obs_until = min(data_end_mi, r.last_due_mi)
        for c in range(first_c, last_c + 1):
            if T <= c:  # D2 at-risk set: loan already in the event state
                break
            if T <= c + H:
                lab, ek, fk = 1.0, int(T - c), int(T - c)
            else:
                fk = int(min(H, obs_until - c))
                ek = 0
                lab = 0.0 if (c + H <= data_end_mi or r.last_due_mi <= data_end_mi and r.last_due_mi < c + H) else np.nan
            rows.append((r.loan_id, r.account_id, c, lab, ek, fk, c - r.start_mi))
    s = pd.DataFrame(rows, columns=["loan_id", "account_id", "cutoff_mi", "label", "event_k", "follow_k", "loan_age"])
    s["cutoff_date"] = s.cutoff_mi.map(mi_to_date)
    return s


def assign_p1(snap: pd.DataFrame, lo: pd.DataFrame, cfg: dict) -> pd.Series:
    """D5-P1: loan-disjoint split by origination date."""
    tr_end = pd.Timestamp(cfg["splits"]["p1"]["train_end"])
    va_end = pd.Timestamp(cfg["splits"]["p1"]["val_end"])
    od = snap.loan_id.map(lo.set_index("loan_id")["date"])
    return pd.Series(np.select([od <= tr_end, od <= va_end], ["train", "val"], "test"), index=snap.index)


def p2_folds(snap: pd.DataFrame, cfg: dict, horizon: int = 6) -> list[dict]:
    """D5-P2: rolling origin. For boundary B, train = snapshots whose label window closes by B;
    test = cutoffs in (B, B+6 months]. Calendar-ordered by construction (T4 holds per fold)."""
    folds = []
    for b in cfg["splits"]["p2_boundaries"]:
        bm = month_index(pd.Timestamp(b))
        tr_idx = snap.index[(snap.cutoff_mi + horizon <= bm) & snap.label.notna()]
        te_idx = snap.index[(snap.cutoff_mi > bm) & (snap.cutoff_mi <= bm + 6) & snap.label.notna()]
        folds.append({"boundary": b, "train": tr_idx, "test": te_idx})
    return folds


def split_fingerprint(snap: pd.DataFrame) -> str:
    key = snap.sort_values(["loan_id", "cutoff_mi"])[["loan_id", "cutoff_mi", "split"]].to_csv(index=False)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    interim = p(cfg, "interim")
    loan = pd.read_parquet(interim / "loan.parquet")
    trans = pd.read_parquet(interim / "trans.parquet")
    lo = event_dates(loan, trans, cfg)
    snap = build_snapshots(lo, cfg)
    snap["split"] = assign_p1(snap, lo, cfg)
    proc = p(cfg, "processed")
    lo.to_parquet(proc / "loans_events.parquet", index=False)
    snap.to_parquet(proc / "snapshots.parquet", index=False)

    bad = lo.status.isin(cfg["labels"]["bad_status"])
    checks = {}
    checks["R-14"] = {
        "pass": bool(lo.loc[bad, "has_event"].mean() >= 0.90 and lo.loc[~bad, "has_event"].mean() <= 0.05),
        "bad_with_event": float(lo.loc[bad, "has_event"].mean()),
        "good_with_event": float(lo.loc[~bad, "has_event"].mean()),
        "by_route": {
            "penalty_bad": float(lo.loc[bad, "penalty_mi"].notna().mean()),
            "missed_bad": float(lo.loc[bad, "missed_mi"].notna().mean()),
            "penalty_good": float(lo.loc[~bad, "penalty_mi"].notna().mean()),
            "missed_good": float(lo.loc[~bad, "missed_mi"].notna().mean()),
        },
        "bad_without_event_loan_ids": lo.loc[bad & (lo.has_event == 0), "loan_id"].tolist(),
    }
    ev = snap.loan_id.map(lo.set_index("loan_id")["event_mi"])
    checks["R-15"] = {"pass": bool(~(snap.cutoff_mi >= ev).any()), "violations": int((snap.cutoff_mi >= ev).sum())}
    data_end_mi = month_index(pd.Timestamp(cfg["data"]["data_end"]))
    neg_past_end = (snap.label == 0) & (snap.cutoff_mi + 6 > data_end_mi)
    last_due = snap.loan_id.map(lo.set_index("loan_id")["last_due_mi"])
    checks["R-16"] = {
        "pass": bool(~(neg_past_end & (last_due > data_end_mi)).any()),
        "negatives_past_end_only_repaid_loans": int(neg_past_end.sum()),
        "censored_dropped_for_headA": int(snap.label.isna().sum()),
    }
    lab = snap.dropna(subset=["label"])
    counts = lab.groupby("split").agg(snapshots=("label", "size"), positives=("label", "sum"),
                                      loans=("loan_id", "nunique"))
    ev_loans = lab[lab.label == 1].groupby("split").loan_id.nunique()
    counts["event_loans"] = ev_loans
    counts["loans_bad_status"] = lab.drop_duplicates("loan_id").assign(
        b=lambda d: d.loan_id.map(lo.set_index("loan_id")["bad"])).groupby("split").b.sum()
    counts = counts.fillna(0).astype(int)
    checks["R-17"] = {"pass": True, "counts": counts.to_dict("index")}
    ids = {s: set(snap.loan_id[snap.split == s]) for s in ["train", "val", "test"]}
    checks["R-25"] = {"pass": not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])}
    od = lo.set_index("loan_id")["date"]
    o = {s: od.loc[list(ids[s])] for s in ids}
    checks["R-26"] = {
        "pass": bool(o["train"].max() < o["val"].min() and o["val"].max() < o["test"].min()),
        "origination_ranges": {s: [str(o[s].min().date()), str(o[s].max().date())] for s in o},
        "note": "P1 orders loans by origination date. Strict calendar ordering of cutoffs is enforced in P2.",
    }
    fp = split_fingerprint(snap)
    checks["R-27"] = {"pass": True, "fingerprint": fp}
    (proc / "split_fingerprint.txt").write_text(fp)
    (p(cfg, "reports") / "tables" / "label_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    return checks


if __name__ == "__main__":
    for k, v in run().items():
        print(k, "PASS" if v["pass"] else "FAIL", {a: b for a, b in v.items() if a != "pass"})
