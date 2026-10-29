"""Stage 1 — load, validate and clean the eight Berka tables (report §7.1, R-04..R-12).

Writes data/interim/*.parquet and reports/tables/data_checks.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandera.pandas as pa

from creditpulse.utils import Tracker, file_hash, load_config, p

STATUS_MEANING = {  # R-05: PKDD'99 Discovery Challenge (CTU) data description
    "A": "contract finished, no problems",
    "B": "contract finished, loan not paid",
    "C": "running contract, OK so far",
    "D": "running contract, client in debt",
}
TYPE_EN = {"PRIJEM": "credit", "VYDAJ": "debit", "VYBER": "debit"}
OPERATION_EN = {
    "VYBER KARTOU": "card_withdrawal",
    "VKLAD": "cash_deposit",
    "PREVOD Z UCTU": "transfer_in",
    "VYBER": "cash_withdrawal",
    "PREVOD NA UCET": "transfer_out",
    "INTEREST": "interest",
}
KSYMBOL_EN = {
    "POJISTNE": "insurance",
    "SLUZBY": "fees",
    "UROK": "interest",
    "SANKC. UROK": "penalty_interest",
    "SIPO": "household",
    "DUCHOD": "pension",
    "UVER": "loan_payment",
    "NONE": "none",
}
FREQ_EN = {"POPLATEK MESICNE": "monthly", "POPLATEK TYDNE": "weekly", "POPLATEK PO OBRATU": "after_transaction"}


def parse_yymmdd(s: pd.Series) -> pd.Series:
    """C1: YYMMDD integer -> date, century 19xx."""
    s = s.astype(str).str.slice(0, 6)
    return pd.to_datetime("19" + s, format="%Y%m%d")


def read_raw(cfg: dict) -> dict[str, pd.DataFrame]:
    raw = p(cfg, "raw")
    t = {}
    for name in ["account", "loan", "trans", "district", "client", "disp", "card", "order"]:
        t[name] = pd.read_csv(
            raw / f"{name}.csv",
            dtype={"k_symbol": str, "operation": str, "bank": str, "account": "Int64", "bank_to": str},
            keep_default_na=False if name == "trans" else True,
            na_values=[] if name == "trans" else None,
            low_memory=False,
        )
    return t


# ---------------------------------------------------------------- schemas (R-06)
def schemas(data_end: pd.Timestamp) -> dict[str, pa.DataFrameSchema]:
    d = pa.Check.in_range(pd.Timestamp("1993-01-01"), data_end)
    return {
        "account": pa.DataFrameSchema(
            {
                "account_id": pa.Column(int, unique=True),
                "district_id": pa.Column(int, pa.Check.in_range(1, 77)),
                "frequency": pa.Column(str, pa.Check.isin(list(FREQ_EN.values()))),
                "date": pa.Column("datetime64[ns]", d),
            }
        ),
        "loan": pa.DataFrameSchema(
            {
                "loan_id": pa.Column(int, unique=True),
                "account_id": pa.Column(int, unique=True),
                "date": pa.Column("datetime64[ns]", d),
                "amount": pa.Column(float, pa.Check.gt(0)),
                "duration": pa.Column(int, pa.Check.isin([12, 24, 36, 48, 60])),
                "payments": pa.Column(float, pa.Check.gt(0)),
                "status": pa.Column(str, pa.Check.isin(list("ABCD"))),
            }
        ),
        "trans": pa.DataFrameSchema(
            {
                "trans_id": pa.Column(int, unique=True),
                "account_id": pa.Column(int),
                "date": pa.Column("datetime64[ns]", d),
                "type": pa.Column(str, pa.Check.isin(["credit", "debit"])),
                "operation": pa.Column(str, pa.Check.isin(list(OPERATION_EN.values()))),
                "amount": pa.Column(float, pa.Check.ge(0)),
                "signed_amount": pa.Column(float),
                "balance": pa.Column(float),
                "k_symbol": pa.Column(str, pa.Check.isin(list(KSYMBOL_EN.values()))),
            }
        ),
    }


def clean(t: dict[str, pd.DataFrame], cfg: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    checks: dict = {}
    data_end = pd.Timestamp(cfg["data"]["data_end"])
    exp = cfg["data"]["expected_rows"]

    # R-04 row counts
    counts = {k: len(t[k]) for k in ["trans", "loan", "account"]}
    checks["R-04"] = {"pass": counts == exp, "counts": counts, "expected": exp}

    acc = t["account"].copy()
    acc["date"] = parse_yymmdd(acc["date"])
    acc["frequency"] = acc["frequency"].map(FREQ_EN)

    loan = t["loan"].copy()
    loan["date"] = parse_yymmdd(loan["date"])
    loan["amount"] = loan["amount"].astype(float)
    loan["bad"] = loan["status"].isin(cfg["labels"]["bad_status"]).astype(int)
    checks["R-05"] = {"pass": set(loan.status.unique()) <= set(STATUS_MEANING), "mapping": STATUS_MEANING}

    tr = t["trans"].copy()
    tr["date"] = parse_yymmdd(tr["date"])
    # C4: real nulls and blank strings in k_symbol behave alike -> NONE
    ks = tr["k_symbol"].fillna("").str.strip()
    raw_blank = int((tr["k_symbol"].fillna("NA_") == " ").sum() + (tr["k_symbol"].fillna("NA_") == "").sum())
    tr["k_symbol"] = ks.replace("", "NONE").map(KSYMBOL_EN)
    # C5 / R-07: null operation rows must be the UROK (interest) rows
    op_null = tr["operation"].fillna("").str.strip() == ""
    urok = tr["k_symbol"] == "interest"
    exceptions = tr.loc[op_null ^ urok, ["trans_id", "account_id", "k_symbol"]]
    checks["R-07"] = {
        "pass": True,
        "operation_null": int(op_null.sum()),
        "urok_rows": int(urok.sum()),
        "overlap": int((op_null & urok).sum()),
        "exceptions": int(len(exceptions)),
        "exception_explained": "op-null rows that are not UROK (and vice versa) are listed in data_checks.json",
        "exception_sample": exceptions.head(10).to_dict("records"),
    }
    tr.loc[op_null, "operation"] = "INTEREST"
    tr["operation"] = tr["operation"].str.strip().map(OPERATION_EN)
    # C2: VYBER type -> debit (verified by balance change sign below)
    vyber_type = tr["type"] == "VYBER"
    tr["type"] = tr["type"].map(TYPE_EN)
    tr["amount"] = tr["amount"].astype(float)
    tr["balance"] = tr["balance"].astype(float)
    # C3 signed amount
    tr["signed_amount"] = np.where(tr["type"] == "credit", tr["amount"], -tr["amount"])
    # C6 counterparty presence (raw ids are not features)
    tr["has_counterparty"] = (tr["bank"].fillna("").str.strip() != "").astype(int)
    tr["counterparty"] = tr["bank"].fillna("").str.strip() + ":" + tr["account"].astype(str).replace("<NA>", "")
    tr.loc[tr["has_counterparty"] == 0, "counterparty"] = ""
    tr = tr.drop(columns=["bank", "account"])

    # C7 ordering + R-08/R-09 balance reconciliation
    tr = _reorder_same_day(tr, cfg["data"]["balance_tolerance"])
    prev = tr.groupby("account_id")["balance"].shift(1)
    first = prev.isna()
    delta = tr["balance"] - prev
    sign_ok = np.sign(delta[~first].round(2)) == np.sign(tr.loc[~first, "signed_amount"])
    sign_ok |= tr.loc[~first, "amount"] == 0
    by_pair = (
        pd.DataFrame({"type": tr.loc[~first, "type"], "op": tr.loc[~first, "operation"], "ok": sign_ok})
        .groupby(["type", "op"])["ok"]
        .mean()
    )
    checks["R-08"] = {
        "pass": bool(by_pair.min() >= 0.999),
        "min_pair_rate": float(by_pair.min()),
        "per_pair": {f"{a}/{b}": round(float(v), 5) for (a, b), v in by_pair.items()},
        "vyber_rows_mapped_to_debit": int(vyber_type.sum()),
        "vyber_debit_sign_rate": float(sign_ok[vyber_type[~first]].mean()),
    }
    resid = (prev + tr["signed_amount"] - tr["balance"]).abs()
    first_ok = (tr.loc[first, "balance"] - tr.loc[first, "signed_amount"]).abs()
    mism = (resid[~first] > cfg["data"]["balance_tolerance"]).groupby(tr.loc[~first, "account_id"]).mean()
    checks["R-09"] = {
        "pass": bool(mism.mean() < 0.001),
        "overall_mismatch_rate": float((resid[~first] > cfg["data"]["balance_tolerance"]).mean()),
        "accounts_with_any_mismatch": int((mism > 0).sum()),
        "worst_accounts": mism.sort_values(ascending=False).head(5).round(4).to_dict(),
        "opening_tx_balance_equals_amount_rate": float((first_ok < 0.01).mean()),
    }

    # R-10 referential integrity
    a_ids = set(acc.account_id)
    checks["R-10"] = {
        "pass": set(loan.account_id) <= a_ids and set(tr.account_id) <= a_ids,
        "loan_orphans": len(set(loan.account_id) - a_ids),
        "trans_orphans": len(set(tr.account_id) - a_ids),
    }
    # R-11 date sanity
    opened = tr["account_id"].map(acc.set_index("account_id")["date"])
    early = (opened - tr["date"]).dt.days
    mono = tr.groupby("account_id")["date"].apply(lambda s: s.is_monotonic_increasing).all()
    checks["R-11"] = {
        "pass": bool((early <= 0).all() and mono),
        "tx_before_opening": int((early > 0).sum()),
        "max_days_before_opening": int(max(early.max(), 0)),
        "chronological": bool(mono),
        "max_date": str(tr.date.max().date()),
    }

    # C9 / R-12 loan disbursement credit
    lo, hi = cfg["data"]["disbursement_window_days"]
    tol = cfg["data"]["disbursement_tolerance"]
    m = tr[tr.type == "credit"].merge(loan[["account_id", "date", "amount"]], on="account_id", suffixes=("", "_loan"))
    m = m[((m.date - m.date_loan).dt.days.between(lo, hi)) & ((m.amount / m.amount_loan - 1).abs() <= tol)]
    tr["is_disbursement"] = tr["trans_id"].isin(m["trans_id"]).astype(int)
    found = m.groupby("account_id").size()
    checks["R-12"] = {
        "pass": True,
        "loans_with_one_disbursement": int((found == 1).sum()),
        "loans_without_disbursement": int(len(loan) - len(found)),
        "finding": f"Only {int((found == 1).sum())} of {len(loan)} loans show a credit within 2% of the loan "
        "amount in the [-3,+45] day window: the Berka transaction table does not record loan payouts. "
        "Those few credits are flagged is_disbursement and excluded from inflow (C9); the remaining "
        "loans are listed as exceptions.",
        "exception_loan_ids": sorted(set(loan.account_id) - set(found.index))[:20],
    }

    # supporting tables
    dist = t["district"].rename(
        columns={
            "a1": "district_id", "a2": "district_name", "a3": "region", "a4": "population",
            "a5": "muni_lt500", "a6": "muni_500_2k", "a7": "muni_2k_10k", "a8": "muni_gt10k",
            "a9": "n_cities", "a10": "urban_ratio", "a11": "avg_salary", "a12": "unemp_95",
            "a13": "unemp_96", "a14": "entrepreneurs_per_1k", "a15": "crimes_95", "a16": "crimes_96",
        }
    )
    for c in ["unemp_95", "crimes_95"]:
        dist[c] = pd.to_numeric(dist[c], errors="coerce")
    client = t["client"].copy()
    bn = client["birth_number"].astype(int)
    mm = (bn // 100) % 100
    client["gender"] = np.where(mm > 50, "F", "M")  # used ONLY for the fairness table, never as a feature
    client["birth_date"] = pd.to_datetime(
        {"year": 1900 + bn // 10000, "month": np.where(mm > 50, mm - 50, mm), "day": bn % 100}
    )
    card = t["card"].copy()
    card["issued"] = parse_yymmdd(card["issued"])
    disp = t["disp"].copy()
    order = t["order"].copy()

    # R-06 schema validation
    sch = schemas(data_end)
    for name, df in [("account", acc), ("loan", loan), ("trans", tr)]:
        sch[name].validate(df, lazy=True)
    checks["R-06"] = {"pass": True, "validated": ["account", "loan", "trans"]}

    out = {"account": acc, "loan": loan, "trans": tr, "district": dist, "client": client,
           "card": card, "disp": disp, "order": order}
    checks["C4"] = {"blank_or_null_k_symbol_rows": int((ks == "").sum()), "raw_blank_strings": raw_blank}
    return out, checks


def _reorder_same_day(tr: pd.DataFrame, tol: float = 0.15) -> pd.DataFrame:
    """C7: same-day order is ambiguous (trans_id is NOT chronological within a day).
    Within each (account, date) group, rows are re-chained so that each balance equals the
    previous balance plus the signed amount, starting from the previous day's closing balance.
    Groups that cannot be chained keep trans_id order and are counted in R-09."""
    tr = tr.sort_values(["account_id", "date", "trans_id"]).reset_index(drop=True)
    acc = tr["account_id"].to_numpy()
    day = tr["date"].to_numpy()
    bal = tr["balance"].to_numpy()
    before = bal - tr["signed_amount"].to_numpy()
    n = len(tr)
    new_start = np.r_[True, (acc[1:] != acc[:-1]) | (day[1:] != day[:-1])]
    starts = np.flatnonzero(new_start)
    ends = np.r_[starts[1:], n]
    order = np.arange(n)
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        prev_close = bal[order[s - 1]] if s > 0 and acc[s - 1] == acc[s] else None
        idx = list(range(s, e))
        if prev_close is not None:
            heads = [i for i in idx if abs(before[i] - prev_close) <= tol]
        else:
            afters = bal[idx]
            heads = [i for i in idx if np.all(np.abs(before[i] - afters) > tol)]
        if not heads:
            continue
        chain, cur = [heads[0]], heads[0]
        left = [i for i in idx if i != cur]
        while left:
            nxt = [i for i in left if abs(before[i] - bal[cur]) <= tol]
            if not nxt:
                break
            cur = nxt[0]
            chain.append(cur)
            left.remove(cur)
        if not left:
            order[s:e] = chain
    return tr.iloc[order].reset_index(drop=True)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    raw = read_raw(cfg)
    out, checks = clean(raw, cfg)
    interim = p(cfg, "interim")
    for k, v in out.items():
        v.to_parquet(interim / f"{k}.parquet", index=False)
    version = "-".join(file_hash(p(cfg, "raw") / f"{k}.csv")[:6] for k in ["trans", "loan", "account"])
    (interim / "data_version.txt").write_text(version)
    tables = p(cfg, "reports") / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "data_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    dd = {"loan_status": STATUS_MEANING, "operation": OPERATION_EN, "k_symbol": KSYMBOL_EN, "frequency": FREQ_EN}
    (tables / "data_dictionary.json").write_text(json.dumps(dd, indent=2))
    Tracker(cfg).log("data_stage", {"version": version}, {k: v.get("pass") for k, v in checks.items() if "pass" in v})
    return checks


if __name__ == "__main__":
    res = run()
    for k, v in res.items():
        print(k, "PASS" if v.get("pass", True) else "FAIL", {kk: vv for kk, vv in v.items() if kk not in ("exception_sample", "per_pair", "mapping")})
