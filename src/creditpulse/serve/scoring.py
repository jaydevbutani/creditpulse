"""Scoring service core (report §11). One code path for the single-loan API and the monthly batch job, so
train/serve parity (R-52) holds by construction: panel -> PanelIndex.gather -> Preprocessor -> M6 ensemble
-> Platt calibration -> queue rank -> integrated-gradient reason codes."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from creditpulse.explain.attrib import REASONS, reason_text
from creditpulse.features.panel import FEATURE_GROUPS, FEATURES
from creditpulse.features.tokens import PanelIndex, Preprocessor
from creditpulse.labels.events import mi_to_date, month_index
from creditpulse.models.deep import SeqModel
from creditpulse.utils import load_config, p

torch.set_num_threads(2)


# Reason codes describe behaviour (report §10): structural, calendar, mask and account-static inputs still feed
# the model but are not offered to analysts as reasons.
REASON_ELIGIBLE = {f for g in ["cash_flow", "balance", "activity", "channel_mix", "obligations", "loan_servicing",
                               "distress", "dynamics"] for f in FEATURE_GROUPS[g]} - {"due_flag", "hist_short", "uver_flag"}
REASON_ELIGIBLE |= {"loan_payment", "loan_amount", "payment_to_amount"}


def scale_hazard(h: np.ndarray, target: np.ndarray, iters: int = 40) -> np.ndarray:
    """Rescale Head-B hazards by a per-loan factor so that 1 - prod(1 - c*h) equals the calibrated Head-A risk:
    the timing *shape* comes from Head B, the *level* from the calibrated Head A."""
    lo = np.zeros(len(h))
    hi = 1.0 / np.clip(h.max(1), 1e-9, None)
    for _ in range(iters):
        mid = (lo + hi) / 2
        F = 1 - np.prod(1 - mid[:, None] * h, 1)
        big = F > target
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    return ((lo + hi) / 2)[:, None] * h


class ScoringError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class LoanState:
    status: str          # scored | in_distress | closed | not_originated
    detail: str


class Scorer:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        art, proc = p(self.cfg, "artifacts"), p(self.cfg, "processed")
        self.serving = json.loads((art / "serving.json").read_text())
        self.panel = pd.read_parquet(proc / "panel.parquet")
        self.index = PanelIndex(self.panel)
        self.pre = Preprocessor.from_dict(json.loads((art / "preprocessor.json").read_text()))
        lo = pd.read_parquet(proc / "loans_events.parquet")
        self.loans = lo.set_index("loan_id")
        self.W = self.cfg["labels"]["window"]
        self.min_age = self.cfg["labels"]["min_loan_age"]
        self.data_end = pd.Timestamp(self.cfg["data"]["data_end"])
        self.data_end_mi = month_index(self.data_end)
        self.models = []
        for sd in self.serving["seeds"]:
            ck = torch.load(art / "models" / f"{self.serving['model']}_s{sd}.pt", weights_only=False)
            hp, arch = ck["hp"], ck["arch"]
            m = SeqModel(len(ck["cols"]), kind=arch["kind"], d_model=hp["d_model"], n_layers=hp["n_layers"],
                         n_heads=hp.get("n_heads", 4), dropout=hp["dropout"], pooling=hp.get("pooling", "cls"),
                         head_b=arch["head_b"], derive_a=hp.get("derive_a", False))
            m.load_state_dict(ck["state"])
            m.eval()
            self.models.append(m)
        self.version = self.serving["model_version"]
        self._month_cache: dict[int, pd.DataFrame] = {}

    # ------------------------------------------------------------------ helpers
    def cutoff_for(self, as_of) -> int:
        d = pd.Timestamp(as_of)
        if d > self.data_end:
            raise ScoringError(422, f"as_of_date {d.date()} is after the last available data ({self.data_end.date()}).")
        if d < pd.Timestamp("1993-01-31"):
            raise ScoringError(422, f"as_of_date {d.date()} is before the first available month-end (1993-01-31).")
        mi = month_index(d)
        return mi if d.is_month_end else mi - 1  # last complete month-end on or before as_of

    def state(self, loan_id: int, c: int) -> LoanState:
        r = self.loans.loc[loan_id]
        if c < r.start_mi:
            return LoanState("not_originated", f"loan originates {r.date.date()}")
        if c - r.start_mi < self.min_age:
            return LoanState("too_new", f"monitoring starts {self.min_age} months after origination (D1); "
                                        "use the origination score (Task O) until then")
        if not np.isnan(r.event_mi) and r.event_mi <= c:
            return LoanState("in_distress", f"distress event already observed in {mi_to_date(r.event_mi).strftime('%Y-%m')}")
        if c >= r.last_due_mi:
            return LoanState("closed", "final instalment month reached")
        return LoanState("scored", "at risk")

    def at_risk(self, c: int) -> pd.DataFrame:
        lo = self.loans
        ev = lo.event_mi.fillna(np.inf)
        m = (lo.start_mi + self.min_age <= c) & (c < lo.last_due_mi) & (ev > c)  # same at-risk rule as training
        return lo[m].reset_index()[["loan_id", "account_id", "start_mi", "date"]]

    def tokens(self, account_ids, loan_ages, c: int):
        X, pad = self.index.gather(np.asarray(account_ids), np.full(len(account_ids), c), self.W)
        Z = self.pre.transform(X, pad)
        age = np.asarray(loan_ages)[:, None] - (self.W - 1 - np.arange(self.W))[None, :]
        return X, torch.tensor(Z), torch.tensor(pad), torch.tensor(age, dtype=torch.float32)

    @torch.no_grad()
    def _forward(self, Z, pad, age):
        probs, hz = [], []
        for m in self.models:
            o = m(Z, pad, age)
            probs.append(torch.sigmoid(o["logit_a"]).numpy())
            hz.append(torch.sigmoid(o["hz_logits"]).numpy())
        raw = np.mean(probs, 0)
        a, b = self.serving["platt"]["a"], self.serving["platt"]["b"]
        z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
        return raw, 1 / (1 + np.exp(-(a * z + b))), np.mean(hz, 0)

    def drivers(self, Z, pad, age, X, k=3, steps=16):
        """Top-k risk-increasing *behaviour* features by integrated gradients (midpoint rule, baseline = training
        mean), summed over the window and averaged over the seed ensemble."""
        n = len(Z)
        alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
        Zs = (alphas.view(-1, 1, 1, 1) * Z.unsqueeze(0)).reshape(steps * n, *Z.shape[1:])
        P, A = pad.repeat(steps, 1), age.repeat(steps, 1)
        total = torch.zeros_like(Z)
        for m in self.models:
            xi = Zs.clone().requires_grad_(True)
            g, = torch.autograd.grad(m(xi, P, A)["logit_a"].sum(), xi)
            total += g.reshape(steps, n, *Z.shape[1:]).sum(0)
        attr = (Z * total / (steps * len(self.models))).sum(1).numpy()  # [N, F]
        out = []
        for i in range(len(attr)):
            order = np.argsort(-attr[i])
            picks = []
            for j in order:
                f = FEATURES[j]
                if attr[i, j] <= 0 or len(picks) == k:
                    break
                if f not in REASON_ELIGIBLE:
                    continue
                picks.append({"feature": f, "contribution": round(float(attr[i, j]), 4),
                              "value": float(X[i, -1, j]), "text": reason_text(f, float(X[i, -1, j]))})
            out.append(picks)
        return out

    # ------------------------------------------------------------------ public API
    def score_month(self, c: int, with_drivers: bool = False) -> pd.DataFrame:
        if c in self._month_cache and (not with_drivers or "drivers" in self._month_cache[c]):
            return self._month_cache[c]
        q = self.at_risk(c)
        if q.empty:
            return q
        ages = c - q.start_mi.values
        X, Z, pad, age = self.tokens(q.account_id.values, ages, c)
        raw, cal, hz = self._forward(Z, pad, age)
        q["cutoff_mi"] = c
        q["loan_age"] = ages
        q["risk_raw"] = raw
        q["risk_6m"] = cal
        hs = scale_hazard(hz, cal)
        for k in range(hz.shape[1]):
            q[f"h{k + 1}"] = hz[:, k]
            q[f"hs{k + 1}"] = hs[:, k]
        q["queue_rank"] = q.risk_6m.rank(ascending=False, method="first").astype(int)
        q["queue_percentile"] = 100.0 * (q.queue_rank - 1) / len(q)
        q["in_training_domain"] = ages >= self.min_age
        if with_drivers:
            q["drivers"] = [json.dumps(d) for d in self.drivers(Z, pad, age, X, steps=8)]
        self._month_cache[c] = q
        return q

    def score(self, loan_id: int, as_of_date) -> dict:
        if loan_id not in self.loans.index:
            raise ScoringError(404, f"unknown loan_id {loan_id}")
        c = self.cutoff_for(as_of_date)
        st = self.state(loan_id, c)
        if st.status == "not_originated":
            raise ScoringError(422, f"loan {loan_id} is not yet originated at {as_of_date} ({st.detail}).")
        base = {"loan_id": int(loan_id), "as_of_date": str(pd.Timestamp(as_of_date).date()),
                "cutoff": mi_to_date(c).strftime("%Y-%m-%d"), "status": st.status, "detail": st.detail,
                "model_version": self.version,
                "window": {"first_month": mi_to_date(c - self.W + 1).strftime("%Y-%m"),
                           "last_month": mi_to_date(c).strftime("%Y-%m")}}
        if st.status != "scored":
            return {**base, "risk_6m": None, "hazard": None, "queue_rank": None, "queue_percentile": None,
                    "queue_size": None, "drivers": []}
        q = self.score_month(c)
        row = q[q.loan_id == loan_id].iloc[0]
        r = self.loans.loc[loan_id]
        X, Z, pad, age = self.tokens([r.account_id], [c - r.start_mi], c)
        drv = self.drivers(Z, pad, age, X)[0]
        return {**base, "risk_6m": float(row.risk_6m), "risk_raw": float(row.risk_raw),
                "hazard": [float(row[f"h{k}"]) for k in range(1, 7)],
                "hazard_scaled": [float(row[f"hs{k}"]) for k in range(1, 7)],
                "cumulative_incidence": [float(v) for v in 1 - np.cumprod(1 - row[[f"hs{k}" for k in range(1, 7)]].values.astype(float))],
                "queue_rank": int(row.queue_rank), "queue_percentile": float(row.queue_percentile),
                "queue_size": int(len(q)), "in_top_10pct": bool(row.risk_6m >= self.serving["threshold_10pct_queue"]),
                "in_training_domain": bool(row.in_training_domain), "drivers": drv}

    def queue(self, as_of_date, top: int = 50) -> list[dict]:
        c = self.cutoff_for(as_of_date)
        q = self.score_month(c).sort_values("queue_rank").head(top)
        return [{"loan_id": int(r.loan_id), "queue_rank": int(r.queue_rank), "risk_6m": float(r.risk_6m),
                 "queue_percentile": float(r.queue_percentile), "loan_age": int(r.loan_age)} for r in q.itertuples()]


def batch_run(cfg=None, start="1995-01-31"):
    """Monthly batch job: score every at-risk loan for every month-end and write the results table."""
    cfg = cfg or load_config()
    sc = Scorer(cfg)
    out, timing = [], []
    for c in range(month_index(pd.Timestamp(start)), sc.data_end_mi + 1):
        t0 = time.time()
        q = sc.score_month(c, with_drivers=True)
        timing.append({"cutoff": mi_to_date(c).strftime("%Y-%m"), "loans": len(q), "seconds": time.time() - t0})
        if len(q):
            out.append(q)
    res = pd.concat(out, ignore_index=True)
    res["cutoff_date"] = res.cutoff_mi.map(mi_to_date)
    d = p(cfg, "artifacts") / "scores"
    d.mkdir(exist_ok=True)
    res.to_parquet(d / "monthly_scores.parquet", index=False)
    tm = pd.DataFrame(timing)
    tm.to_csv(p(cfg, "reports") / "tables" / "batch_timing.csv", index=False)
    return res, tm


if __name__ == "__main__":
    r, t = batch_run()
    print(len(r), "scored loan-months; max seconds per month", t.seconds.max())
