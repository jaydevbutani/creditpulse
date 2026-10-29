"""Stage 7 — sensitivity runs (report §9.4, R-46) and the rolling-origin protocol P2 (D5).

Each variant rebuilds labels/snapshots/tokens from the config, then refits M2 (LightGBM, frozen tuned
params, 5 seeds) and M6 (frozen tuned hyper-parameters, 3 seeds). No re-tuning: the variants test the
design, not a new search."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import torch

from creditpulse.eval.metrics import ci, cluster_bootstrap, pr_auc, roc_auc
from creditpulse.features.tokens import PanelIndex, Preprocessor, aggregate
from creditpulse.labels.events import assign_p1, build_snapshots, event_dates, month_index, p2_folds
from creditpulse.models.baselines import make_lgbm
from creditpulse.train.data import Bundle, token_ages
from creditpulse.train.deep import MODELS, predict, train_one
from creditpulse.utils import Tracker, load_config, p, seed_everything


def make_bundle(cfg, loan, trans, index, pre=None, horizon=None, window=None, split_fn=None):
    lo = event_dates(loan, trans, cfg)
    snap = build_snapshots(lo, cfg, horizon)
    snap["split"] = assign_p1(snap, lo, cfg) if split_fn is None else split_fn(snap)
    snap["event_mi"] = snap.loan_id.map(lo.set_index("loan_id")["event_mi"])
    snap["orig_date"] = snap.loan_id.map(lo.set_index("loan_id")["date"])
    W = window or cfg["labels"]["window"]
    X, pad = index.gather(snap.account_id.values, snap.cutoff_mi.values, W)
    if pre is None:
        tr = (snap.split == "train").values
        pre = Preprocessor().fit(X[tr], pad[tr], cfg["features"]["winsor_q"])
    Z = pre.transform(X, pad)
    A, acols = aggregate(Z, pad)
    return Bundle(snap=snap, Z=Z, pad=pad, A=A, acols=acols, age=token_ages(snap, W), y=snap.label.values.astype(float),
                  event_k=snap.event_k.values, follow_k=snap.follow_k.values)


def fit_m2(cfg, B, params, seeds, tr, te):
    s = np.mean([make_lgbm(params, sd).fit(B.A[tr], B.y[tr]).predict_proba(B.A[te])[:, 1] for sd in seeds], 0)
    return s


def fit_m6(cfg, B, m6, seeds, tr_all, te, horizon=6):
    T = (torch.tensor(B.Z), torch.tensor(B.pad), torch.tensor(B.age, dtype=torch.float32),
         torch.tensor(np.nan_to_num(B.y, nan=-1.0), dtype=torch.float32),
         torch.tensor(B.event_k, dtype=torch.long), torch.tensor(B.follow_k, dtype=torch.long))
    arch = {**MODELS["M6_multitask"], "horizon": horizon}
    y = T[3].numpy()
    out = []
    for sd in seeds:
        m, _ = train_one(T, y, arch, m6["hp"], sd, tr_all, None, epochs=m6["epochs"], cfg=cfg)
        out.append(predict(m, T, te)[0])
    return np.mean(out, 0)


def summarise(name, model, B, te, s, nb):
    d = B.snap.iloc[te][["loan_id"]].assign(y=B.y[te], score=s)
    lo, hi = ci(cluster_bootstrap(d, lambda x: pr_auc(x.y.values, x.score.values), nb, 0))
    return {"variant": name, "model": model, "test_pr_auc": pr_auc(d.y.values, s), "pr_lo": lo, "pr_hi": hi,
            "test_roc_auc": roc_auc(d.y.values, s), "snapshots": len(d), "positives": int(d.y.sum()),
            "event_loans": int(d[d.y == 1].loan_id.nunique()), "prevalence": float(d.y.mean())}


def run(cfg=None):
    base_cfg = cfg or load_config()
    tracker = Tracker(base_cfg)
    interim, proc, art = p(base_cfg, "interim"), p(base_cfg, "processed"), p(base_cfg, "artifacts")
    loan = pd.read_parquet(interim / "loan.parquet")
    trans = pd.read_parquet(interim / "trans.parquet")
    index = PanelIndex(pd.read_parquet(proc / "panel.parquet"))
    m2_params = json.loads((art / "M2_lgbm_params.json").read_text())
    m6 = json.loads((art / "hparams" / "M6_multitask.json").read_text())
    nb = 1000
    m2_seeds, m6_seeds = base_cfg["train"]["seeds"], [0, 1, 2]

    variants = {
        "baseline (H=6, W=12, both routes, min age 3)": {},
        "horizon 3": {"horizon": 3},
        "horizon 12": {"horizon": 12},
        "window 6": {"window": 6},
        "window 18": {"window": 18},
        "event route (a) penalty only": {"labels.event_routes": ["penalty"]},
        "event route (b) missed instalment only": {"labels.event_routes": ["missed"]},
        "min loan age 1": {"labels.min_loan_age": 1},
        "no class weighting": {"no_weight": True},
    }
    rows = []
    for name, v in variants.items():
        cfg = copy.deepcopy(base_cfg)
        for k, val in v.items():
            if "." in k:
                a, b = k.split(".")
                cfg[a][b] = val
        H = v.get("horizon", 6)
        B = make_bundle(cfg, loan, trans, index, horizon=H, window=v.get("window"))
        lab = ~np.isnan(B.y)
        tr = np.flatnonzero((B.snap.split == "train").values & lab)
        tr_all = np.flatnonzero((B.snap.split == "train").values)
        te = np.flatnonzero((B.snap.split == "test").values & lab)
        prm = {**m2_params, "spw": 1.0} if v.get("no_weight") else m2_params
        hp6 = {**m6, "hp": {**m6["hp"], "pos_weight": 1.0}} if v.get("no_weight") else m6
        seed_everything(0)
        rows.append(summarise(name, "M2_lgbm", B, te, fit_m2(cfg, B, prm, m2_seeds, tr, te), nb))
        rows.append(summarise(name, "M6_multitask", B, te, fit_m6(cfg, B, hp6, m6_seeds, tr_all, te, H), nb))
        print(rows[-2], rows[-1], sep="\n", flush=True)

    # ---------------- P2 rolling origin
    lo = event_dates(loan, trans, base_cfg)
    snap0 = build_snapshots(lo, base_cfg)
    p2_rows, pooled = [], {"M2_lgbm": [], "M6_multitask": []}
    t4 = []
    for f in p2_folds(snap0.assign(split="x"), base_cfg):
        tr_set, te_set = set(f["train"]), set(f["test"])

        def split_fn(s, tr_set=tr_set, te_set=te_set):
            return pd.Series(np.where(s.index.isin(list(tr_set)), "train",
                                      np.where(s.index.isin(list(te_set)), "test", "unused")), index=s.index)

        B = make_bundle(base_cfg, loan, trans, index, split_fn=split_fn)  # preprocessor refit on fold train (T5)
        tr = np.flatnonzero((B.snap.split == "train").values)
        te = np.flatnonzero((B.snap.split == "test").values)
        bm = month_index(pd.Timestamp(f["boundary"]))
        t4.append({"boundary": f["boundary"], "max_train_label_end": int(B.snap.cutoff_mi.iloc[tr].max() + 6),
                   "boundary_mi": bm, "min_test_cutoff": int(B.snap.cutoff_mi.iloc[te].min()),
                   "ordered": bool(B.snap.cutoff_mi.iloc[tr].max() + 6 <= bm < B.snap.cutoff_mi.iloc[te].min())})
        s2 = fit_m2(base_cfg, B, m2_params, m2_seeds, tr, te)
        tr_all = tr  # P2 train already contains only fully observed label windows
        s6 = fit_m6(base_cfg, B, m6, m6_seeds, tr_all, te)
        for model, s in [("M2_lgbm", s2), ("M6_multitask", s6)]:
            r = summarise(f"P2 fold {f['boundary']}", model, B, te, s, nb)
            p2_rows.append(r)
            pooled[model].append(B.snap.iloc[te][["loan_id"]].assign(y=B.y[te], score=s,
                                                                      fold=f["boundary"]))
            print(r, flush=True)
    for model, parts in pooled.items():
        d = pd.concat(parts)
        d["loan_id"] = d.loan_id.astype(str) + "_" + d.fold  # loans can recur across folds; cluster within fold
        lo_, hi_ = ci(cluster_bootstrap(d, lambda x: pr_auc(x.y.values, x.score.values), nb, 0))
        p2_rows.append({"variant": "P2 pooled (3 folds)", "model": model, "test_pr_auc": pr_auc(d.y.values, d.score.values),
                        "pr_lo": lo_, "pr_hi": hi_, "test_roc_auc": roc_auc(d.y.values, d.score.values),
                        "snapshots": len(d), "positives": int(d.y.sum()),
                        "event_loans": int(d[d.y == 1].loan_id.nunique()), "prevalence": float(d.y.mean())})
    out = pd.DataFrame(rows + p2_rows)
    out.to_csv(p(base_cfg, "reports") / "tables" / "sensitivity.csv", index=False)
    chk = {"R-46": {"pass": True, "variants": len(variants), "p2_folds": 3},
           "T4_time_order_P2": {"pass": all(x["ordered"] for x in t4), "folds": t4}}
    (p(base_cfg, "reports") / "tables" / "sensitivity_checks.json").write_text(json.dumps(chk, indent=2))
    tracker.log("sensitivity", {"variants": list(variants)}, {"rows": len(out)})
    return out, chk


if __name__ == "__main__":
    run()
