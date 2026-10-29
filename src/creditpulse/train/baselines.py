"""Stage 5 — train and tune the classical ladder (M0, M1, M2-LGBM, M2-XGB, S1, S2), Task O, and the
negative controls T2 (label shuffle) and T8 (time shift). Every model gets the same Optuna budget (R-35)."""
from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd

from creditpulse.eval.metrics import Platt, c_index, cumulative_incidence, pr_auc, roc_auc
from creditpulse.features.tokens import PanelIndex, Preprocessor, aggregate
from creditpulse.labels.events import month_index
from creditpulse.models.baselines import Cox, DiscreteHazard, Prior, make_lgbm, make_lr, make_xgb
from creditpulse.train.data import load_bundle, save_preds
from creditpulse.utils import Tracker, load_config, p, seed_everything

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(name, objective, n_trials, seed, tracker):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    tracker.log(f"tune_{name}", {"n_trials": n_trials, "best_params": study.best_params},
                {"tuning_score": study.best_value}, kind="tuning")
    return study.best_params, study.best_value


def pooled_score(B, y, folds, fit_predict, metric=pr_auc):
    """Fit on each tuning fold, pool held-out predictions, score once."""
    ys, ss = [], []
    for fit_idx, ev_idx in folds:
        ss.append(fit_predict(fit_idx, ev_idx))
        ys.append(y[ev_idx])
    return metric(np.concatenate(ys), np.concatenate(ss))


def fit_tabular(cfg, B, tracker, name_prefix="", col_mask=None, y_override=None, save=True, seeds=None):
    """Tune and fit M0, M1, M2 on aggregated features. Returns dict name -> list of score arrays."""
    seeds = seeds if seeds is not None else cfg["train"]["seeds"]
    A = B.A if col_mask is None else B.A[:, col_mask]
    y = B.y if y_override is None else y_override
    tr, va = B.idx("train"), B.idx("val")
    folds = B.tuning_folds(cfg["splits"]["p1"]["inner_start"])
    n_trials = cfg["train"]["n_trials"]
    out = {}

    # M0 prior
    m0 = Prior().fit(A[tr], y[tr])
    s = m0.predict_proba(A)
    if save:
        save_preds(cfg, f"{name_prefix}M0_prior_s0", B.snap, s, s)
    out["M0_prior"] = [s]

    # M1 logistic regression
    def obj_lr(t):
        C = t.suggest_float("C", 1e-4, 1.0, log=True)
        return pooled_score(B, y, folds, lambda f, e: make_lr(C).fit(A[f], y[f]).predict_proba(A[e])[:, 1])

    bp, _ = tune(f"{name_prefix}M1", obj_lr, n_trials, cfg["seed"], tracker)
    m1 = make_lr(bp["C"]).fit(A[tr], y[tr])
    s = m1.predict_proba(A)[:, 1]
    cal = Platt().fit(s[va], y[va])
    if save:
        save_preds(cfg, f"{name_prefix}M1_logreg_s0", B.snap, s, cal(s))
    out["M1_logreg"] = [s]

    # M2 LightGBM / XGBoost
    for nm, maker, space in [
        ("M2_lgbm", make_lgbm, lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": t.suggest_int("num_leaves", 4, 31),
            "min_child_samples": t.suggest_int("min_child_samples", 10, 80),
            "colsample": t.suggest_float("colsample", 0.2, 0.8),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 10, log=True),
            "spw": t.suggest_float("spw", 1, 40, log=True)}),
        ("M2_xgb", make_xgb, lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 5),
            "min_child_weight": t.suggest_float("min_child_weight", 1, 20, log=True),
            "colsample": t.suggest_float("colsample", 0.2, 0.8),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 10, log=True),
            "spw": t.suggest_float("spw", 1, 40, log=True)}),
    ]:
        def obj(t, maker=maker, space=space):
            prm = space(t)
            return pooled_score(B, y, folds,
                                lambda f, e: maker(prm, cfg["seed"]).fit(A[f], y[f]).predict_proba(A[e])[:, 1])

        bp, bv = tune(f"{name_prefix}{nm}", obj, n_trials, cfg["seed"], tracker)
        scores = []
        for sd in seeds:
            seed_everything(sd)
            m = maker(bp, sd).fit(A[tr], y[tr])
            s = m.predict_proba(A)[:, 1]
            cal = Platt().fit(s[va], y[va])
            if save:
                save_preds(cfg, f"{name_prefix}{nm}_s{sd}", B.snap, s, cal(s))
                if nm == "M2_lgbm" and sd == seeds[0] and not name_prefix:
                    import joblib

                    joblib.dump({"model": m, "params": bp, "columns": B.acols, "platt": cal.params()},
                                p(cfg, "artifacts") / "m2_lgbm.joblib")
            tracker.log(f"{name_prefix}{nm}_s{sd}", bp, {"val_pr_auc": pr_auc(y[va], s[va])})
            scores.append(s)
        out[nm] = scores
        (p(cfg, "artifacts") / f"{name_prefix}{nm}_params.json").write_text(json.dumps(bp))
    return out


def fit_survival(cfg, B, tracker):
    tr, va = B.idx("train", labelled=False), B.idx("val", labelled=False)
    orig = B.snap.orig_date.values
    inner = np.datetime64(cfg["splits"]["p1"]["inner_start"])
    sfolds = [(tr[orig[tr] < inner], tr[orig[tr] >= inner]), (tr, va)]
    n_trials = cfg["train"]["n_trials"]
    res = {}
    for nm, cls, lo, hi in [("S1_cox", Cox, 1e-3, 0.5), ("S2_hazard", DiscreteHazard, 1e-4, 1.0)]:
        key = "alpha" if nm == "S1_cox" else "C"

        def obj(t, cls=cls, key=key, lo=lo, hi=hi):
            v = t.suggest_float(key, lo, hi, log=True)
            ek, fk, rk = [], [], []
            for f, e in sfolds:
                m = cls(**{key: v}).fit(B.A[f], B.event_k[f], B.follow_k[f])
                rk.append(cumulative_incidence(m.hazard(B.A[e]))[:, -1])
                ek.append(B.event_k[e])
                fk.append(B.follow_k[e])
            return c_index(np.concatenate(ek), np.concatenate(fk), np.concatenate(rk))

        bp, bv = tune(nm, obj, n_trials, cfg["seed"], tracker)
        m = cls(**bp).fit(B.A[tr], B.event_k[tr], B.follow_k[tr])
        h = m.hazard(B.A)
        F6 = cumulative_incidence(h)[:, -1]
        vl = B.idx("val")
        cal = Platt().fit(F6[vl], B.y[vl])
        save_preds(cfg, f"{nm}_s0", B.snap, F6, cal(F6), hazard=h)
        tracker.log(nm, bp, {"val_c_index": bv, "train_c_index": c_index(B.event_k[tr], B.follow_k[tr], F6[tr])})
        res[nm] = {"val_c_index": bv, "train_c_index": c_index(B.event_k[tr], B.follow_k[tr], F6[tr]), **bp}
    return res


def task_origination(cfg, tracker):
    """Task O: one sample per loan at origination, pre-loan history only, label = final status B/D."""
    proc = p(cfg, "processed")
    lo = pd.read_parquet(proc / "loans_events.parquet")
    panel = pd.read_parquet(proc / "panel.parquet")
    idx = PanelIndex(panel)
    pre = Preprocessor.from_dict(json.loads((p(cfg, "artifacts") / "preprocessor.json").read_text()))
    cut = lo.start_mi.values - 1
    m0 = np.array([idx.mi0[a] for a in lo.account_id])
    ok = cut >= m0  # needs at least one full month of history before the loan
    lo = lo[ok].reset_index(drop=True)
    X, pad = idx.gather(lo.account_id.values, lo.start_mi.values - 1, cfg["labels"]["window"])
    # loan terms are known at origination: overwrite the pre-loan rows' loan-static values at M0
    Z = pre.transform(X, pad)
    A, _ = aggregate(Z, pad)
    y = lo.bad.values
    tr_end, va_end = (pd.Timestamp(cfg["splits"]["p1"][k]) for k in ["train_end", "val_end"])
    split = np.select([lo.date <= tr_end, lo.date <= va_end], ["train", "val"], "test")
    tr, va, te = (np.flatnonzero(split == s) for s in ["train", "val", "test"])
    out = {"n_loans": int(len(lo)), "positives": {s: int(y[split == s].sum()) for s in ["train", "val", "test"]}}
    for nm, fit in [("O_M1_logreg", lambda C: make_lr(C)), ("O_M2_lgbm", None)]:
        if fit:
            def obj(t):
                C = t.suggest_float("C", 1e-4, 1.0, log=True)
                return pr_auc(y[va], make_lr(C).fit(A[tr], y[tr]).predict_proba(A[va])[:, 1])
            bp, _ = tune(nm, obj, cfg["train"]["n_trials"], cfg["seed"], tracker)
            s = make_lr(bp["C"]).fit(A[tr], y[tr]).predict_proba(A)[:, 1]
        else:
            def obj(t):
                prm = {"n_estimators": t.suggest_int("n_estimators", 50, 400, step=50),
                       "num_leaves": t.suggest_int("num_leaves", 3, 15),
                       "min_child_samples": t.suggest_int("min_child_samples", 5, 40),
                       "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                       "spw": t.suggest_float("spw", 1, 20, log=True)}
                return pr_auc(y[va], make_lgbm(prm, 0).fit(A[tr], y[tr]).predict_proba(A[va])[:, 1])
            bp, _ = tune(nm, obj, cfg["train"]["n_trials"], cfg["seed"], tracker)
            s = np.mean([make_lgbm(bp, sd).fit(A[tr], y[tr]).predict_proba(A)[:, 1] for sd in cfg["train"]["seeds"]], 0)
        out[nm] = {"test_pr_auc": pr_auc(y[te], s[te]), "test_roc_auc": roc_auc(y[te], s[te]),
                   "val_pr_auc": pr_auc(y[va], s[va]), "test_prevalence": float(y[te].mean())}
    pd.DataFrame({"loan_id": lo.loan_id, "split": split, "y": y, "score": s}).to_parquet(
        p(cfg, "artifacts") / "preds" / "taskO_M2_lgbm.parquet", index=False)
    tracker.log("task_O", {}, out, kind="test")
    return out


def controls(cfg, B, tracker):
    """T2 label shuffle (R-28) and T8 time shift (inputs end 6 months earlier)."""
    res = {}
    tr, te = B.idx("train"), B.idx("test")
    reps = {}
    for r in range(5):
        rng = np.random.default_rng(cfg["seed"] + r)
        y_sh = B.y.copy()
        y_sh[tr] = rng.permutation(y_sh[tr])
        out = fit_tabular(cfg, B, tracker, name_prefix="CTRL_shuffle_", y_override=y_sh, save=False, seeds=[0])
        for k, v in out.items():
            if k != "M0_prior":
                reps.setdefault(k, []).append(roc_auc(B.y[te], v[0][te]))
    res["T2_label_shuffle"] = {k: {"test_roc_auc_mean": float(np.mean(v)), "min": float(np.min(v)),
                                   "max": float(np.max(v)), "reps": len(v)} for k, v in reps.items()}
    res["R-28_prior_prauc_equals_prevalence"] = {
        "prior_pr_auc": pr_auc(B.y[te], np.full(len(te), B.y[tr].mean())), "prevalence": float(B.y[te].mean())}
    # T8 time shift
    proc = p(cfg, "processed")
    idx = PanelIndex(pd.read_parquet(proc / "panel.parquet"))
    pre = Preprocessor.from_dict(json.loads((p(cfg, "artifacts") / "preprocessor.json").read_text()))
    cut = B.snap.cutoff_mi.values - 6
    m0 = np.array([idx.mi0[a] for a in B.snap.account_id])
    X, pad = idx.gather(B.snap.account_id.values, np.maximum(cut, m0), cfg["labels"]["window"])
    Z = pre.transform(X, pad)
    A, _ = aggregate(Z, pad)
    B2 = type(B)(**{**B.__dict__, "A": A})
    out = fit_tabular(cfg, B2, tracker, name_prefix="CTRL_shift6_", save=False, seeds=[0])
    res["T8_time_shift"] = {k: {"test_pr_auc": pr_auc(B.y[te], v[0][te]), "test_roc_auc": roc_auc(B.y[te], v[0][te])}
                            for k, v in out.items() if k != "M0_prior"}
    return res


def run(cfg=None):
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    tracker = Tracker(cfg)
    B = load_bundle(cfg)
    (p(cfg, "artifacts") / "preds").mkdir(exist_ok=True, parents=True)
    fit_tabular(cfg, B, tracker)
    surv = fit_survival(cfg, B, tracker)
    to = task_origination(cfg, tracker)
    ctrl = controls(cfg, B, tracker)
    res = {"survival": surv, "task_O": to, "controls": ctrl}
    (p(cfg, "reports") / "tables" / "baseline_summary.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=float))
