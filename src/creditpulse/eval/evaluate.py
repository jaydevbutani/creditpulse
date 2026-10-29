"""Stage 8 — evaluation (report §9): one test evaluation per final model (R-45), loan-clustered bootstrap
intervals, seed variance, paired differences, calibration, capture@K, lead time, Head-B survival metrics,
ablation table, subgroup table and figures."""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from creditpulse.eval.metrics import (  # noqa: E402
    Platt, brier, brier_skill, c_index, capture_table, ci, cluster_bootstrap, ece, event_capture,
    false_alarm_load, integrated_brier, lead_time_stats, naive_bootstrap, paired_diff, pr_auc, reliability, roc_auc,
)
from creditpulse.utils import Tracker, load_config, p  # noqa: E402

FINAL = ["M0_prior", "M1_logreg", "M2_lgbm", "M2_xgb", "S1_cox", "S2_hazard", "M3_gru", "M4_lstm",
         "M5_transformer", "M6_multitask", "M7_pretrained"]
ABLATIONS = ["M6_no_distress_inputs", "M6_no_loan_static", "M6_mean_pooling", "M6_headA_from_F6",
             "M6_focal_loss", "M6_no_class_weight"]
REFERENCE = "M2_lgbm"


def load_preds(cfg, prefix=""):
    files = glob.glob(str(p(cfg, "artifacts") / "preds" / f"{prefix}*_s*.parquet"))
    groups = defaultdict(list)
    for f in sorted(files):
        name = re.sub(r"_s\d+\.parquet$", "", f.split("/")[-1])
        groups[name].append(pd.read_parquet(f))
    return groups


def ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Seed-average raw scores and hazards, then Platt-calibrate the ensemble on validation."""
    base = frames[0].copy()
    base["score_raw"] = np.mean([f.score_raw.values for f in frames], 0)
    hz = [c for c in base.columns if re.fullmatch(r"h\d", c)]
    for c in hz:
        base[c] = np.mean([f[c].values for f in frames], 0)
    va = (base.split == "val") & base.label.notna()
    cal = Platt().fit(base.score_raw[va].values, base.label[va].values)
    base["score"] = cal(base.score_raw.values)
    base.attrs["platt"] = cal.params()
    return base


def evaluate_model(name, frames, cfg, base_rate, ref=None):
    e = ensemble(frames)
    te = e[(e.split == "test")]
    lab = te[te.label.notna()].rename(columns={"label": "y"})
    nb = cfg["eval"]["n_boot"]
    y, s = lab.y.values, lab.score.values
    row = {"model": name, "n_seeds": len(frames), "platt_slope_fallback": e.attrs["platt"]["slope_fallback"]}
    for mname, fn in [("pr_auc", pr_auc), ("roc_auc", roc_auc)]:
        row[mname] = fn(y, lab.score_raw.values)
        row[f"{mname}_lo"], row[f"{mname}_hi"] = ci(cluster_bootstrap(lab, lambda d, fn=fn: fn(d.y.values, d.score_raw.values), nb, 0))
    row["brier"] = brier(y, s)
    row["brier_skill"] = brier_skill(y, s, base_rate)
    row["ece"] = ece(y, s, cfg["eval"]["ece_bins"])
    row["ece_uncalibrated"] = ece(y, lab.score_raw.values, cfg["eval"]["ece_bins"])
    seed_pr = [pr_auc(f[(f.split == "test") & f.label.notna()].label.values,
                      f[(f.split == "test") & f.label.notna()].score_raw.values) for f in frames]
    row["pr_auc_seed_mean"], row["pr_auc_seed_std"] = float(np.mean(seed_pr)), float(np.std(seed_pr))
    # monitoring metrics on the full at-risk queue (censored snapshots are ranked too)
    for k in cfg["eval"]["capture_k"]:
        row[f"capture@{k}"] = event_capture(te, k)
    va = e[(e.split == "val")]
    thr = float(np.quantile(va.score, 1 - cfg["eval"]["queue_fill"]))
    lt = lead_time_stats(te, thr)
    row.update({"threshold_10pct_queue": thr, "event_loans_test": lt["event_loans"], "flagged_share": lt["flagged_share"],
                "median_lead_months": lt["median_lead"], "share_lead_ge3": lt["share_lead_ge3"],
                "false_alarms_per_month": false_alarm_load(te, thr)})
    if "h1" in e.columns:
        h = te[[f"h{k}" for k in range(1, 7)]].values
        F6 = 1 - np.prod(1 - h, 1)
        row["c_index"] = c_index(te.event_k.values, te.follow_k.values, F6)
        row["ibs"] = integrated_brier(te.event_k.values, te.follow_k.values, h)
    if ref is not None and name != REFERENCE:
        m = lab[["loan_id", "y", "score_raw"]].rename(columns={"score_raw": "score"}).merge(
            ref[["score_raw"]].rename(columns={"score_raw": "score_ref"}), left_index=True, right_index=True)
        d = paired_diff(m.rename(columns={"loan_id": "loan_id"}), "score", "score_ref", pr_auc, nb, 1)
        row.update({f"diff_vs_{REFERENCE}": d["diff"], "diff_lo": d["lo"], "diff_hi": d["hi"],
                    "diff_excludes_zero": d["excludes_zero"]})
    return row, e


def run(cfg=None):
    cfg = cfg or load_config()
    tracker = Tracker(cfg)
    tracker.test_log.write_text("")  # one evaluation per final model per pipeline run (R-45)
    tables, figs = p(cfg, "reports") / "tables", p(cfg, "reports") / "figures"
    groups = load_preds(cfg)
    snap_tr = groups["M0_prior"][0]
    base_rate = float(snap_tr[(snap_tr.split == "train") & snap_tr.label.notna()].label.mean())
    ref_e = ensemble(groups[REFERENCE])
    ref = ref_e[(ref_e.split == "test") & ref_e.label.notna()]
    rows, ens = [], {}
    for name in FINAL + ABLATIONS:
        if name not in groups:
            continue
        row, e = evaluate_model(name, groups[name], cfg, base_rate, ref)
        rows.append(row)
        ens[name] = e
        if name in FINAL:
            tracker.log(f"TEST::{name}", {"n_seeds": row["n_seeds"]}, row, kind="test")
        print(f"{name:26s} PR-AUC {row['pr_auc']:.3f} [{row['pr_auc_lo']:.3f},{row['pr_auc_hi']:.3f}] "
              f"ROC {row['roc_auc']:.3f} cap@10 {row['capture@10']:.2f} lead {row['median_lead_months']}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(tables / "results_test_p1.csv", index=False)
    for name, e in ens.items():
        e.to_parquet(p(cfg, "artifacts") / "preds" / f"ENS_{name}.parquet", index=False)

    # serving manifest: the calibrated M6 seed ensemble is the served model (only model with both heads)
    if "M6_multitask" in ens:
        r6 = res.set_index("model").loc["M6_multitask"]
        seeds = sorted(int(re.search(r"_s(\d+)", f).group(1)) for f in glob.glob(
            str(p(cfg, "artifacts") / "models" / "M6_multitask_s*.pt")))
        manifest = {"model": "M6_multitask", "seeds": seeds, "platt": ens["M6_multitask"].attrs["platt"],
                    "threshold_10pct_queue": float(r6["threshold_10pct_queue"]),
                    "model_version": f"M6_multitask-ens{len(seeds)}-cfg{cfg['_config_hash']}-data{tracker.data_version}"}
        (p(cfg, "artifacts") / "serving.json").write_text(json.dumps(manifest, indent=2))

    # validation table (model selection evidence)
    vrows = []
    for name, e in ens.items():
        v = e[(e.split == "val") & e.label.notna()]
        vrows.append({"model": name, "val_pr_auc": pr_auc(v.label.values, v.score_raw.values),
                      "val_roc_auc": roc_auc(v.label.values, v.score_raw.values)})
    pd.DataFrame(vrows).to_csv(tables / "results_val_p1.csv", index=False)

    # R-40: clustered vs naive bootstrap width
    e = ens[REFERENCE]
    lab = e[(e.split == "test") & e.label.notna()].rename(columns={"label": "y"})
    f = lambda d: pr_auc(d.y.values, d.score_raw.values)  # noqa: E731
    cw = np.diff(ci(cluster_bootstrap(lab, f, cfg["eval"]["n_boot"], 0)))[0]
    nw = np.diff(ci(naive_bootstrap(lab, f, cfg["eval"]["n_boot"], 0)))[0]
    checks = {"R-40": {"pass": bool(cw > nw), "clustered_width": float(cw), "naive_width": float(nw)}}

    # R-34 seed variance rule: no ranking claim when seed std > half the gap
    r = res.set_index("model")
    rank_rows = []
    for a in ["M5_transformer", "M6_multitask", "M7_pretrained", "M3_gru", "M4_lstm", "M1_logreg"]:
        if a in r.index:
            gap = abs(r.loc[a, "pr_auc"] - r.loc[REFERENCE, "pr_auc"])
            sd = max(r.loc[a, "pr_auc_seed_std"], r.loc[REFERENCE, "pr_auc_seed_std"])
            rank_rows.append({"model": a, "vs": REFERENCE, "gap": gap, "max_seed_std": sd,
                              "ranking_claim_allowed_R34": bool(sd <= gap / 2),
                              "paired_ci_excludes_zero_R47": bool(r.loc[a, "diff_excludes_zero"])})
    pd.DataFrame(rank_rows).to_csv(tables / "ranking_claims.csv", index=False)
    checks["R-34"] = {"pass": True, "rows": rank_rows}
    checks["R-44"] = {"pass": bool(res[["pr_auc", "pr_auc_lo", "pr_auc_hi"]].notna().all().all())}

    # ----- figures: reliability before/after (R-41)
    show = [m for m in ["M1_logreg", "M2_lgbm", "M5_transformer", "M6_multitask", "M7_pretrained"] if m in ens]
    fig, axes = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6), squeeze=False)
    for j, m in enumerate(show):
        for i, sp in enumerate(["val", "test"]):
            d = ens[m][(ens[m].split == sp) & ens[m].label.notna()]
            ax = axes[i, j]
            for col, lbl, st in [("score_raw", "before", "o--"), ("score", "after (Platt)", "s-")]:
                rr = reliability(d.label.values, d[col].values, cfg["eval"]["ece_bins"])
                ax.plot(rr.mean_pred, rr.obs_rate, st, label=lbl, ms=4)
            lim = max(0.05, float(d.label.mean()) * 4)
            ax.plot([0, lim], [0, lim], color="grey", lw=0.8)
            ax.set_title(f"{m}\n{sp}", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 1 and j == 0:
                ax.legend(fontsize=7)
    fig.supxlabel("mean predicted probability")
    fig.supylabel("observed event rate")
    fig.tight_layout()
    fig.savefig(figs / "reliability.png", dpi=130)
    plt.close(fig)
    checks["R-41"] = {"pass": True, "figure": "reports/figures/reliability.png", "models": show}

    # ----- forest plot of PR-AUC with intervals
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(res) + 1))
    rr = res.iloc[::-1]
    ax.errorbar(rr.pr_auc, range(len(rr)), xerr=[rr.pr_auc - rr.pr_auc_lo, rr.pr_auc_hi - rr.pr_auc],
                fmt="o", color="#2a6f97", ecolor="#8fb8d0", capsize=2)
    ax.set_yticks(range(len(rr)))
    ax.set_yticklabels(rr.model, fontsize=8)
    ax.axvline(res.loc[res.model == "M0_prior", "pr_auc"].iloc[0], color="grey", ls=":", lw=1, label="prevalence")
    ax.set_xlabel("Test PR-AUC (95% loan-clustered bootstrap CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "pr_auc_forest.png", dpi=130)
    plt.close(fig)

    # ----- subgroup table (R-57) for the served model
    served = "M6_multitask" if "M6_multitask" in ens else REFERENCE
    sub = subgroup_table(cfg, ens[served])
    sub.to_csv(tables / "subgroups.csv", index=False)
    checks["R-57"] = {"pass": True, "rows": len(sub)}
    (tables / "eval_checks.json").write_text(json.dumps(checks, indent=2, default=float))
    return res, checks


def subgroup_table(cfg, e: pd.DataFrame) -> pd.DataFrame:
    interim = p(cfg, "interim")
    loan = pd.read_parquet(interim / "loan.parquet")
    acc = pd.read_parquet(interim / "account.parquet")
    dist = pd.read_parquet(interim / "district.parquet")
    disp = pd.read_parquet(interim / "disp.parquet")
    cli = pd.read_parquet(interim / "client.parquet")
    own = disp[disp.type == "OWNER"].merge(cli, on="client_id")[["account_id", "gender", "birth_date"]]
    meta = loan.merge(acc[["account_id", "district_id"]], on="account_id").merge(
        dist[["district_id", "region"]], on="district_id").merge(own, on="account_id", how="left")
    meta["amount_band"] = pd.qcut(meta.amount, 3, labels=["low", "mid", "high"]).astype(str)
    meta["duration"] = meta.duration.astype(str) + "m"
    meta["age_band"] = pd.cut((meta.date - meta.birth_date).dt.days / 365.25, [0, 30, 45, 60, 100],
                              labels=["<30", "30-45", "45-60", "60+"]).astype(str)
    meta["region_group"] = np.where(meta.region == "Prague", "Prague", meta.region)
    d = e[(e.split == "test") & e.label.notna()].merge(meta, on="loan_id", suffixes=("", "_m")).rename(columns={"label": "y"})
    rows = []
    for col in ["region_group", "amount_band", "duration", "gender", "age_band"]:
        for g, x in d.groupby(col):
            pos = int(x.y.sum())
            ev = int(x[x.y == 1].loan_id.nunique())
            r = {"dimension": col, "group": g, "snapshots": len(x), "positives": pos, "event_loans": ev,
                 "prevalence": float(x.y.mean())}
            if pos >= 1 and pos < len(x):
                r["roc_auc"] = roc_auc(x.y.values, x.score.values)
                r["roc_lo"], r["roc_hi"] = ci(cluster_bootstrap(x, lambda z: roc_auc(z.y.values, z.score.values), 500, 0))
                r["pr_auc"] = pr_auc(x.y.values, x.score.values)
            r["note"] = "too few event loans to judge (<3)" if ev < 3 else ""
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
