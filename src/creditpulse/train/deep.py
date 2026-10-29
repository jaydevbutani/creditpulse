"""Stage 6 — deep model ladder M3 GRU, M4 LSTM, M5 Transformer, M6 multi-task, M7 pretrained M6,
plus M6 ablations. Identical tuning budget for every model (R-35); multi-seed final fits."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F

from creditpulse.eval.metrics import Platt, pr_auc
from creditpulse.features.panel import FEATURES
from creditpulse.features.tokens import PanelIndex, Preprocessor
from creditpulse.models.deep import SeqModel, count_params, cosine_warmup, focal_bce, hazard_nll
from creditpulse.train.data import load_bundle, save_preds
from creditpulse.utils import Tracker, load_config, p, seed_everything

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(int(__import__("os").environ.get("CP_THREADS", "2")))

MODELS = {  # name -> fixed architecture choices; the rest is tuned
    "M3_gru": {"kind": "gru", "head_b": False},
    "M4_lstm": {"kind": "lstm", "head_b": False},
    "M5_transformer": {"kind": "transformer", "head_b": False},
    "M6_multitask": {"kind": "transformer", "head_b": True},
}


def tensors(B, cols):
    return (torch.tensor(B.Z[:, :, cols]), torch.tensor(B.pad), torch.tensor(B.age, dtype=torch.float32),
            torch.tensor(np.nan_to_num(B.y, nan=-1.0), dtype=torch.float32),
            torch.tensor(B.event_k, dtype=torch.long), torch.tensor(B.follow_k, dtype=torch.long))


def stratified_batches(idx, y, bs, rng):
    """Spread positives evenly over batches (report §8.4 'stratified batches')."""
    pos, neg = rng.permutation(idx[y[idx] == 1]), rng.permutation(idx[y[idx] != 1])
    nb = max(1, int(np.ceil(len(idx) / bs)))
    batches = [list(a) + list(b) for a, b in zip(np.array_split(pos, nb), np.array_split(neg, nb))]
    rng.shuffle(batches)
    return [np.array(b) for b in batches if len(b)]


@torch.no_grad()
def predict(model, T, idx, bs=1024):
    model.eval()
    la, hz = [], []
    for s in range(0, len(idx), bs):
        b = torch.as_tensor(idx[s:s + bs])
        o = model(T[0][b], T[1][b], T[2][b])
        la.append(o["logit_a"])
        if "hz_logits" in o:
            hz.append(torch.sigmoid(o["hz_logits"]))
    la = torch.cat(la).numpy()
    return 1 / (1 + np.exp(-la)), (torch.cat(hz).numpy() if hz else None), la


def build_model(n_feat, arch, hp):
    return SeqModel(n_feat, kind=arch["kind"], d_model=hp["d_model"], n_layers=hp["n_layers"],
                    n_heads=hp.get("n_heads", 4), dropout=hp["dropout"], pooling=hp.get("pooling", "cls"),
                    head_b=arch["head_b"], derive_a=hp.get("derive_a", False), horizon=arch.get("horizon", 6))


def train_one(T, y, arch, hp, seed, fit_idx, eval_idx=None, epochs=None, cfg=None, init_state=None, history=None):
    """Train one model. With eval_idx: early stopping on eval PR-AUC (patience). Without: fixed epochs."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = build_model(T[0].shape[-1], arch, hp)
    if init_state is not None:
        missing = model.load_state_dict(init_state, strict=False)
        assert not [k for k in missing.missing_keys if not k.startswith(("head", "recon"))], missing
    if not arch["head_b"]:
        fit_idx = fit_idx[y[fit_idx] >= 0]  # Head-A-only models train on labelled snapshots
    max_ep = epochs or cfg["train"]["max_epochs"]
    patience = cfg["train"]["patience"]
    bs = cfg["train"]["batch_size"]
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    steps_per = int(np.ceil(len(fit_idx) / bs))
    total = steps_per * max_ep
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_warmup(s, total, max(1, int(cfg["train"]["warmup_frac"] * total))))
    pw = torch.tensor(hp["pos_weight"])
    lam = hp.get("lam", 0.0)
    best, best_ep, best_state, bad, hist = -1.0, 0, None, 0, []
    for ep in range(max_ep):
        model.train()
        losses, gnorms = [], []
        for b in stratified_batches(fit_idx, y, bs, rng):
            bt = torch.as_tensor(b)
            o = model(T[0][bt], T[1][bt], T[2][bt])
            yb = T[3][bt]
            lab = yb >= 0
            if hp.get("focal"):
                la = focal_bce(o["logit_a"][lab], yb[lab], pw) if lab.any() else 0.0
            else:
                la = F.binary_cross_entropy_with_logits(o["logit_a"][lab], yb[lab], pos_weight=pw) if lab.any() else torch.tensor(0.0)
            loss = la
            if arch["head_b"] and lam > 0:
                loss = loss + lam * hazard_nll(o["hz_logits"], T[4][bt], T[5][bt]).mean()
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            if not torch.isfinite(gn):
                raise FloatingPointError("non-finite gradient (R-38)")
            opt.step()
            sched.step()
            losses.append(float(loss))
            gnorms.append(float(gn))
        rec = {"epoch": ep + 1, "loss": float(np.mean(losses)), "grad_norm": float(np.mean(gnorms))}
        if eval_idx is not None:
            s, _, _ = predict(model, T, eval_idx)
            sc = pr_auc(y[eval_idx], s)
            rec["eval_pr_auc"] = sc
            if sc > best + 1e-6:
                best, best_ep, best_state, bad = sc, ep + 1, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    hist.append(rec)
                    break
        hist.append(rec)
    if history is not None:
        history.extend(hist)
    if eval_idx is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep or max_ep, "best_eval": best, "epochs_run": len(hist), "history": hist}


def space(trial, arch):
    hp = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-4, 1e-2, log=True),
        "d_model": trial.suggest_categorical("d_model", [32, 64]),
        "n_layers": trial.suggest_int("n_layers", 1, 2 if arch["kind"] != "transformer" else 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        "pos_weight": trial.suggest_float("pos_weight", 1.0, 30.0, log=True),
    }
    if arch["kind"] == "transformer":
        hp["n_heads"] = trial.suggest_categorical("n_heads", [2, 4])
    if arch["head_b"]:
        hp["lam"] = trial.suggest_categorical("lam", [0.1, 0.3, 1.0])
    return hp


def tune_deep(name, arch, B, T, cfg, tracker):
    folds = B.tuning_folds(cfg["splits"]["p1"]["inner_start"])
    all_tr = np.flatnonzero((B.snap.split == "train").to_numpy())
    y = T[3].numpy()
    orig = B.snap.orig_date.values
    inner = np.datetime64(cfg["splits"]["p1"]["inner_start"])
    fit_sets = [all_tr[orig[all_tr] < inner], all_tr]  # Head-B models also see censored snapshots

    def objective(trial):
        hp = space(trial, arch)
        ys, ss, eps = [], [], []
        for (_, ev), fit in zip(folds, fit_sets):
            m, info = train_one(T, y, arch, hp, cfg["seed"], fit, ev, cfg=cfg)
            s, _, _ = predict(m, T, ev)
            ys.append(y[ev])
            ss.append(s)
            eps.append(info["best_epoch"])
        trial.set_user_attr("epochs", eps)
        return pr_auc(np.concatenate(ys), np.concatenate(ss))

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=cfg["train"]["n_trials"])
    bt = study.best_trial
    hp = space(optuna.trial.FixedTrial(bt.params), arch)
    epochs = int(max(5, round(np.mean(bt.user_attrs["epochs"]))))
    tracker.log(f"tune_{name}", {"n_trials": cfg["train"]["n_trials"], "max_epochs": cfg["train"]["max_epochs"],
                                 "patience": cfg["train"]["patience"], "best_params": bt.params, "final_epochs": epochs},
                {"pooled_tuning_pr_auc": bt.value, "minutes": (time.time() - t0) / 60}, kind="tuning")
    return hp, epochs, bt.value


def final_fits(name, arch, hp, epochs, B, T, cfg, tracker, seeds, init_state=None, save_models=False, cols=None):
    tr = np.flatnonzero((B.snap.split == "train").to_numpy())
    va = B.idx("val")
    y = T[3].numpy()
    all_idx = np.arange(len(B.snap))
    vals = []
    for sd in seeds:
        hist = []
        m, info = train_one(T, y, arch, hp, sd, tr, None, epochs=epochs, cfg=cfg, init_state=init_state, history=hist)
        s, hz, logit = predict(m, T, all_idx)
        cal = Platt().fit(s[va], y[va])
        save_preds(cfg, f"{name}_s{sd}", B.snap, s, cal(s), hazard=hz)
        vp = pr_auc(y[va], s[va])
        vals.append(vp)
        tracker.log(f"{name}_s{sd}", {**hp, "epochs": epochs, "n_params": count_params(m), "seed": sd},
                    {"val_pr_auc": vp, "final_loss": hist[-1]["loss"], "loss_curve": [h["loss"] for h in hist],
                     "grad_norm_curve": [h["grad_norm"] for h in hist]})
        if save_models:
            d = p(cfg, "artifacts") / "models"
            d.mkdir(exist_ok=True)
            torch.save({"state": m.state_dict(), "hp": hp, "arch": arch, "cols": cols, "platt": cal.params(),
                        "n_params": count_params(m)}, d / f"{name}_s{sd}.pt")
    return vals


def pretrain_encoder(arch, hp, cfg, tracker, cols):
    """M7: masked-month reconstruction on all 4,500 accounts, months up to the training end only (R-36)."""
    proc = p(cfg, "processed")
    panel = pd.read_parquet(proc / "panel.parquet")
    idx = PanelIndex(panel)
    pre = Preprocessor.from_dict(json.loads((p(cfg, "artifacts") / "preprocessor.json").read_text()))
    from creditpulse.labels.events import month_index
    end_mi = month_index(pd.Timestamp(cfg["splits"]["p1"]["train_end"]))
    W = cfg["labels"]["window"]
    rows = [(a, c) for a in idx.mi0 for c in range(idx.mi0[a] + 2, min(idx.mi1[a], end_mi) + 1)]
    rng = np.random.default_rng(cfg["seed"])
    rows = [rows[i] for i in rng.permutation(len(rows))[: cfg["pretrain"]["max_windows"]]]
    acc = np.array([r[0] for r in rows])
    cut = np.array([r[1] for r in rows])
    assert cut.max() <= end_mi, "R-36: pretraining read a month after the training end date"
    X, pad = idx.gather(acc, cut, W)
    Z = pre.transform(X, pad)[:, :, cols]
    la = FEATURES.index("loan_age")
    has = FEATURES.index("has_loan")
    age = np.where(X[:, :, has] > 0, X[:, :, la], -24).astype(np.float32)
    seed_everything(cfg["seed"])
    model = build_model(len(cols), arch, hp)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=hp["wd"])
    Zt, Pt, At = torch.tensor(Z), torch.tensor(pad), torch.tensor(age)
    curve = []
    for ep in range(cfg["pretrain"]["epochs"]):
        model.train()
        perm = rng.permutation(len(Z))
        ls = []
        for s in range(0, len(perm), 256):
            b = torch.as_tensor(perm[s:s + 256])
            loss = model.pretrain_loss(Zt[b], Pt[b], At[b], cfg["pretrain"]["mask_frac"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ls.append(float(loss))
        curve.append(float(np.mean(ls)))
    tracker.log("M7_pretrain", {"windows": len(Z), "max_cutoff_mi": int(cut.max()), "train_end_mi": end_mi},
                {"recon_loss_curve": curve})
    state = {k: v for k, v in model.state_dict().items() if not k.startswith(("head", "recon"))}
    return state, {"windows": int(len(Z)), "max_cutoff": str(pd.Period(year=int(cut.max()) // 12, month=int(cut.max()) % 12 + 1, freq="M")),
                   "loss_first": curve[0], "loss_last": curve[-1]}


def hparams_path(cfg, name):
    d = p(cfg, "artifacts") / "hparams"
    d.mkdir(exist_ok=True)
    return d / f"{name}.json"


def run(cfg=None, only=None, post=False, ablation_seeds=None):
    """Tune + fit the ladder (`only` selects models so two processes can share the CPU), or, with
    post=True, run M7 pretraining and the M6 ablations from the saved M6 hyperparameters."""
    cfg = cfg or load_config()
    tracker = Tracker(cfg)
    B = load_bundle(cfg)
    all_cols = np.arange(len(FEATURES))
    T = tensors(B, all_cols)
    seeds = cfg["train"]["seeds"]
    summary = {}
    if not post:
        for name, arch in MODELS.items():
            if only and name not in only:
                continue
            t0 = time.time()
            hp, epochs, score = tune_deep(name, arch, B, T, cfg, tracker)
            hparams_path(cfg, name).write_text(json.dumps({"hp": hp, "epochs": epochs, "tuning_pr_auc": score}))
            vals = final_fits(name, arch, hp, epochs, B, T, cfg, tracker, seeds,
                              save_models=(name == "M6_multitask"), cols=all_cols.tolist())
            summary[name] = {"tuning_pr_auc": score, "val_pr_auc_mean": float(np.mean(vals)), "epochs": epochs,
                             "hp": hp, "minutes": (time.time() - t0) / 60}
            print(name, summary[name], flush=True)
        return summary

    # M7 = M6 + self-supervised pretraining (shares M6's tuned hyperparameters; no extra tuning budget)
    m6 = json.loads(hparams_path(cfg, "M6_multitask").read_text())
    arch6 = MODELS["M6_multitask"]
    enc_file = p(cfg, "artifacts") / "models" / "M7_encoder.pt"
    enc_file.parent.mkdir(exist_ok=True)
    if enc_file.exists():
        saved = torch.load(enc_file, weights_only=False)
        state, pinfo = saved["state"], saved["info"]
    else:
        state, pinfo = pretrain_encoder(arch6, m6["hp"], cfg, tracker, all_cols)
        torch.save({"state": state, "info": pinfo}, enc_file)
    vals = final_fits("M7_pretrained", arch6, m6["hp"], m6["epochs"], B, T, cfg, tracker, seeds, init_state=state)
    summary["M7_pretrained"] = {"val_pr_auc_mean": float(np.mean(vals)), "pretrain": pinfo}
    print("M7_pretrained", summary["M7_pretrained"], flush=True)

    # Ablations on M6 (deck slide 13) — same hyperparameters and epochs
    abl = {
        "M6_no_distress_inputs": {"drop": ["distress"]},
        "M6_no_loan_static": {"drop": ["loan_static"]},
        "M6_mean_pooling": {"hp": {"pooling": "mean"}},
        "M6_headA_from_F6": {"hp": {"derive_a": True}},
        "M6_focal_loss": {"hp": {"focal": True}},
        "M6_no_class_weight": {"hp": {"pos_weight": 1.0}},
    }
    for name, spec in abl.items():
        cols = np.flatnonzero(B.feature_mask(spec.get("drop", ())))
        Ta = tensors(B, cols) if "drop" in spec else T
        hp = {**m6["hp"], **spec.get("hp", {})}
        vals = final_fits(name, arch6, hp, m6["epochs"], B, Ta, cfg, tracker, ablation_seeds or seeds)
        summary[name] = {"val_pr_auc_mean": float(np.mean(vals))}
        print(name, summary[name], flush=True)
    (p(cfg, "reports") / "tables" / "deep_post_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--post":
        run(post=True, ablation_seeds=[0, 1, 2])
    else:
        run(only=args or None)
