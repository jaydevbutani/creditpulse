"""Data/label/split/leakage checks recorded by the pipeline (R-04..R-17, R-21, R-25..R-28, T2..T7)."""
import json

import pandas as pd
import pytest

from creditpulse.labels.events import split_fingerprint
from creditpulse.utils import ROOT

T = ROOT / "reports" / "tables"


def load(name):
    f = T / name
    if not f.exists():
        pytest.skip(f"{name} missing — run the pipeline first")
    return json.loads(f.read_text())


@pytest.mark.parametrize("rid", ["R-04", "R-05", "R-06", "R-07", "R-08", "R-09", "R-10", "R-11", "R-12"])
def test_data_checks(rid):
    assert load("data_checks.json")[rid]["pass"]


@pytest.mark.parametrize("rid", ["R-14", "R-15", "R-16", "R-17", "R-25", "R-26", "R-27"])
def test_label_checks(rid):
    assert load("label_checks.json")[rid]["pass"]


def test_panel_complete():
    assert load("panel_checks.json")["R-21"]["pass"]


def test_split_fingerprint_frozen():
    """R-27: the split hash recomputed from the snapshot table equals the frozen value."""
    snap = pd.read_parquet(ROOT / "data" / "processed" / "snapshots.parquet")
    frozen = (ROOT / "data" / "processed" / "split_fingerprint.txt").read_text().strip()
    assert split_fingerprint(snap) == frozen


def test_no_loan_in_two_splits():
    snap = pd.read_parquet(ROOT / "data" / "processed" / "snapshots.parquet")
    assert snap.groupby("loan_id").split.nunique().max() == 1  # T3


def test_label_shuffle_is_chance():
    ctrl = load("baseline_summary.json")["controls"]
    for m, v in ctrl["T2_label_shuffle"].items():
        assert 0.4 <= v["test_roc_auc_mean"] <= 0.6, m  # R-28
    pr = ctrl["R-28_prior_prauc_equals_prevalence"]
    assert abs(pr["prior_pr_auc"] - pr["prevalence"]) < 1e-9


def test_pretraining_boundary():
    """T7 / R-36: pretraining windows end on or before the training end date."""
    f = ROOT / "reports" / "runs.jsonl"
    if not f.exists():
        pytest.skip("no runs")
    recs = [json.loads(line) for line in f.read_text().splitlines() if '"M7_pretrain"' in line]
    if not recs:
        pytest.skip("M7 not run")
    for r in recs:
        assert r["params"]["max_cutoff_mi"] <= r["params"]["train_end_mi"]


def test_p2_time_order():
    c = T / "sensitivity_checks.json"
    if not c.exists():
        pytest.skip("sensitivity not run")
    assert json.loads(c.read_text())["T4_time_order_P2"]["pass"]


def test_equal_tuning_budget():
    """R-35: every tuned model used the same number of trials."""
    f = ROOT / "reports" / "runs.jsonl"
    recs = [json.loads(line) for line in f.read_text().splitlines()]
    trials = {r["name"]: r["params"]["n_trials"] for r in recs if r["kind"] == "tuning" and "CTRL" not in r["name"]}
    assert len(set(trials.values())) == 1, trials


def test_one_test_evaluation_per_model():
    """R-45: the test log holds exactly one evaluation per final model."""
    f = ROOT / "reports" / "test_evaluations.jsonl"
    if not f.exists() or not f.read_text().strip():
        pytest.skip("evaluation not run")
    names = [json.loads(line)["name"] for line in f.read_text().splitlines() if line.startswith("{")]
    names = [n for n in names if n.startswith("TEST::")]
    assert len(names) == len(set(names))
