"""R-51 API contract, R-52 train/serve parity, R-55 batch speed."""
import time

import numpy as np
import pandas as pd
import pytest

from creditpulse.utils import ROOT

ART = ROOT / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "serving.json").exists(), reason="run the pipeline (make all) first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from creditpulse.serve.api import app

    return TestClient(app)


@pytest.fixture(scope="module")
def scorer():
    from creditpulse.serve.api import scorer as s

    return s()


def a_scored_pair(scorer):
    q = scorer.score_month(scorer.data_end_mi - 6)
    return int(q.loan_id.iloc[0]), "1998-06-30"


def test_valid_request(client, scorer):
    lid, d = a_scored_pair(scorer)
    r = client.post("/score", json={"loan_id": lid, "as_of_date": d})
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "scored" and 0 <= b["risk_6m"] <= 1 and len(b["hazard"]) == 6
    assert 1 <= b["queue_rank"] <= b["queue_size"] and 0 <= b["queue_percentile"] < 100
    assert len(b["drivers"]) <= 3 and b["window"]["last_month"] == "1998-06" and b["model_version"]


def test_unknown_loan(client):
    assert client.post("/score", json={"loan_id": 999999, "as_of_date": "1998-06-30"}).status_code == 404


def test_date_after_data(client, scorer):
    lid, _ = a_scored_pair(scorer)
    r = client.post("/score", json={"loan_id": lid, "as_of_date": "1999-03-31"})
    assert r.status_code == 422 and "after the last available data" in r.json()["detail"]


def test_malformed_body(client):
    assert client.post("/score", json={"loan_id": "abc", "as_of_date": "1998-06-30"}).status_code == 422
    assert client.post("/score", json={"loan_id": 5314}).status_code == 422
    assert client.post("/score", json={"loan_id": 5314, "as_of_date": "1998-13-40"}).status_code == 422
    assert client.post("/score", json={"loan_id": 5314, "as_of_date": "1998-06-30", "x": 1}).status_code == 422


def test_not_originated(client, scorer):
    lo = scorer.loans.reset_index()
    late = lo.sort_values("date").iloc[-1]
    r = client.post("/score", json={"loan_id": int(late.loan_id), "as_of_date": "1994-01-31"})
    assert r.status_code == 422


def test_too_new_loan_gets_no_score(client, scorer):
    lo = scorer.loans.reset_index()
    r0 = lo.iloc[0]
    d = (r0.date + __import__("pandas").offsets.MonthEnd(1)).date()
    r = client.post("/score", json={"loan_id": int(r0.loan_id), "as_of_date": str(d)})
    assert r.status_code == 200 and r.json()["status"] == "too_new" and r.json()["risk_6m"] is None


def test_health_and_queue(client):
    assert client.get("/health").json()["status"] == "ok"
    q = client.get("/queue", params={"as_of_date": "1998-06-30", "top": 5}).json()["queue"]
    assert [x["queue_rank"] for x in q] == [1, 2, 3, 4, 5]


def test_train_serve_parity(scorer):
    """R-52: for 100 random (loan, cutoff) pairs the service equals the offline pipeline within 1e-6."""
    ens_path = ART / "preds" / f"ENS_{scorer.serving['model']}.parquet"
    if not ens_path.exists():
        ens_path = ART / "scores" / "monthly_scores.parquet"
    ens = pd.read_parquet(ens_path)
    score_col = "score" if "score" in ens.columns else "risk_6m"
    pick = ens.sample(100, random_state=0)
    diffs = []
    for r in pick.itertuples():
        q = scorer.score_month(int(r.cutoff_mi))
        diffs.append(abs(float(q.loc[q.loan_id == r.loan_id, "risk_6m"].iloc[0]) - getattr(r, score_col)))
    assert max(diffs) < 1e-6, max(diffs)


def test_batch_speed(scorer):
    """R-55: scoring every at-risk loan for one month takes under 60 s on CPU."""
    scorer._month_cache.clear()
    t0 = time.time()
    q = scorer.score_month(scorer.data_end_mi - 1)
    assert len(q) > 50 and time.time() - t0 < 60
    scorer._month_cache.clear()
    t0 = time.time()
    scorer.score_month(scorer.data_end_mi - 1, with_drivers=True)  # incl. reason codes for every loan
    assert time.time() - t0 < 120
