"""FastAPI scoring service (report §11 API contract). Run: uvicorn creditpulse.serve.api:app --port 8000"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from creditpulse.serve.scoring import Scorer, ScoringError

app = FastAPI(
    title="CreditPulse scoring service",
    version="1.0.0",
    description="Early-warning risk scores for loan monitoring (decision support for human analysts only; "
                "not for automated approval, rejection, pricing or intervention).",
)

# Allow the Vite dev server to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def scorer() -> Scorer:
    return Scorer()


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    loan_id: int = Field(..., description="loan_id from loan.csv", examples=[5314])
    as_of_date: dt.date = Field(..., description="scores use data up to the last month-end on or before this date",
                                examples=["1998-06-30"])


class Driver(BaseModel):
    feature: str
    contribution: float
    value: float
    text: str


class Window(BaseModel):
    first_month: str
    last_month: str


class ScoreResponse(BaseModel):
    loan_id: int
    as_of_date: str
    cutoff: str
    status: str = Field(..., description="scored | too_new | in_distress | closed")
    detail: str
    risk_6m: float | None = Field(None, description="calibrated probability of a distress event in the next 6 months")
    risk_raw: float | None = None
    hazard: list[float] | None = Field(None, description="monthly discrete-time hazard h1..h6")
    hazard_scaled: list[float] | None = Field(None, description="Head-B timing shape rescaled so that cumulative "
                                               "incidence at 6 months equals risk_6m")
    cumulative_incidence: list[float] | None = Field(None, description="from hazard_scaled; last value = risk_6m")
    queue_rank: int | None = None
    queue_percentile: float | None = None
    queue_size: int | None = None
    in_top_10pct: bool | None = None
    in_training_domain: bool | None = None
    drivers: list[Driver] = []
    window: Window
    model_version: str


@app.get("/health")
def health():
    s = scorer()
    return {"status": "ok", "model_version": s.version, "data_end": str(s.data_end.date()), "loans": len(s.loans)}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    try:
        return scorer().score(req.loan_id, req.as_of_date)
    except ScoringError as e:
        raise HTTPException(status_code=e.code, detail=e.message)


@app.get("/queue")
def queue(as_of_date: dt.date = Query(..., examples=["1998-06-30"]), top: int = Query(50, ge=1, le=1000)):
    try:
        return {"as_of_date": str(as_of_date), "model_version": scorer().version, "queue": scorer().queue(as_of_date, top)}
    except ScoringError as e:
        raise HTTPException(status_code=e.code, detail=e.message)


@app.get("/loans")
def loans(limit: int = Query(100, ge=1, le=1000)):
    lo = scorer().loans.reset_index().head(limit)
    return [{"loan_id": int(r.loan_id), "account_id": int(r.account_id), "origination": str(r.date.date()),
             "amount": float(r.amount), "duration": int(r.duration)} for r in lo.itertuples()]
