"""Classical baselines: M0 prior, M1 logistic regression, M2 gradient boosting, S1 Cox, S2 discrete hazard."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Prior:
    def fit(self, X, y):
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)


def make_lr(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))


def make_lgbm(params: dict, seed: int):
    return lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 300), learning_rate=params.get("learning_rate", 0.03),
        num_leaves=params.get("num_leaves", 15), min_child_samples=params.get("min_child_samples", 30),
        subsample=params.get("subsample", 0.8), subsample_freq=1, colsample_bytree=params.get("colsample", 0.5),
        reg_lambda=params.get("reg_lambda", 1.0), scale_pos_weight=params.get("spw", 10.0),
        random_state=seed, verbose=-1, n_jobs=2,
    )


def make_xgb(params: dict, seed: int):
    return xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 300), learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 3), min_child_weight=params.get("min_child_weight", 5),
        subsample=params.get("subsample", 0.8), colsample_bytree=params.get("colsample", 0.5),
        reg_lambda=params.get("reg_lambda", 1.0), scale_pos_weight=params.get("spw", 10.0),
        random_state=seed, n_jobs=2, eval_metric="aucpr", tree_method="hist",
    )


def person_period(X: np.ndarray, event_k: np.ndarray, follow_k: np.ndarray, H: int = 6):
    """Expand snapshots into (snapshot, month) rows for the discrete-time hazard (S2)."""
    rows, ys, ks = [], [], []
    for i in range(len(X)):
        last = event_k[i] if event_k[i] > 0 else follow_k[i]
        for k in range(1, int(min(last, H)) + 1):
            rows.append(i)
            ks.append(k)
            ys.append(1 if (event_k[i] == k) else 0)
    rows, ks = np.array(rows), np.array(ks)
    onehot = np.eye(H)[ks - 1]
    return np.hstack([X[rows], onehot]), np.array(ys), rows


class DiscreteHazard:
    """S2: logistic regression on person-period rows with a month-specific intercept."""

    def __init__(self, C=0.1, H=6):
        self.C, self.H = C, H

    def fit(self, X, event_k, follow_k):
        self.sc = StandardScaler().fit(X)
        Xp, yp, _ = person_period(self.sc.transform(X), event_k, follow_k, self.H)
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(Xp, yp)
        return self

    def hazard(self, X):
        Xs = self.sc.transform(X)
        out = np.zeros((len(X), self.H))
        for k in range(self.H):
            oh = np.zeros((len(X), self.H))
            oh[:, k] = 1
            out[:, k] = self.lr.predict_proba(np.hstack([Xs, oh]))[:, 1]
        return out


class Cox:
    """S1: elastic-net Cox model (scikit-survival) on aggregated features; monthly hazards from S(t)."""

    def __init__(self, alpha=0.05, H=6):
        self.alpha, self.H = alpha, H

    def fit(self, X, event_k, follow_k):
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv

        self.sc = StandardScaler().fit(X)
        t = np.where(event_k > 0, event_k, follow_k).astype(float)
        keep = t > 0
        y = Surv.from_arrays(event_k[keep] > 0, t[keep])
        self.m = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[self.alpha], fit_baseline_model=True, max_iter=5000)
        self.m.fit(self.sc.transform(X[keep]), y)
        return self

    def hazard(self, X):
        fns = self.m.predict_survival_function(self.sc.transform(X))
        S = np.array([[f(min(k, f.x[-1])) for k in range(1, self.H + 1)] for f in fns])
        S = np.clip(S, 1e-6, 1)
        prev = np.hstack([np.ones((len(X), 1)), S[:, :-1]])
        return np.clip(1 - S / prev, 0, 1)
