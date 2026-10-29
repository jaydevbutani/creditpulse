"""R-29: M1 and M2 reproduce across two clean runs and M1 matches a plain scikit-learn run of the same features."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from creditpulse.models.baselines import make_lgbm, make_lr
from creditpulse.train.data import load_bundle
from creditpulse.utils import load_config, p

cfg = load_config()
B = load_bundle(cfg)
tr = B.idx("train")
te = B.idx("test")
C = json.loads([l for l in (p(cfg, "reports") / "runs.jsonl").read_text().splitlines() if '"tune_M1"' in l][-1])["params"]["best_params"]["C"]
a = make_lr(C).fit(B.A[tr], B.y[tr]).predict_proba(B.A[te])[:, 1]
b = make_lr(C).fit(B.A[tr], B.y[tr]).predict_proba(B.A[te])[:, 1]
sc = StandardScaler().fit(B.A[tr])
plain = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(B.A[tr]), B.y[tr])
c = plain.predict_proba(sc.transform(B.A[te]))[:, 1]
prm = json.loads((p(cfg, "artifacts") / "M2_lgbm_params.json").read_text())
g1 = make_lgbm(prm, 0).fit(B.A[tr], B.y[tr]).predict_proba(B.A[te])[:, 1]
g2 = make_lgbm(prm, 0).fit(B.A[tr], B.y[tr]).predict_proba(B.A[te])[:, 1]
import pandas as pd
off = pd.read_parquet(p(cfg, "artifacts") / "preds" / "M2_lgbm_s0.parquet").score_raw.values[te]
res = {"R-29": {"pass": bool(np.max(np.abs(a - b)) < 1e-9 and np.max(np.abs(a - c)) < 1e-6 and
                             np.max(np.abs(g1 - g2)) < 1e-9 and np.max(np.abs(g1 - off)) < 1e-9),
                "m1_run_to_run": float(np.max(np.abs(a - b))), "m1_vs_plain_sklearn": float(np.max(np.abs(a - c))),
                "m2_run_to_run": float(np.max(np.abs(g1 - g2))), "m2_vs_pipeline_artifact": float(np.max(np.abs(g1 - off)))}}
(p(cfg, "reports") / "tables" / "repro_checks.json").write_text(json.dumps(res, indent=2))
print(res)
