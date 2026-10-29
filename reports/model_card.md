# Model card — CreditPulse M6 (multi-task Transformer, served ensemble)

- **Version:** `M6_multitask-ens5-cfg29476ad6e2-data1a2c09-f49070-9277f1`
- **Intended use:** monthly early-warning ranking of active loans for human credit analysts.
- **Out of scope:** automated approval, rejection, pricing or intervention; populations unlike 1990s Czech
  retail banking.
- **Inputs:** 12 monthly behaviour tokens × 83 features (cash flow, balance, activity, channel mix, obligations,
  loan servicing, distress flags, dynamics, loan and account static features, seasonality, masks). Data up to
  the month-end cutoff only.
- **Outputs:**
  - `risk_6m`: Platt-calibrated probability of a distress event in the next 6 months.
  - `hazard`: monthly hazards h1–h6, uncalibrated and exploratory.
  - Queue rank and percentile.
  - Top-3 reason codes.
- **Architecture:** pre-LN Transformer encoder with CLS pooling, learned month-position and loan-age embeddings,
  and a padding mask in every layer. Head A is a sigmoid; Head B is 6 sigmoids. Loss is BCE(pos_weight) +
  λ·discrete-time hazard NLL. Hyperparameters: `{"lr": 0.00021775224101934097, "wd": 0.00014254757044027455, "d_model": 32, "n_layers": 3, "dropout": 0.2616240759128834, "pos_weight": 8.622121759478738, "n_heads": 2, "lam": 0.3}`.
- **Training data:** 305 loans originated through
  1996-12-31, with 6916 labelled monthly
  snapshots and 147 positive snapshots. Validation for calibration covers
  loans originated through 1997-06-30.
- **Evaluation:** the out-of-time test covers loans originated from 1997-07 onward, with
  9 event loans. Test PR-AUC is 0.240 [0.097, 0.493]
  and ROC-AUC is 0.915 [0.852, 0.962]. Brier is
  0.0378 (skill 0.02), ECE is 0.0421, and capture@10% is
  0.78. See `final_report.md` §4–§7.
- **Calibration:** Platt scaling on validation (isotonic was not used because validation has too few positives).
  Reliability curves before and after calibration are in `reports/figures/reliability.png`.
- **Known failure modes:**
  - It cannot see events in the first 3 months of a loan.
  - Signal is dominated by persistent balance level, so financially thin but reliable payers may be over-flagged.
  - Drift on age features (PSI above 0.25).
  - Head B is weakly identified.
- **Fairness:** gender and age are not inputs. The subgroup table (§12) shows performance by region, amount band,
  duration, gender and age band, with notes where a subgroup has too few events to judge.
- **Reproducibility:**
  - `make all` runs from the raw CSVs. Every run logs the config hash, data version, environment and metrics to
    `reports/runs.jsonl`.
  - Test evaluations are logged once per final model in `reports/test_evaluations.jsonl`.
  - Seeds are [0, 1, 2, 3, 4].
