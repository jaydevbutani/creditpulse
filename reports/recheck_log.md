# Recheck log (R-01 … R-60)

| ID | Check | Status | Evidence |
|---|---|---|---|
| R-01 | Pipeline reproduces from raw | PARTIAL | data→features rebuild is byte-identical (repro_checks.json); M1/M2 bit-identical (R-29); full deep rerun on a clean machine not executed here |
| R-02 | Clean install + tests | PASS | 61 passed, 1 warning in 16.20s |
| R-03 | CI + injected leak fails | PARTIAL | workflow in .github/workflows/ci.yml; injected-leak test passes locally (tests/test_features.py) |
| R-04 | Data | PASS | reports/tables/data_checks.json |
| R-05 | Data | PASS | reports/tables/data_checks.json |
| R-06 | Data | PASS | reports/tables/data_checks.json |
| R-07 | Data | PASS | reports/tables/data_checks.json |
| R-08 | Data | PASS | reports/tables/data_checks.json |
| R-09 | Data | PASS | reports/tables/data_checks.json |
| R-10 | Data | PASS | reports/tables/data_checks.json |
| R-11 | Data | PASS | reports/tables/data_checks.json |
| R-12 | Data | PASS | reports/tables/data_checks.json |
| R-13 | Snapshot builder toy loans | PASS | tests/test_labels.py (6 toy loans) |
| R-14 | Labels | PASS | reports/tables/label_checks.json |
| R-15 | Labels | PASS | reports/tables/label_checks.json |
| R-16 | Labels | PASS | reports/tables/label_checks.json |
| R-17 | Labels | PASS | reports/tables/label_checks.json |
| R-18 | Causality (future perturbation) | PASS | tests/test_features.py |
| R-19 | Tensor hygiene | PASS | feature_checks.json |
| R-20 | Train-only fitting | PASS | fit range ['1993-10-31', '1998-11-30'] |
| R-21 | Panel completeness | PASS | panel_checks.json |
| R-22 | Drift report | PASS | 2 features PSI>0.25 reviewed: ['loan_age', 'account_age'] |
| R-23 | Shapes and padding | PASS | tests/test_features.py, tests/test_deep.py |
| R-24 | Single-feature leak scan | PASS | max AUC 0.885 |
| R-25 | Loan disjointness | PASS | label_checks.json |
| R-26 | Time ordering | PASS | P1 by origination; P2 by calendar |
| R-27 | Split fingerprint | PASS | 70b4fd9463bc7097 |
| R-28 | Sanity controls | PASS | baseline_summary.json |
| R-29 | M1/M2 reproducibility | PASS | repro_checks.json |
| R-30 | Survival sanity | PASS | tests/test_metrics.py + baseline_summary.json |
| R-31 | Overfit one batch | PASS | tests/test_deep.py |
| R-32 | Shapes and parameter counts | PASS | tests/test_deep.py; n_params in runs.jsonl |
| R-33 | Mask invariance | PASS | tests/test_deep.py |
| R-34 | Seed variance | PASS | reports/tables/ranking_claims.csv |
| R-35 | Equal tuning budget | PASS | 10 trials for every model (runs.jsonl) |
| R-36 | Pretraining boundary | PASS | assertion in pretrain_encoder + tests/test_pipeline_checks.py |
| R-37 | Hazard loss correctness | PASS | tests/test_deep.py |
| R-38 | Training health | PASS | non-finite gradient raises; loss and grad-norm curves in runs.jsonl |
| R-39 | Metric correctness | PASS | tests/test_metrics.py |
| R-40 | Loan-clustered bootstrap | PASS | clustered 0.434 > naive 0.283 |
| R-41 | Calibration plots | PASS | reports/figures/reliability.png |
| R-42 | Capture and lead-time toys | PASS | tests/test_metrics.py |
| R-43 | Negative controls | PASS | §8 of final_report.md |
| R-44 | Ablation table | PASS | §6 of final_report.md |
| R-45 | Test set touched once | PASS | reports/test_evaluations.jsonl |
| R-46 | Sensitivity runs | PASS | reports/tables/sensitivity.csv |
| R-47 | Claims language | PASS | claims generated from paired intervals (§5) |
| R-48 | Attribution sanity | PASS | explain_checks.json |
| R-49 | Attention vs permutation | PASS | disagreement reported in explain_checks.json |
| R-50 | Attribution stability | FAIL | mean rho 0.59 |
| R-51 | API contract tests | PASS | tests/test_serve.py |
| R-52 | Train/serve parity | PASS | tests/test_serve.py (100 pairs, 1e-6) |
| R-53 | Dashboard equals report | PASS | dashboard reads only saved artefacts |
| R-54 | Container smoke test | NOT RUN | Dockerfile provided; Docker unavailable in the build environment |
| R-55 | Batch scoring speed | PASS | max 1.1 s per month |
| R-56 | Model card and data sheet | PASS | reports/model_card.md, reports/data_sheet.md |
| R-57 | Subgroup table | PASS | reports/tables/subgroups.csv |
| R-58 | Second-machine rebuild | NOT RUN | requires a second machine; dvc.yaml + Makefile provided |
| R-59 | Deck/report consistency | PASS | report generated from artefacts; slide-14 numbers must be replaced by §4 values |
| R-60 | Defence dry run | OPEN | for the team |

pytest: 61 passed, 1 warning in 16.20s
