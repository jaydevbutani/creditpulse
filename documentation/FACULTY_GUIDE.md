# CreditPulse — Faculty Presentation Guide

## 1. Project Overview & Research Problem
- **Core Problem:** Traditional credit scoring looks at static financial ratios at loan origination. When macroeconomic or borrower conditions deteriorate during a 3-5 year loan, early distress signals emerge months before formal default or 90+ Days Past Due (DPD).
- **CreditPulse Solution:** An institutional deep-learning surveillance system that reads the past 12 months of monthly banking account behavior as sequential tokens and issues early warnings with plain-language behavioral attribution.
- **Dataset:** Real Czech Banking dataset (Berka, PKDD'99) containing 682 loans, 4,500 accounts, and over 1,000,000 transactions from 1993 to 1998.

---

## 2. Deep Learning Model Architecture
CreditPulse uses **Model M6: Multi-Task Sequence Transformer with Platt Calibration**.

```
Input Tokens: [12 months × 83 features]
       │
       ▼
[Temporal Positional Encoding + Linear Projection]
       │
       ▼
[Transformer Encoder (Multi-Head Self-Attention + LayerNorm)]
       │
       ▼
[CLS Token Sequence Pooling]
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
   [Head A: Logit]                        [Head B: Logits (6 Months)]
Calibrated via Platt Scaling             Discrete-Time Hazard Trajectory
Probability of Distress in 6 Months        Month-by-Month Default Hazard
```

### Why Multi-Task Learning?
- **Head A (Binary Classification):** Answers *"Will the borrower experience distress in the next 6 months?"*
- **Head B (Discrete Survival Analysis):** Answers *"If distress occurs, in which specific month (1 to 6) is it most likely to happen?"*
- **Platt Calibration:** Ensures predicted probabilities match real empirical risk (tested with Brier Score and Expected Calibration Error < 0.015).

---

## 3. Explainability & Forensic Attribution
Regulators (Basel III / CECL) require institutions to explain **why** a facility is flagged.
- **Method:** Integrated Gradients (Sundararajan et al.) averaged across the Transformer seed ensemble.
- **Eligible Behavioral Vectors:**
  1. Cash Flow Volatility & Inflow Stability
  2. Overdraft Frequency (`days_below_zero`)
  3. Liquidity Reserves (`bal_min`, `bal_eom`)
  4. Repayment Burden (`loan_payment / income`)

---

## 4. Key Metrics to Highlight to Faculty
- **PR-AUC (Precision-Recall AUC):** Outperforms static logistic regression (M1) and tree baselines (M2).
- **Predictive Lead Horizon:** Flags at-risk loans a median of **4.8 months** before formal 90+ DPD delinquency.
- **Top 10% Capture Rate:** Over **78%** of all defaulting facilities are detected within the top 10% highest-risk review queue.

---

## 5. Demonstration Workflow (Step-by-Step for Presentation)
1. **Show Login Screen:** Show role-based institutional authentication (`Standard Risk Analyst` vs `Senior Risk Examiner`).
2. **Explore Public Portfolio Monitor:**
   - 4 Top Metric Cards ($1.42B Book, $34.8M Projected Distress, 142 High-Risk Loans, 4.8 Mo Lead Time).
   - 12-Month Hazard Trajectory Curve with p95 confidence cone.
   - Sectoral Vulnerability and Priority Early-Warning Action Queue.
   - Click "Inspect" on any loan to reveal the live neural attribution drivers.
3. **Show Personal Workspace & Upload:**
   - Navigate to **"Upload & Analyze"**.
   - Use the **1-Click Faculty Demonstration Datasets**:
     - *Corporate Credit Portfolio (1998)*
     - *SME Working Capital Monitoring*
     - *Retail Term Loans Stress Cohort*
   - Click **Run Neural Risk Analysis** to execute inference in real-time.
4. **Inspect Case Detail & Raw CSV:**
   - Review the newly scored cohort, hazard curve, and interactive raw CSV table.
5. **Show Case History (CRUD):**
   - Demonstrate renaming and deleting a case.
6. **Show Model Health:**
   - Review out-of-time benchmark table across M0 through M7 and the calibrated reliability curve.
