# CreditPulse Teacher & Demo Synthetic Loan Datasets

**Files Available for Direct Upload from File Explorer:**
- `demo_dataset/creditpulse_teacher_demo.csv`
- `demo_dataset/creditpulse_teacher_demo.zip`
- `demo_dataset/creditpulse_demo_loans.csv`
- `demo_dataset/creditpulse_demo_loans.zip`

---

## 1. Overview & Context

This dataset is a **deterministic, realistic synthetic cohort of 150 commercial loan facilities** structured specifically for academic presentations, classroom walkthroughs, and teacher evaluations of the **CreditPulse Temporal Distress Monitor**.

* **Synthetic Data Disclosure:** This dataset is 100% synthetic and generated following the empirical financial ratios of the Czech Banking (Berka / PKDD'99) longitudinal study. It does **not** contain proprietary or real bank customer data.
* **Facilities Count:** 150 unique facilities across 6 distinct cohorts (25 facilities per cohort).
* **Observation Horizon:** 12-month temporal sequence with current evaluation cutoff month `1998-12`.
* **Reproducibility:** Seeded with fixed random initialization (`seed=42`) for 100% deterministic reproducibility.

---

## 2. Cohort Taxonomy (6 Distinct Behavioral Profiles)

| Cohort | Facilities | Target Default | Behavioral Pattern |
| :--- | :---: | :---: | :--- |
| **1. Stable Performing** | 25 | `0` (Normal) | Healthy liquidity ($>150\text{k}$ CZK), $0$ days below zero, inflow exceeds debt service by $2.5\times$ to $4\times$. |
| **2. Gradual Deterioration** | 25 | `1` (Distress) | Outflow exceeds inflow for 3+ consecutive months; minimum balance steadily declines into negative territory. |
| **3. Acute Liquidity Crisis** | 25 | `1` (Distress) | Multiple overdraft triggers ($>5$ days below zero), severe negative liquidity, repayment failure risk. |
| **4. Cash-Flow Volatility** | 25 | `0` (Normal) | High transaction volume variance with seasonal swings, but underlying liquidity recovers before cycle close. |
| **5. Human Flagged (Override)** | 25 | `0` (Normal) | Model predicts benign risk ($<2.85\%$), but credit risk examiner flags based on qualitative supply-chain exposure. |
| **6. Model Alert (Pending Review)**| 25 | `1` (Distress) | Transformer predicts elevated risk ($\ge 2.85\%$) from sequence velocity before an examiner has reviewed it. |

---

## 3. Column Catalog & Schema Specification

| Column | Data Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `loan_id` | `INTEGER` | ID | Unique facility identifier (`#7001` to `#7150`) |
| `account_id` | `INTEGER` | ID | Operating bank account linked to loan facility |
| `borrower_name` | `STRING` | Entity | Synthetic commercial borrower name |
| `month` | `STRING` | Temporal | As-of observation period (`1998-12`) |
| `sector` | `STRING` | Industry | Industry sector classification |
| `loan_amount` | `FLOAT` | Leverage | Original sanctioned principal amount in CZK |
| `balance` | `FLOAT` | Feature | Current end-of-month ledger balance |
| `balance_min` | `FLOAT` | Feature | Lowest intra-month liquidity trough |
| `inflow` | `FLOAT` | Feature | Total monthly operating cash credits |
| `outflow` | `FLOAT` | Feature | Total monthly operating cash debits |
| `transaction_count` | `INTEGER` | Feature | Total account transactions in 30-day window |
| `payment_amount` | `FLOAT` | Feature | Scheduled monthly amortization installment |
| `days_below_zero` | `INTEGER` | Feature | Cumulative days operating account balance $< 0$ |
| `loan_age` | `INTEGER` | Seasoning | Months elapsed since loan origination |
| `dpd_status` | `STRING` | Credit | Current Days Past Due payment status |
| `cohort` | `STRING` | Metadata | Evaluation cohort group |
| `target` / `default_indicator`| `INTEGER` | Ground Truth | `1` if facility experienced default within next 6M |

---

## 4. How to Demonstrate to a Teacher / Examiner

1. Open **Windows File Explorer** and navigate to `creditpulse/demo_dataset/`.
2. Locate either:
   - `creditpulse_teacher_demo.csv`
   - or `creditpulse_teacher_demo.zip`
3. Drag the file directly from File Explorer into the **Upload & Analyze** page in CreditPulse (or click *Browse Files*).
4. Observe the automated multi-stage ingestion pipeline:
   - `✓ Dataset loaded`
   - `✓ 150 loans detected`
   - `✓ 12-month behavioural sequence detected`
   - `✓ Required features available`
   - `✓ Missing values checked`
   - `✓ Model ready`
5. Click **Run Neural Risk Analysis** to execute `M6_multitask` Transformer scoring, generating portfolio distributions, hazard curves, and Integrated Gradients driver attributions.
