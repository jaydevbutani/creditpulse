# CreditPulse – Temporal Distress Monitor

CreditPulse is a temporal credit-risk monitoring system that analyzes historical banking behavior to estimate loan distress risk, identify changes in financial behavior, and prioritize high-risk accounts for human review.

---

## 1. Overview

Traditional credit assessment often relies on static financial snapshots taken at loan origination or infrequent annual reviews. CreditPulse treats borrower credit health as an evolving temporal process, analyzing sequential monthly banking behavior to detect emerging financial distress before delinquency materializes.

CreditPulse is designed as an academic and research decision-support system for risk analysts and supervisory examiners. It does not automatically execute lending decisions, loan modifications, or regulatory accounting classifications; rather, it highlights deteriorating behavioral trajectories and ranks accounts for human investigation.

---

## 2. Problem Statement

Financial institutions and credit monitoring teams must detect deteriorating debt facilities early enough to intervene, restructure, or mitigate credit loss. Static scoring models frequently miss high-frequency behavioral deterioration—such as thinning liquidity cushions, sudden cash burn, escalating overdraft reliance, and payment-to-inflow compression.

CreditPulse formulates early warning as a multi-horizon temporal prediction problem over sequential account activity, producing calibrated distress probabilities and discrete monthly hazard trajectories to support risk officer prioritization.

---

## 3. Dataset & Data Provenance

CreditPulse uses the **Berka Financial Dataset** (also known as the **PKDD'99 Financial Dataset**), originally released for the European Conference on Principles and Practice of Knowledge Discovery in Databases (PKDD'99) Discovery Challenge in Prague, Czech Republic (1999).

- **Original Academic Source:** PKDD'99 Discovery Challenge, Prague, Czech Republic, 1999.
- **Historical Distribution:** Prague University of Economics and Business (Vysoká škola ekonomická v Praze – VŠE).
- **Historical Archive:** [http://lisp.vse.cz/pkdd99/DATA/data_berka.zip](http://lisp.vse.cz/pkdd99/DATA/data_berka.zip)
- **Historical Challenge Documentation:** [http://lisp.vse.cz/pkdd99/Challenge/berka.htm](http://lisp.vse.cz/pkdd99/Challenge/berka.htm)
- **Documentation Mirror:** [http://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm](http://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm)

*Note on Provenance:* CreditPulse uses this historical research dataset to construct the temporal representation used for modeling. The project does not claim original collection of this banking data.

---

## 4. Original Dataset Structure

The original Berka dataset is an anonymized relational database describing Czech bank accounts, transactions, permanent orders, credit cards, and granted loans between 1993 and 1998:

| Relation | Description | Scale |
| :--- | :--- | :--- |
| `account` | Static account characteristics, creation date, and district link | 4,500 accounts |
| `client` | Demographic information regarding bank customers | 5,369 clients |
| `disp` | Disposition rights linking clients to accounts (owner / user) | 5,369 relations |
| `order` | Standing recurring payment orders | 6,471 orders |
| `trans` | Transaction-level debit, credit, balance, and operation records | **1,056,320 transactions** |
| `loan` | Granted loans, terms, amounts, payments, and eventual repayment status | **682 loans** |
| `card` | Issued credit/debit cards | 892 cards |
| `district` | Socio-economic and regional demographic statistics | 77 districts |

> **Important Distinction:** The `trans` table contains **1,056,320 transaction records**, while the `loan` table contains **682 loan facilities**. Transactions reflect high-frequency ledger entries, not individual loans.

---

## 5. Project Data Preparation

To train temporal sequence architectures, the relational transactional ledger is transformed into a longitudinal monthly panel:

1. **Relation Linking:** Joins accounts, loans, dispositions, and transactions along historical timestamps.
2. **Monthly Aggregation:** Groups transaction streams into monthly accounting intervals per borrower.
3. **Behavioral Feature Engineering:** Computes financial indicators covering cash flow, liquidity floor, debt burden, volatility, and trajectory.
4. **Temporal Windowing:** Formulates a rolling **12-month sequence** of monthly behavioral observations for each monitoring cutoff.
5. **Feature Representation:** Each monthly observation vector comprises **44 engineered behavioral features**.

The final model input tensor has shape:
$$\text{Input Shape} = 12 \times 44 \quad (\text{12 consecutive months} \times \text{44 behavioral features})$$

> **Important:** The $12 \times 44$ matrix is the project's engineered temporal representation; it is not the raw format in which the Berka relational tables were distributed.

---

## 6. Feature Groups

The 44 engineered monthly features are structured across five financial behavioral groups:

1. **Liquidity & Balance Levels:** End-of-month balance (`bal_eom`), minimum monthly liquidity floor (`bal_min`), days spent in overdraft (`days_below_zero`), and liquidity buffer relative to loan installment.
2. **Cash Flow & Turnover Dynamics:** Total monthly cash inflows (`inflow`), total monthly disbursements (`outflow`), net cash flow (`net_flow`), transaction count (`n_tx`), and withdrawal intensity.
3. **Debt Service & Credit Burden:** Monthly loan installment (`loan_payment`), cumulative loan payments cleared (`uver_paid`), payment-to-inflow ratio, and debt service coverage strain.
4. **Month-over-Month Momentum:** 1-month rate of change in balance ($\Delta\text{bal}$), inflow acceleration, outflow surge, and minimum balance trajectory.
5. **Medium-Term 3M Stability:** 3-month rolling mean and standard deviation of balance, 3-month inflow trend, 3-month cash burn coefficient, and quarterly volatility ratios.

---

## 7. Model Zoo

CreditPulse implements and benchmarks an 11-model taxonomy evaluated under identical temporal splits:

| Model ID | Architecture / Family | Primary Purpose & Role |
| :--- | :--- | :--- |
| **M0_prior** | Empirical Prior Baseline | Reference baseline using empirical historical default rate |
| **M1_logreg** | Calibrated Logistic Regression | Interpretable linear baseline over aggregated temporal summaries |
| **M2_lgbm** | LightGBM Gradient Boosting | High-performance tabular gradient boosted decision tree |
| **M2_xgb** | XGBoost Gradient Boosting | Boosted tree comparison evaluating non-linear tabular interactions |
| **S1_cox** | Cox Proportional Hazards | Classical semi-parametric survival analysis baseline |
| **S2_hazard** | Neural Time-to-Event MLP | Deep survival baseline estimating discrete interval hazard rates |
| **M3_gru** | Gated Recurrent Unit (GRU) | Recurrent neural baseline modeling temporal token transitions |
| **M4_lstm** | Long Short-Term Memory (LSTM) | Recurrent sequence baseline with explicit memory gating |
| **M5_transformer** | Single-Task Transformer | Pre-LN Transformer encoder trained solely on 6-month binary distress |
| **M6_multitask** | Multi-Task Transformer (**Champion**) | Joint Pre-LN Transformer predicting 6M distress and discrete monthly hazard trajectory |
| **M7_pretrained** | Masked Pretrained Transformer | Self-supervised temporal sequence pretraining followed by fine-tuning |

---

## 8. Final Model — M6 MultiTask Transformer

The served champion model is **M6_multitask**, a Pre-Layer Normalization (Pre-LN) Transformer operating on 12-month behavioral sequence tokens.

### Architecture Specifications (`ml_models/M6_multitask/config.py`, `model/model_config.py`):
- **Input Dimension:** $12 \times 44$ ($T = 12$ months, $D = 44$ features per month)
- **Token Embedding:** Linear projection ($44 \to 32$) + Learnable `[CLS]` sequence token + Positional & Age Embeddings
- **Encoder Blocks:** 3 Pre-LN Transformer encoder layers
- **Hidden Dimension ($d_{\text{model}}$):** 32
- **Self-Attention Heads:** 2 heads ($d_k = 16$)
- **Feed-Forward Dimension:** 64 ($2 \times d_{\text{model}}$)
- **Activation Function:** GELU
- **Dropout Rate:** 0.20
- **Pooling Strategy:** `[CLS]` token representation ($d = 32$)
- **Total Parameters:** 33,383 (100% trainable)

### Dual-Head Multi-Task Prediction:
- **Head A (Binary Distress):** Linear projection ($32 \to 1$) outputting logit for calibrated 6-month distress probability.
- **Head B (Discrete Monthly Hazard):** Linear projection ($32 \to 6$) outputting discrete hazard rates across forward horizons $+1\text{m}, +2\text{m}, +3\text{m}, +4\text{m}, +5\text{m}, +6\text{m}$.

### Multi-Task Loss Formulation:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{distress}} + 0.30 \times \mathcal{L}_{\text{hazard}}$$
Where $\mathcal{L}_{\text{distress}}$ is weighted binary cross-entropy ($\text{pos\_weight} = 8.622$ to account for class imbalance), and $\mathcal{L}_{\text{hazard}}$ is discrete-time survival negative log-likelihood.

---

## 9. Evaluation

Models are evaluated Out-of-Time on the held-out test split (P1, 1998) under real-world default class imbalance (~4% prevalence).

### Primary Metrics:
- **PR-AUC (Precision-Recall Area Under Curve):** Evaluates minority positive (distress) detection without inflation from true negatives.
- **ROC-AUC (Receiver Operating Characteristic AUC):** Evaluates pairwise ranking discrimination between distressed and non-distressed facilities across all thresholds.
- **Brier Score:** Mean squared error between predicted probabilities and actual binary outcomes. Lower is better.
- **ECE (Expected Calibration Error):** Measures alignment between predicted confidence bins and empirical event frequency after Platt scaling. Lower is better.
- **Capture @ Top 10%:** Fraction of total distressed loans captured within the 10% highest-risk queue positions. Higher is better.

---

## 10. System Architecture

```text
Berka Banking Relational Data (1993–1998)
                  │
                  ▼
Data Preparation & Monthly Aggregation
                  │
                  ▼
Feature Engineering (5 Behavioral Groups)
                  │
                  ▼
12-Month Temporal Sequence Windows [12 × 44]
                  │
                  ▼
Model Zoo Evaluation (11 Architectures)
                  │
                  ▼
M6 Multi-Task Transformer (Pre-LN Encoder)
                  │
         ┌────────┴────────┐
         ▼                 ▼
   Head A Logit      Head B Hazards
   (6M Distress)     (+1m to +6m)
         │                 │
         └────────┬────────┘
                  ▼
Calibrated Decision Support & Early Warning
                  │
  ┌───────────────┼───────────────┐
  ▼               ▼               ▼
Portfolio     Loan Detail &    Supervisory
 Monitor      What-If Scenarios Audit Trail
```

---

## 11. Application

The CreditPulse web interface provides interactive decision support across dedicated functional modules:

- **Portfolio Monitor (`/`):** Ranked supervisory review queue, dynamic multi-month cutoff filter, two-period comparison drawer (Period A vs Period B drift diagnostics), and live record streaming simulation.
- **Loan Detail (`/loan-detail`):** Single-facility deep dive displaying calibrated 6-month risk, percentile queue urgency, 8-month historical trajectory, 6-month discrete forward hazard timing, Integrated Gradients behavioral drivers, and interactive what-if cash-flow stress testing.
- **Upload Portfolio (`/upload`):** Batch ingestion pipeline supporting CSV and ZIP loan books with automatic schema validation, feature extraction, and dual-head inference. Includes pre-loaded demonstration cohorts.
- **Model Health (`/model-health`):** Comprehensive transparency dashboard presenting the 11-model benchmark zoo table, multi-metric comparison charts (PR-AUC, ROC-AUC, Brier score, Capture@10%), and full M6 layer specifications.
- **Process New Record:** 4-step live pipeline test simulating real-time ledger updates (ingestion $\to$ storage $\to$ 12M sequence inference $\to$ risk update).
- **Human Review & Audit Trail (`/history`):** Human loan officer review workflow enabling analysts to log review notes, assign supervisory concern flags (Watch / Concern / Critical), and maintain an auditable event ledger.

---

## 12. Repository Structure

```text
creditpulse/
├── backend/                  # FastAPI REST API, database isolation, ML engine
│   ├── main.py               # API endpoints, middleware, routing
│   ├── database.py           # SQLite database layer, audit logging, schema seeding
│   └── ml_engine.py          # Pipeline artifact loading, inference, what-if engine
├── frontend/                 # React 18 + TypeScript + Vite + Tailwind CSS UI
│   ├── src/
│   │   ├── pages/            # PortfolioDashboard, LoanDetail, UploadCase, ModelHealth, etc.
│   │   ├── components/       # Navigation header, telemetry ticker, UI widgets
│   │   ├── api.ts            # Typed REST client
│   │   └── types.ts          # TypeScript domain interfaces
│   ├── package.json
│   └── vite.config.ts
├── model/                    # Standalone core DL model package (Viva & Code Inspection)
│   ├── model_config.py       # Configuration parameters and hyperparameters
│   ├── architecture.py       # MultiTaskDistressTransformer PyTorch implementation
│   ├── train.py              # Self-contained training script
│   ├── evaluate.py           # Evaluation script and metric computations
│   ├── inference.py          # Standalone inference demonstration
│   └── explainability.py     # Attribution and driver extraction
├── ml_models/                # Full 11-Model Benchmark Zoo
│   ├── M0_prior/             # Empirical prior baseline
│   ├── M1_logreg/            # Logistic regression baseline
│   ├── M2_lgbm/              # LightGBM boosted tree baseline
│   ├── M2_xgb/               # XGBoost boosted tree baseline
│   ├── S1_cox/               # Cox survival baseline
│   ├── S2_hazard/            # Neural hazard survival model
│   ├── M3_gru/               # GRU sequence baseline
│   ├── M4_lstm/              # LSTM sequence baseline
│   ├── M5_transformer/       # Single-task transformer
│   ├── M6_multitask/         # Champion multi-task transformer
│   ├── M7_pretrained/        # Masked sequence pretrained transformer
│   ├── common/               # Shared metrics, datasets, loss functions
│   └── MODEL_CONFIGS.md      # Comprehensive architecture comparison table
├── documentation/            # Academic documentation & viva presentation guides
│   ├── DL_CODE_MAP.md        # Deep learning code map
│   ├── FEATURE_DICTIONARY.md # 44-dimensional feature dictionary
│   ├── MODEL_ARCHITECTURE.md # Detailed layer and tensor mathematical flow
│   └── VIVA_GUIDE.md         # Teacher examination and viva defense questions
├── app/                      # Streamlit auxiliary dashboard (headless port 8505)
│   └── dashboard.py
├── sample_data/              # Pre-configured demonstration cohorts for 1-click evaluation
├── tests/                    # Comprehensive automated pytest test suite (61 tests)
├── reports/                  # Evaluation tables, baseline summaries, and figures
├── README.md                 # Primary project documentation
└── .gitignore                # Production public repository gitignore
```

---

## 13. Running the Project

### Prerequisites:
- Python 3.10+
- Node.js 18+ and npm

### 1. Backend Service
```bash
# Navigate to project directory and activate environment
cd creditpulse
.\venv\Scripts\activate   # Windows (or: source venv/bin/activate on Unix)

# Start FastAPI server on port 8000
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
*Backend URL:* `http://127.0.0.1:8000`

### 2. Frontend Application
```bash
# Navigate to frontend directory and install dependencies if not present
cd frontend
npm install

# Start Vite development server on port 5173
npm run dev -- --host 127.0.0.1
```
*Frontend URL:* `http://127.0.0.1:5173`

### 3. Streamlit Auxiliary Service (Optional)
```bash
# In project root
streamlit run app/dashboard.py --server.port 8505 --server.headless true
```
*Streamlit URL:* `http://localhost:8505`

### 4. Running Model Verification Commands
```bash
# Verify M6 Transformer architecture tensor shapes and parameters
python -m model.architecture

# Run standalone model inference
python -m model.inference

# Run test suite
pytest
```

---

## 14. Limitations

1. **Historical Banking Data:** The empirical analysis is based on historical Czech banking records (PKDD'99) reflecting 1993–1998 retail banking structures and macroeconomic conditions.
2. **Academic & Research Prototype:** CreditPulse is designed for academic demonstration, algorithmic research, and educational examination. It does not replace institutional credit committee governance.
3. **Decision Support Boundary:** Model predictions and discrete hazard shapes are advisory indicators designed to prioritize accounts for human review, not to execute automated debt enforcement or statutory default recognition.
4. **External Generalization:** Application to modern banking books requires re-calibrating to contemporary portfolio credit distributions, accounting frameworks (CECL / IFRS 9), and institutional core banking schemas.

---

## 15. License & Dataset Note

- **Project Software:** Source code and architecture implementations developed for CreditPulse are provided for research, academic evaluation, and educational demonstration.
- **Dataset Provenance:** The Berka PKDD'99 dataset is an academic research dataset released by the PKDD'99 Discovery Challenge committee. Users wishing to reconstruct the raw database from scratch should consult the official PKDD'99 challenge archive at VŠE.
