# CreditPulse — 44-Feature Behavioral Registry & Mathematical Dictionary

> **Model Input Matrix:** $[B, 12, 44]$ — 12 historical monthly tokens $\times$ 44 normalized behavioral banking features.  
> Every feature position is populated with continuous financial formulas. Zero-padding has been eliminated.

---

## Complete Feature Specification

| Index | Feature Name | Category | Mathematical Formula / Source | Banking Intuition & Distress Rationale |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `log_balance` | Liquidity | $\text{sign}(\text{bal}) \cdot \frac{\log(1 + \|\text{bal}\|)}{10}$ | Current ending liquidity scale normalized for wide currency distributions. |
| **1** | `log_min_balance` | Liquidity | $\text{sign}(b_{\min}) \cdot \frac{\log(1 + \|b_{\min}\|)}{10}$ | Intra-month minimum balance trough indicating depth of liquidity depletion. |
| **2** | `balance_to_loan_ratio` | Liquidity | $\text{bal} / (\text{loan\_amount} + 1)$ | Liquidity buffer relative to total outstanding principal obligation. |
| **3** | `negative_balance_flag` | Liquidity | $\mathbb{I}(b_{\min} < 0)$ | Binary flag indicating whether account penetrated below zero (overdraft state). |
| **4** | `days_below_zero_ratio` | Liquidity | $\text{days\_zero} / 30.0$ | Fraction of days in the month spent operating in sub-zero overdraft. |
| **5** | `severe_overdraft_flag` | Liquidity | $\mathbb{I}(\text{days\_zero} \ge 10)$ | Flag for chronic, severe overdraft (>10 days/month), a top default precursor. |
| **6** | `liquidity_runway_months`| Liquidity | $\max(0, \text{bal}) / (\text{outflow} + 1)$ | Estimated months of operational expenditure remaining from cash reserves. |
| **7** | `intra_month_drawdown` | Liquidity | $\max(0, \text{bal} - b_{\min}) / (\|\text{bal}\| + 1)$ | Magnitude of cash drawdown experienced between month-end and trough. |
| **8** | `log_inflow` | Cash Flow | $\log(1 + \max(0, \text{inflow})) / 10$ | Total operating revenue and cash receipts credited to account. |
| **9** | `log_outflow` | Cash Flow | $\log(1 + \max(0, \text{outflow})) / 10$ | Total monthly disbursements, operating expenditures, and withdrawals. |
| **10** | `net_cashflow_normalized`| Cash Flow | $\frac{\text{inflow} - \text{outflow}}{\text{inflow} + \text{outflow} + 10^{-4}}$ | Normalized cash-flow margin bounded in $[-1, +1]$; negative indicates cash burn. |
| **11** | `cash_burn_rate` | Cash Flow | $\max(0, \text{outflow} - \text{inflow}) / (\text{outflow} + 1)$ | Severity of operating deficit relative to monthly expenditure burn. |
| **12** | `inflow_to_outflow_ratio`| Cash Flow | $\text{inflow} / (\text{outflow} + 1)$ | Operating revenue coverage ratio; <1.0 denotes cash insufficiency. |
| **13** | `transaction_intensity` | Cash Flow | $n_{\text{tx}} / 50.0$ | Turnover velocity; sudden contraction suggests commercial disruption. |
| **14** | `avg_ticket_size` | Cash Flow | $\frac{\text{inflow} + \text{outflow}}{n_{\text{tx}} + 1} / 1000$ | Average transaction ticket size in thousands of CZK. |
| **15** | `debt_disbursement_share`| Cash Flow | $\text{pmt} / (\text{outflow} + 1)$ | Share of monthly disbursements absorbed by loan amortization. |
| **16** | `low_activity_flag` | Cash Flow | $\mathbb{I}(n_{\text{tx}} < 5)$ | Flag for dormant or abandoned operating accounts preceding default. |
| **17** | `severe_deficit_flag` | Cash Flow | $\mathbb{I}(\text{inflow} < 0.5 \cdot \text{outflow})$ | Major cash crunch flag: receipts cover less than 50% of expenditures. |
| **18** | `debt_service_to_inflow` | Debt Service | $\text{pmt} / (\text{inflow} + 1)$ | Debt service coverage ratio; >0.4 indicates heavy repayment strain. |
| **19** | `pmt_to_balance_ratio` | Debt Service | $\text{pmt} / (\|\text{bal}\| + 1)$ | Installment payment relative to reserve balance. |
| **20** | `pmt_to_loan_amount` | Debt Service | $\text{pmt} / (\text{loan\_amount} + 1)$ | Contractual monthly repayment rate relative to principal size. |
| **21** | `uncovered_payment_flag` | Debt Service | $\mathbb{I}(\text{inflow} < \text{pmt})$ | Flag indicating monthly cash receipts fail to cover the loan installment. |
| **22** | `balance_insufficient_pmt`| Debt Service | $\mathbb{I}(\text{bal} < \text{pmt})$ | Flag indicating ending balance cannot cover upcoming monthly payment. |
| **23** | `loan_seasoning_fraction`| Debt Service | $\min(1.0, \text{age} / 60.0)$ | Fraction of loan maturity elapsed; early defaults occur in months 6-18. |
| **24** | `remaining_term_fraction`| Debt Service | $\max(0.0, 1.0 - \text{age} / 60.0)$ | Remaining unseasoned horizon exposed to credit loss. |
| **25** | `high_leverage_flag` | Debt Service | $\mathbb{I}(\text{loan} > 4 \cdot \text{annual\_inflow})$ | Borrower leverage exceeds 4x annualized cash income. |
| **26** | `delta_balance_1m` | 1M Momentum | $(\text{bal}_t - \text{bal}_{t-1}) / (\|\text{bal}_{t-1}\| + 1)$ | Percentage change in ending balance from previous month. |
| **27** | `delta_inflow_1m` | 1M Momentum | $(\text{inf}_t - \text{inf}_{t-1}) / (\text{inf}_{t-1} + 1)$ | Revenue acceleration/contraction over previous month. |
| **28** | `delta_outflow_1m` | 1M Momentum | $(\text{out}_t - \text{out}_{t-1}) / (\text{out}_{t-1} + 1)$ | Disbursement expansion over previous month. |
| **29** | `delta_net_cashflow_1m` | 1M Momentum | $\text{net}_t - \text{net}_{t-1}$ | Direct change in normalized net cash margin. |
| **30** | `delta_days_zero_1m` | 1M Momentum | $(\text{days\_zero}_t - \text{days\_zero}_{t-1}) / 30$ | Monthly increase or decrease in overdraft days. |
| **31** | `consecutive_deficit_flag`| 1M Momentum | $\mathbb{I}(\text{net}_t < 0 \land \text{net}_{t-1} < 0)$ | Flag for at least two consecutive months burning operating cash. |
| **32** | `balance_depletion_velocity`| 1M Momentum | $\max(0, \text{bal}_{t-1} - \text{bal}_t) / (\|\text{bal}_{t-1}\| + 1)$ | Speed of reserve depletion during negative balance months. |
| **33** | `sudden_outflow_surge` | 1M Momentum | $\mathbb{I}(\text{out}_t > 1.5 \cdot \text{out}_{t-1})$ | Unexpected 50%+ spike in disbursements, draining operating runway. |
| **34** | `sudden_inflow_drop` | 1M Momentum | $\mathbb{I}(\text{inf}_t < 0.6 \cdot \text{inf}_{t-1})$ | 40%+ contraction in monthly sales/cash receipts. |
| **35** | `distress_acceleration_comp`| 1M Momentum | $0.4 \Delta \text{bal} + 0.3 \Delta \text{overdraft} + 0.3 \Delta \text{inflow}$ | Multi-dimensional momentum marker combining liquidity and cash flow. |
| **36** | `rolling_mean_balance_3m`| 3M Stability | $\text{mean}(\text{bal}_{t-2..t}) / 10000$ | 3-month smoothed liquidity buffer eliminating short-term timing noise. |
| **37** | `rolling_net_cashflow_3m`| 3M Stability | $\text{mean}(\text{net}_{t-2..t})$ | 3-month medium-term cash burn or surplus trend. |
| **38** | `balance_vs_3m_average` | 3M Stability | $(\text{bal}_t - \bar{B}_{3m}) / (\|\bar{B}_{3m}\| + 1)$ | Divergence of current balance from 3-month baseline. |
| **39** | `cum_days_zero_3m` | 3M Stability | $\sum_{i=t-2}^t \text{days\_zero}_i / 90$ | Cumulative overdraft exposure across the preceding quarter. |
| **40** | `inflow_stability_3m` | 3M Stability | $\min(\text{inf}_{3m}) / (\max(\text{inf}_{3m}) + 1)$ | Quarterly revenue volatility ratio; values near 1.0 indicate stable turnover. |
| **41** | `chronic_deficit_3m_flag`| 3M Stability | $\mathbb{I}(\text{all 3 months net} < 0)$ | Full-quarter cash burn: operating income continuously below costs. |
| **42** | `drawdown_from_3m_high` | 3M Stability | $\max(0, \max_{3m}(\text{bal}) - \text{bal}_t) / (\|\max_{3m}\| + 1)$ | Peak-to-trough balance drawdown over the rolling quarter. |
| **43** | `composite_liquidity_stress`| 3M Stability | Weighted composite of overdraft, cash deficit, and balance decline | Integrated behavioral distress index combining liquidity and solvency. |
