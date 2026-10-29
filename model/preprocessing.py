"""CreditPulse Feature Preprocessing and 44-Feature Behavioral Pipeline.
Constructs all 44 genuine, documented behavioral features from longitudinal monthly banking data.
Zero-padding is strictly eliminated: every position [0..43] represents a real, defined metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from model.model_config import INPUT_DIM, SEQUENCE_LENGTH

# ==============================================================================
# 44-FEATURE BEHAVIORAL REGISTRY
# ==============================================================================
FEATURE_REGISTRY: list[dict[str, Any]] = [
    # --- Category 1: Liquidity & Balance Levels (8 features) ---
    {
        "index": 0,
        "name": "log_balance",
        "category": "Liquidity",
        "formula": "sign(bal) * log1p(|bal|) / 10",
        "viva_meaning": "Current ending liquidity scale normalized for wide currency distributions."
    },
    {
        "index": 1,
        "name": "log_min_balance",
        "category": "Liquidity",
        "formula": "sign(bmin) * log1p(|bmin|) / 10",
        "viva_meaning": "Intra-month minimum balance trough indicating depth of liquidity depletion."
    },
    {
        "index": 2,
        "name": "balance_to_loan_ratio",
        "category": "Liquidity",
        "formula": "bal / (loan_amount + 1.0)",
        "viva_meaning": "Liquidity buffer relative to total outstanding principal obligation."
    },
    {
        "index": 3,
        "name": "negative_balance_flag",
        "category": "Liquidity",
        "formula": "1.0 if bmin < 0 else 0.0",
        "viva_meaning": "Binary flag indicating whether account penetrated below zero (overdraft state)."
    },
    {
        "index": 4,
        "name": "days_below_zero_ratio",
        "category": "Liquidity",
        "formula": "days_zero / 30.0",
        "viva_meaning": "Fraction of days in the month spent operating in sub-zero overdraft."
    },
    {
        "index": 5,
        "name": "severe_overdraft_flag",
        "category": "Liquidity",
        "formula": "1.0 if days_zero >= 10 else 0.0",
        "viva_meaning": "Flag for chronic, severe overdraft (>10 days/month), a top default precursor."
    },
    {
        "index": 6,
        "name": "liquidity_runway_months",
        "category": "Liquidity",
        "formula": "max(0, bal) / (outflow + 1.0)",
        "viva_meaning": "Estimated months of operational expenditure remaining from cash reserves."
    },
    {
        "index": 7,
        "name": "intra_month_drawdown",
        "category": "Liquidity",
        "formula": "max(0, bal - bmin) / (abs(bal) + 1.0)",
        "viva_meaning": "Magnitude of cash drawdown experienced between month-end and trough."
    },

    # --- Category 2: Cash Flow & Turnover Dynamics (10 features) ---
    {
        "index": 8,
        "name": "log_inflow",
        "category": "Cash Flow",
        "formula": "log1p(max(0, inflow)) / 10",
        "viva_meaning": "Total operating revenue and cash receipts credited to account."
    },
    {
        "index": 9,
        "name": "log_outflow",
        "category": "Cash Flow",
        "formula": "log1p(max(0, outflow)) / 10",
        "viva_meaning": "Total monthly disbursements, operating expenditures, and withdrawals."
    },
    {
        "index": 10,
        "name": "net_cashflow_normalized",
        "category": "Cash Flow",
        "formula": "(inflow - outflow) / (inflow + outflow + 1e-4)",
        "viva_meaning": "Normalized cash-flow margin bounded in [-1, +1]; negative indicates cash burn."
    },
    {
        "index": 11,
        "name": "cash_burn_rate",
        "category": "Cash Flow",
        "formula": "max(0, outflow - inflow) / (outflow + 1.0)",
        "viva_meaning": "Severity of operating deficit relative to monthly expenditure burn."
    },
    {
        "index": 12,
        "name": "inflow_to_outflow_ratio",
        "category": "Cash Flow",
        "formula": "inflow / (outflow + 1.0)",
        "viva_meaning": "Operating revenue coverage ratio; <1.0 denotes cash insufficiency."
    },
    {
        "index": 13,
        "name": "transaction_intensity",
        "category": "Cash Flow",
        "formula": "n_tx / 50.0",
        "viva_meaning": "Turnover velocity; sudden contraction suggests commercial disruption."
    },
    {
        "index": 14,
        "name": "avg_ticket_size",
        "category": "Cash Flow",
        "formula": "(inflow + outflow) / (n_tx + 1.0) / 1000.0",
        "viva_meaning": "Average transaction ticket size in thousands of CZK."
    },
    {
        "index": 15,
        "name": "debt_disbursement_share",
        "category": "Cash Flow",
        "formula": "pmt / (outflow + 1.0)",
        "viva_meaning": "Share of monthly disbursements absorbed by loan amortization."
    },
    {
        "index": 16,
        "name": "low_activity_flag",
        "category": "Cash Flow",
        "formula": "1.0 if n_tx < 5 else 0.0",
        "viva_meaning": "Flag for dormant or abandoned operating accounts preceding default."
    },
    {
        "index": 17,
        "name": "severe_deficit_flag",
        "category": "Cash Flow",
        "formula": "1.0 if inflow < 0.5 * outflow else 0.0",
        "viva_meaning": "Major cash crunch flag: receipts cover less than 50% of expenditures."
    },

    # --- Category 3: Debt Service & Credit Burden (8 features) ---
    {
        "index": 18,
        "name": "debt_service_to_inflow",
        "category": "Debt Service",
        "formula": "pmt / (inflow + 1.0)",
        "viva_meaning": "Debt service coverage ratio; >0.4 indicates heavy repayment strain."
    },
    {
        "index": 19,
        "name": "pmt_to_balance_ratio",
        "category": "Debt Service",
        "formula": "pmt / (abs(bal) + 1.0)",
        "viva_meaning": "Installment payment relative to reserve balance."
    },
    {
        "index": 20,
        "name": "pmt_to_loan_amount",
        "category": "Debt Service",
        "formula": "pmt / (loan_amount + 1.0)",
        "viva_meaning": "Contractual monthly repayment rate relative to principal size."
    },
    {
        "index": 21,
        "name": "uncovered_payment_flag",
        "category": "Debt Service",
        "formula": "1.0 if inflow < pmt else 0.0",
        "viva_meaning": "Flag indicating monthly cash receipts fail to cover the loan installment."
    },
    {
        "index": 22,
        "name": "balance_insufficient_for_pmt",
        "category": "Debt Service",
        "formula": "1.0 if bal < pmt else 0.0",
        "viva_meaning": "Flag indicating ending balance cannot cover upcoming monthly payment."
    },
    {
        "index": 23,
        "name": "loan_seasoning_fraction",
        "category": "Debt Service",
        "formula": "min(1.0, loan_age / 60.0)",
        "viva_meaning": "Fraction of loan maturity elapsed; early defaults occur in months 6-18."
    },
    {
        "index": 24,
        "name": "remaining_term_fraction",
        "category": "Debt Service",
        "formula": "max(0.0, 1.0 - loan_age / 60.0)",
        "viva_meaning": "Remaining unseasoned horizon exposed to credit loss."
    },
    {
        "index": 25,
        "name": "high_leverage_flag",
        "category": "Debt Service",
        "formula": "1.0 if loan_amount > 4.0 * max(1000.0, inflow * 12.0) else 0.0",
        "viva_meaning": "Borrower leverage exceeds 4x annualized cash income."
    },

    # --- Category 4: Month-over-Month Dynamics & Acceleration (10 features) ---
    {
        "index": 26,
        "name": "delta_balance_1m",
        "category": "1M Momentum",
        "formula": "(bal_t - bal_{t-1}) / (abs(bal_{t-1}) + 1.0)",
        "viva_meaning": "Percentage change in ending balance from previous month."
    },
    {
        "index": 27,
        "name": "delta_inflow_1m",
        "category": "1M Momentum",
        "formula": "(inf_t - inf_{t-1}) / (inf_{t-1} + 1.0)",
        "viva_meaning": "Revenue acceleration/contraction over previous month."
    },
    {
        "index": 28,
        "name": "delta_outflow_1m",
        "category": "1M Momentum",
        "formula": "(out_t - out_{t-1}) / (out_{t-1} + 1.0)",
        "viva_meaning": "Disbursement expansion over previous month."
    },
    {
        "index": 29,
        "name": "delta_net_cashflow_1m",
        "category": "1M Momentum",
        "formula": "net_t - net_{t-1}",
        "viva_meaning": "Direct change in normalized net cash margin."
    },
    {
        "index": 30,
        "name": "delta_days_zero_1m",
        "category": "1M Momentum",
        "formula": "(days_zero_t - days_zero_{t-1}) / 30.0",
        "viva_meaning": "Monthly increase or decrease in overdraft days."
    },
    {
        "index": 31,
        "name": "consecutive_deficit_flag",
        "category": "1M Momentum",
        "formula": "1.0 if net_t < 0 and net_{t-1} < 0 else 0.0",
        "viva_meaning": "Flag for at least two consecutive months burning operating cash."
    },
    {
        "index": 32,
        "name": "balance_depletion_velocity",
        "category": "1M Momentum",
        "formula": "max(0, bal_{t-1} - bal_t) / (abs(bal_{t-1}) + 1.0)",
        "viva_meaning": "Speed of reserve depletion during negative balance months."
    },
    {
        "index": 33,
        "name": "sudden_outflow_surge",
        "category": "1M Momentum",
        "formula": "1.0 if out_t > 1.5 * out_{t-1} else 0.0",
        "viva_meaning": "Unexpected 50%+ spike in disbursements, draining operating runway."
    },
    {
        "index": 34,
        "name": "sudden_inflow_drop",
        "category": "1M Momentum",
        "formula": "1.0 if inf_t < 0.6 * inf_{t-1} else 0.0",
        "viva_meaning": "40%+ contraction in monthly sales/cash receipts."
    },
    {
        "index": 35,
        "name": "distress_acceleration_composite",
        "category": "1M Momentum",
        "formula": "Composite of falling balance, expanding days zero, and falling inflow",
        "viva_meaning": "Multi-dimensional momentum marker combining liquidity and cash flow."
    },

    # --- Category 5: Medium-Term 3-Month Moving Trends & Volatility (8 features) ---
    {
        "index": 36,
        "name": "rolling_mean_balance_3m",
        "category": "3M Stability",
        "formula": "mean(bal_{t-2..t}) / 10000.0",
        "viva_meaning": "3-month smoothed liquidity buffer eliminating short-term timing noise."
    },
    {
        "index": 37,
        "name": "rolling_net_cashflow_3m",
        "category": "3M Stability",
        "formula": "mean(net_{t-2..t})",
        "viva_meaning": "3-month medium-term cash burn or surplus trend."
    },
    {
        "index": 38,
        "name": "balance_vs_3m_average",
        "category": "3M Stability",
        "formula": "(bal_t - mean_bal_3m) / (abs(mean_bal_3m) + 1.0)",
        "viva_meaning": "Divergence of current balance from 3-month baseline."
    },
    {
        "index": 39,
        "name": "cum_days_zero_3m",
        "category": "3M Stability",
        "formula": "sum(days_zero_{t-2..t}) / 90.0",
        "viva_meaning": "Cumulative overdraft exposure across the preceding quarter."
    },
    {
        "index": 40,
        "name": "inflow_stability_3m",
        "category": "3M Stability",
        "formula": "min(inf_{t-2..t}) / (max(inf_{t-2..t}) + 1.0)",
        "viva_meaning": "Quarterly revenue volatility ratio; values near 1.0 indicate stable turnover."
    },
    {
        "index": 41,
        "name": "chronic_deficit_3m_flag",
        "category": "3M Stability",
        "formula": "1.0 if all 3 recent months had net < 0 else 0.0",
        "viva_meaning": "Full-quarter cash burn: operating income continuously below costs."
    },
    {
        "index": 42,
        "name": "drawdown_from_3m_high",
        "category": "3M Stability",
        "formula": "max(0, max_bal_3m - bal_t) / (abs(max_bal_3m) + 1.0)",
        "viva_meaning": "Peak-to-trough balance drawdown over the rolling quarter."
    },
    {
        "index": 43,
        "name": "composite_liquidity_stress",
        "category": "3M Stability",
        "formula": "Normalized index of overdraft, cash deficit, and balance decline",
        "viva_meaning": "Integrated behavioral distress index combining liquidity and solvency."
    }
]

FEATURE_NAMES: list[str] = [f["name"] for f in FEATURE_REGISTRY]
assert len(FEATURE_NAMES) == 44, f"Expected 44 features, got {len(FEATURE_NAMES)}"


def extract_monthly_features(
    inflows: list[float],
    outflows: list[float],
    balances: list[float],
    min_balances: list[float],
    days_zero: list[int],
    loan_payments: list[float],
    n_txs: list[int],
    loan_amount: float = 150000.0,
    loan_ages: list[int] | None = None
) -> np.ndarray:
    """Computes genuine, normalized 44-dimensional feature vector for a 12-month sequence window.
    Every position from 0 to 43 is fully calculated using documented financial formulas.
    No zero-padding exists.
    """
    W = len(balances)
    token_features = np.zeros((W, INPUT_DIM), dtype=np.float32)

    for t in range(W):
        inf = float(inflows[t]) if t < len(inflows) else 0.0
        outf = float(outflows[t]) if t < len(outflows) else 0.0
        bal = float(balances[t]) if t < len(balances) else 0.0
        bmin = float(min_balances[t]) if t < len(min_balances) else 0.0
        dneg = int(days_zero[t]) if t < len(days_zero) else 0
        pmt = float(loan_payments[t]) if t < len(loan_payments) else 0.0
        ntx = int(n_txs[t]) if t < len(n_txs) else 0
        lage = int(loan_ages[t]) if loan_ages and t < len(loan_ages) else 12 + t

        # Category 1: Liquidity & Balance Levels (0..7)
        token_features[t, 0] = np.sign(bal) * np.log1p(abs(bal)) / 10.0
        token_features[t, 1] = np.sign(bmin) * np.log1p(abs(bmin)) / 10.0
        token_features[t, 2] = float(np.clip(bal / (loan_amount + 1.0), -2.0, 5.0))
        token_features[t, 3] = 1.0 if bmin < 0 else 0.0
        token_features[t, 4] = float(np.clip(dneg / 30.0, 0.0, 1.0))
        token_features[t, 5] = 1.0 if dneg >= 10 else 0.0
        token_features[t, 6] = float(np.clip(max(0.0, bal) / (outf + 1.0), 0.0, 12.0) / 12.0)
        token_features[t, 7] = float(np.clip(max(0.0, bal - bmin) / (abs(bal) + 1.0), 0.0, 5.0))

        # Category 2: Cash Flow & Turnover Dynamics (8..17)
        token_features[t, 8] = float(np.log1p(max(0.0, inf)) / 10.0)
        token_features[t, 9] = float(np.log1p(max(0.0, outf)) / 10.0)
        net_cf = (inf - outf) / (inf + outf + 1e-4)
        token_features[t, 10] = float(np.clip(net_cf, -1.0, 1.0))
        token_features[t, 11] = float(np.clip(max(0.0, outf - inf) / (outf + 1.0), 0.0, 1.0))
        token_features[t, 12] = float(np.clip(inf / (outf + 1.0), 0.0, 5.0) / 5.0)
        token_features[t, 13] = float(np.clip(ntx / 50.0, 0.0, 2.0))
        token_features[t, 14] = float(np.clip(((inf + outf) / (ntx + 1.0)) / 1000.0, 0.0, 10.0) / 10.0)
        token_features[t, 15] = float(np.clip(pmt / (outf + 1.0), 0.0, 1.0))
        token_features[t, 16] = 1.0 if ntx < 5 else 0.0
        token_features[t, 17] = 1.0 if inf < (0.5 * outf) else 0.0

        # Category 3: Debt Service & Credit Burden (18..25)
        token_features[t, 18] = float(np.clip(pmt / (inf + 1.0), 0.0, 2.0))
        token_features[t, 19] = float(np.clip(pmt / (abs(bal) + 1.0), 0.0, 2.0))
        token_features[t, 20] = float(np.clip(pmt / (loan_amount + 1.0), 0.0, 0.2))
        token_features[t, 21] = 1.0 if inf < pmt else 0.0
        token_features[t, 22] = 1.0 if bal < pmt else 0.0
        token_features[t, 23] = float(np.clip(lage / 60.0, 0.0, 1.0))
        token_features[t, 24] = float(np.clip(max(0.0, 1.0 - lage / 60.0), 0.0, 1.0))
        token_features[t, 25] = 1.0 if loan_amount > 4.0 * max(1000.0, inf * 12.0) else 0.0

        # Category 4: Month-over-Month Dynamics & Acceleration (26..35)
        if t > 0:
            prev_bal = float(balances[t-1])
            prev_inf = float(inflows[t-1])
            prev_out = float(outflows[t-1])
            prev_dneg = int(days_zero[t-1])
            prev_net = (prev_inf - prev_out) / (prev_inf + prev_out + 1e-4)

            d_bal = (bal - prev_bal) / (abs(prev_bal) + 1.0)
            d_inf = (inf - prev_inf) / (prev_inf + 1.0)
            d_out = (outf - prev_out) / (prev_out + 1.0)
            d_net = net_cf - prev_net
            d_dneg = (dneg - prev_dneg) / 30.0

            token_features[t, 26] = float(np.clip(d_bal, -2.0, 2.0))
            token_features[t, 27] = float(np.clip(d_inf, -2.0, 2.0))
            token_features[t, 28] = float(np.clip(d_out, -2.0, 2.0))
            token_features[t, 29] = float(np.clip(d_net, -2.0, 2.0))
            token_features[t, 30] = float(np.clip(d_dneg, -1.0, 1.0))
            token_features[t, 31] = 1.0 if (net_cf < 0 and prev_net < 0) else 0.0
            token_features[t, 32] = float(np.clip(max(0.0, prev_bal - bal) / (abs(prev_bal) + 1.0), 0.0, 2.0))
            token_features[t, 33] = 1.0 if outf > 1.5 * prev_out else 0.0
            token_features[t, 34] = 1.0 if inf < 0.6 * prev_inf else 0.0
            # Composite acceleration
            token_features[t, 35] = float(np.clip((0.4 * max(0.0, -d_bal) + 0.3 * max(0.0, d_dneg) + 0.3 * max(0.0, -d_inf)), 0.0, 2.0))
        else:
            token_features[t, 26:36] = 0.0

        # Category 5: Medium-Term 3-Month Moving Trends & Volatility (36..43)
        start_3m = max(0, t - 2)
        bals_3m = [float(balances[i]) for i in range(start_3m, t + 1)]
        infs_3m = [float(inflows[i]) for i in range(start_3m, t + 1)]
        outfs_3m = [float(outflows[i]) for i in range(start_3m, t + 1)]
        dnegs_3m = [int(days_zero[i]) for i in range(start_3m, t + 1)]

        mean_bal_3m = float(np.mean(bals_3m))
        nets_3m = [(infs_3m[k] - outfs_3m[k]) / (infs_3m[k] + outfs_3m[k] + 1e-4) for k in range(len(bals_3m))]
        mean_net_3m = float(np.mean(nets_3m))
        max_bal_3m = float(np.max(bals_3m))

        token_features[t, 36] = float(np.sign(mean_bal_3m) * np.log1p(abs(mean_bal_3m)) / 10.0)
        token_features[t, 37] = float(np.clip(mean_net_3m, -1.0, 1.0))
        token_features[t, 38] = float(np.clip((bal - mean_bal_3m) / (abs(mean_bal_3m) + 1.0), -2.0, 2.0))
        token_features[t, 39] = float(np.clip(sum(dnegs_3m) / 90.0, 0.0, 1.0))
        
        min_inf_3m = min(infs_3m)
        max_inf_3m = max(infs_3m)
        token_features[t, 40] = float(np.clip(min_inf_3m / (max_inf_3m + 1.0), 0.0, 1.0))
        token_features[t, 41] = 1.0 if all(n < 0 for n in nets_3m) else 0.0
        token_features[t, 42] = float(np.clip(max(0.0, max_bal_3m - bal) / (abs(max_bal_3m) + 1.0), 0.0, 2.0))
        
        # Composite 3M liquidity stress index
        stress_idx = (0.35 * token_features[t, 4] + 0.35 * (1.0 if mean_net_3m < 0 else 0.0) + 0.30 * (1.0 if bmin < 0 else 0.0))
        token_features[t, 43] = float(np.clip(stress_idx, 0.0, 1.0))

    return token_features
