# Data sheet — Berka / PKDD'99 financial dataset as used by CreditPulse

- **Source:** PKDD'99 Discovery Challenge financial dataset (Czech bank, 1993–1998), as distributed by the CTU
  relational dataset repository.
- **Tables:** account (4,500), loan (682), trans (1,056,320), district (77), client (5,369), disp (5,369),
  card (892), order (6,471).
- **Loan status (R-05):** A = finished, no problems; B = finished, not paid; C = running, OK so far;
  D = running, client in debt. Working label: bad = B or D.
- **Cleaning rules:**
  - C1: dates are YYMMDD integers in the 19xx century.
  - C2: VYBER type is a debit (sign verified, 100.00%).
  - C3: amounts are signed.
  - C4: null and blank `k_symbol` both map to NONE.
  - C5: null `operation` means INTEREST (exactly the UROK rows).
  - C6: counterparty ids are not features; only presence and distinct-counterparty counts are used.
  - C7: rows are sorted by account and date, and same-day rows are re-chained by balance.
  - C8: negative balances are kept and signed-log transformed.
  - C9: payout credits (the 5 that exist) are excluded from inflow.
  - C10: features are winsorised at the 1st/99th percentiles of training tokens only.
- **Event definition (D3):** T is the earlier of (a) the first SANKC. UROK (penalty interest) row after
  origination, or (b) the first scheduled instalment month with UVER paid below 50% of the instalment. The
  schedule starts in the origination month if a UVER debit appears there, otherwise in the next month.
- **Derived attributes:**
  - `gender` and `birth_date` are decoded from `birth_number` (month + 50 means female). They are used only for
    the fairness table.
  - Card issue dates are used causally.
  - The `order` table is excluded because it is undated.
- **Translations:** see `reports/tables/data_dictionary.json`.
