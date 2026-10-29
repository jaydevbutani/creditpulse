// API type definitions — derived from confirmed backend response schemas.
// Do not add fields that haven't been verified from the actual FastAPI responses.

// ─── /health ────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;           // "ok"
  model_version: string;
  data_end: string;         // "1998-12-31"
  loans: number;            // 682
}

// ─── /loans ─────────────────────────────────────────────────────────────────

export interface Loan {
  loan_id: number;
  account_id: number;
  origination: string;      // "YYYY-MM-DD"
  amount: number;           // CZK
  duration: number;         // months
}

// ─── /queue ─────────────────────────────────────────────────────────────────

export interface QueueEntry {
  loan_id: number;
  queue_rank: number;
  risk_6m: number;          // calibrated probability [0,1]
  queue_percentile: number; // 0 = highest risk
  loan_age: number;         // months since origination
}

export interface QueueResponse {
  as_of_date: string;
  model_version: string;
  queue: QueueEntry[];
}

// ─── /score ─────────────────────────────────────────────────────────────────

export interface ScoreRequest {
  loan_id: number;
  as_of_date: string; // "YYYY-MM-DD"
}

export interface Driver {
  feature: string;
  contribution: number;
  value: number;
  text: string;
}

export interface Window {
  first_month: string;  // "YYYY-MM"
  last_month: string;
}

export type ScoreStatus = 'scored' | 'too_new' | 'in_distress' | 'closed' | 'not_originated';

export interface ScoreResponse {
  loan_id: number;
  as_of_date: string;
  cutoff: string;
  status: ScoreStatus;
  detail: string;
  risk_6m: number | null;
  risk_raw: number | null;
  hazard: number[] | null;           // raw monthly hazard h1–h6 (uncalibrated)
  hazard_scaled: number[] | null;    // rescaled so cumulative incidence at 6m = risk_6m
  cumulative_incidence: number[] | null;
  queue_rank: number | null;
  queue_percentile: number | null;
  queue_size: number | null;
  in_top_10pct: boolean | null;
  in_training_domain: boolean | null;
  drivers: Driver[];
  window: Window;
  model_version: string;
}

// ─── UI helpers ─────────────────────────────────────────────────────────────

export interface ApiError {
  message: string;
  status?: number;
  detail?: unknown;
}
