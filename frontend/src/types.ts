export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
  role: string;
  department: string;
}

export interface CutoffMonth {
  cutoff_mi: number;
  label: string;
}

export interface HeadlineEval {
  model_name: string;
  pr_auc: number;
  pr_auc_ci: string;
  prevalence: number;
  capture_at_10: number;
  median_lead_months: number;
  brier_score: number;
  event_loans_test: number;
}

export interface QueueRow {
  queue_rank: number;
  loan_id: number;
  account_id: number;
  sector?: string;
  risk_6m: number;
  risk_percent: number;
  delta_vs_last_month: number | null;
  queue_percentile: number;
  loan_age: number;
  top_driver: string;
  above_alert: boolean;
  review_status?: string;
  has_note?: boolean;
  agreement?: 'Confirmed Concern' | 'Model Alert' | 'Human Override' | 'Normal';
  human_flag?: HumanLoanFlag;
}

export interface EarlyWarningSignal {
  id: string;
  signal: string;
  facilities: number;
  severity: 'High' | 'Medium' | 'Low';
  description: string;
}

export interface SegmentSummary {
  segment: string;
  total: number;
  low: number;
  moderate: number;
  high: number;
  critical: number;
}

export interface ComparisonMetrics {
  period_a_label: string;
  period_b_label: string;
  at_risk_loans_a: number;
  at_risk_loans_b: number;
  at_risk_loans_diff: number;
  mean_risk_a: number;
  mean_risk_b: number;
  mean_risk_diff_pp: number;
  above_threshold_a: number;
  above_threshold_b: number;
  above_threshold_diff: number;
  high_risk_a: number;
  high_risk_b: number;
  high_risk_diff: number;
}

export interface HumanLoanFlag {
  id?: number;
  loan_id: number;
  is_flagged: boolean;
  severity: 'Watch' | 'Concern' | 'Critical' | 'None';
  reason: string;
  note: string;
  flagged_by: string;
  created_at?: string | null;
  updated_at?: string | null;
  resolved_at?: string | null;
  resolved_by?: string | null;
}

export interface AgreementSummary {
  confirmed_concern: number;
  model_alert: number;
  human_override: number;
  normal: number;
}

export interface SimulatedStreamRecord {
  id: number;
  record_id: number;
  loan_id: number;
  account_id: number;
  borrower_name: string;
  event_type: string;
  amount: number;
  balance: number;
  model_version: string;
  predicted_risk: number;
  risk_category: string;
  model_flag: number;
  human_flag: number;
  processing_status: string;
  features_json?: string;
  timestamp: string;
}

export interface PortfolioMonitorData {
  current_cutoff_mi: number;
  current_cutoff_date: string;
  range_start_mi?: number | null;
  range_start_date?: string | null;
  is_range?: boolean;
  total_at_risk_loans: number;
  top_5_10_20: string;
  above_alert_threshold: number;
  alert_threshold_val: number;
  mean_risk_6m: number;
  mean_risk_percent: number;
  mom_entered_top_10: number;
  mom_left_top_10: number;
  headline: HeadlineEval;
  early_warnings?: EarlyWarningSignal[];
  segment_summary?: SegmentSummary[];
  comparison?: ComparisonMetrics | null;
  agreement_summary?: AgreementSummary;
  human_review_queue?: Array<Record<string, any>>;
  why_risk_changed?: Array<{ factor: string; delta: string; explanation: string; impact: string }>;
  queue: QueueRow[];
  model_version: string;
}

export interface TrajectoryPoint {
  month: string;
  risk_6m: number;
  threshold: number;
}

export interface HazardBreakdownPoint {
  horizon: string;
  monthly_hazard: number;
  cumulative_risk: number;
}

export interface BehaviorRow {
  month: string;
  inflow: number;
  outflow: number;
  bal_eom: number;
  bal_min: number;
  days_below_zero: number;
  uver_paid: number;
  loan_payment: number;
  n_tx: number;
}

export interface AttributionDriver {
  feature: string;
  contribution: number;
  value: number;
  text: string;
  direction?: 'increasing' | 'protective';
}

export interface WhatChangedItem {
  category: string;
  severity: 'Critical' | 'High' | 'Medium' | 'Low' | 'Normal';
  text: string;
}

export interface LoanReviewState {
  loan_id: number;
  status: 'normal' | 'reviewed' | 'watchlist' | 'escalated';
  note: string;
  updated_at: string | null;
  updated_by: string | null;
}

export interface LoanDetailData {
  loan_id: number;
  account_id: number;
  cutoff_date: string;
  cutoff_mi: number;
  risk_6m: number;
  risk_percent: number;
  queue_rank: number;
  queue_total: number;
  queue_percentile: number;
  loan_age_months: number;
  above_threshold: boolean;
  trajectory: TrajectoryPoint[];
  hazard_breakdown: HazardBreakdownPoint[];
  drivers: AttributionDriver[];
  risk_increasing_drivers?: AttributionDriver[];
  protective_drivers?: AttributionDriver[];
  what_changed?: WhatChangedItem[];
  review?: LoanReviewState;
  human_flag?: HumanLoanFlag;
  model_flag?: boolean;
  agreement?: 'Confirmed Concern' | 'Model Alert' | 'Human Override' | 'Normal';
  risk_trend?: {
    direction: 'Increasing' | 'Decreasing' | 'Stable';
    delta_pp: number;
    baseline_month: string;
    description: string;
  };
  decision_support_guidance?: {
    category: string;
    summary: string;
    recommended_action: string;
    suggested_timeline: string;
    supervisory_consideration: string;
  };
  behavior: BehaviorRow[];
  model_version: string;
}

export interface StressTestResult {
  loan_id: number;
  cutoff_date: string;
  base_risk_percent: number;
  stressed_risk_percent: number;
  risk_delta_percent: number;
  stressed_tier: string;
  scenario_parameters: {
    inflow_pct: number;
    outflow_pct: number;
    bal_pct: number;
    payment_delay_months: number;
  };
  stressed_hazard_breakdown: Array<{
    horizon: string;
    base_hazard: number;
    stressed_hazard: number;
    base_cumulative: number;
    stressed_cumulative: number;
  }>;
  is_simulation: boolean;
}

export interface SimulationStreamEvent {
  account_id: number;
  event: string;
  signal: string;
  impact: string;
  time: string;
}

export interface SimulationStreamData {
  status: string;
  label: string;
  last_update: string;
  incoming_records?: number;
  records_stored?: number;
  records_processed?: number;
  predictions_generated?: number;
  high_risk_detected?: number;
  human_flagged_loans?: number;
  new_records_session?: number;
  events_processed?: number;
  events_flagged?: number;
  active_speed?: string;
  events: SimulationStreamEvent[];
  model_monitoring: {
    new_labelled_cases: number;
    performance_drift: string;
    calibration_drift: string;
    current_pr_auc: number;
    previous_pr_auc: number;
    retraining_recommended: boolean;
    status_text: string;
  };
}

export interface AuditEvent {
  id: number;
  user_id: number | null;
  username: string;
  action: string;
  details: string;
  created_at: string;
}

export interface BenchmarkRow {
  model: string;
  pr_auc: number | null;
  pr_auc_lo: number | null;
  pr_auc_hi: number | null;
  roc_auc: number | null;
  brier: number | null;
  ece: number | null;
  'capture@5': number | null;
  'capture@10': number | null;
  'capture@20': number | null;
  median_lead_months: number | null;
  share_lead_ge3: number | null;
  false_alarms_per_month: number | null;
}

export interface ModelHealthData {
  served_model: string;
  model_version: string;
  threshold_10pct: number;
  benchmarks: BenchmarkRow[];
  capture_ladder: Array<{
    model: string;
    capture_top5: number;
    capture_top10: number;
    capture_top20: number;
  }>;
  calibration_curve: Array<{
    mean_pred: number;
    observed_rate: number;
    baseline_perfect: number;
  }>;
  training_history?: Array<{
    epoch: number;
    train_loss: number;
    val_loss: number;
    val_pr_auc: number;
    val_roc_auc: number;
    val_brier?: number;
  }>;
  psi_drift: Array<Record<string, any>>;
  model_configuration?: {
    model_name: string;
    input_dim: number;
    d_model: number;
    n_layers: number;
    n_heads: number;
    dropout: number;
    positive_weight: number;
    loss_lambda: number;
    optimizer: string;
    learning_rate: number;
    epochs: number;
    batch_size: number;
    window_length: number;
    platt_slope: number;
    platt_intercept: number;
    alert_threshold: number;
  };
  training_summary?: {
    training_samples: number;
    event_prevalence: string;
    calibration_method: string;
    ensemble_seeds: number[];
    train_test_split: string;
    loss_function: string;
  };
}

export interface CaseSummary {
  id: number;
  case_name: string;
  dataset_filename: string;
  original_filename: string;
  notes: string;
  total_facilities: number;
  avg_risk_6m: number;
  critical_count: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface FacilityDriver {
  feature: string;
  contribution: number;
  value: number;
  text: string;
}

export interface FacilityResult {
  id: number;
  loan_id: number;
  borrower_name: string;
  sector: string;
  principal_amount: number;
  dpd_status: string;
  risk_6m: number;
  risk_tier: string;
  hazard: number[];
  drivers: FacilityDriver[];
}

export interface CaseDetailType extends CaseSummary {
  summary: {
    mean_hazard_trajectory?: number[];
    critical_count?: number;
    emerging_count?: number;
    watchlist_count?: number;
  };
  facilities: FacilityResult[];
  raw_csv_data: Array<Record<string, any>>;
}

export type CaseDetail = CaseDetailType;

export interface SampleDataset {
  filename: string;
  name: string;
  description: string;
  records: number;
  sectors: string[];
}
