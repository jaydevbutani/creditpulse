import {
  PortfolioMonitorData,
  CutoffMonth,
  LoanDetailData,
  ModelHealthData,
  CaseSummary,
  CaseDetailType,
  SampleDataset,
  User,
  StressTestResult,
  SimulationStreamData,
  AuditEvent,
  LoanReviewState
} from './types';

const API_BASE = 'http://127.0.0.1:8000';

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = localStorage.getItem('cp_token');
  const headers = new Headers(options.headers || {});
  
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    let errorDetail = 'API request failed';
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errJson.message || errorDetail;
    } catch {
      errorDetail = await res.text() || res.statusText;
    }
    throw new Error(errorDetail);
  }

  return res.json();
}

export const api = {
  // Auth
  login: async (username: string, password?: string) => {
    return request<{ token: string; user: User }>('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
  },

  register: async (payload: { username: string; email: string; password: string; full_name: string; role?: string }) => {
    return request<{ token: string; user: User }>('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  getMe: async () => {
    return request<User>('/api/auth/me');
  },

  // Portfolio Monitor & Loan Detail
  getCutoffMonths: async () => {
    return request<CutoffMonth[]>('/api/portfolio/months');
  },

  getPortfolioMonitor: async (params?: {
    cutoff_mi?: number;
    range_start_mi?: number;
    compare_cutoff_mi?: number;
    tier_filter?: string;
    segment_filter?: string;
    warning_filter?: string;
  }) => {
    const q = new URLSearchParams();
    if (params?.cutoff_mi) q.set('cutoff_mi', params.cutoff_mi.toString());
    if (params?.range_start_mi) q.set('range_start_mi', params.range_start_mi.toString());
    if (params?.compare_cutoff_mi) q.set('compare_cutoff_mi', params.compare_cutoff_mi.toString());
    if (params?.tier_filter) q.set('tier_filter', params.tier_filter);
    if (params?.segment_filter) q.set('segment_filter', params.segment_filter);
    if (params?.warning_filter) q.set('warning_filter', params.warning_filter);

    const queryStr = q.toString() ? `?${q.toString()}` : '';
    return request<PortfolioMonitorData>(`/api/portfolio/monitor${queryStr}`);
  },

  getLoanDetail: async (loanId: number, cutoffMi?: number) => {
    const q = cutoffMi ? `?cutoff_mi=${cutoffMi}` : '';
    return request<LoanDetailData>(`/api/loan/${loanId}${q}`);
  },

  runStressTest: async (loanId: number, payload: {
    inflow_pct: number;
    outflow_pct: number;
    bal_pct: number;
    payment_delay_months: number;
    cutoff_mi?: number;
  }) => {
    return request<StressTestResult>(`/api/loan/${loanId}/stress-test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  getLoanReview: async (loanId: number) => {
    return request<LoanReviewState>(`/api/loan/${loanId}/review`);
  },

  setLoanReview: async (loanId: number, status: string, note: string) => {
    return request<{ status: string; message: string; data: LoanReviewState }>(`/api/loan/${loanId}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, note }),
    });
  },

  // Audit Events
  getAuditEvents: async (limit: number = 50) => {
    return request<AuditEvent[]>(`/api/audit?limit=${limit}`);
  },

  logAuditEvent: async (action: string, details?: string) => {
    return request<{ status: string }>('/api/audit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, details }),
    });
  },

  // Human Review & Flagging
  getHumanFlags: async () => {
    return request<any[]>('/api/human-flags');
  },

  getLoanFlag: async (loanId: number) => {
    return request<any>(`/api/loan/${loanId}/flag`);
  },

  flagLoan: async (loanId: number, data: { severity: string; reason: string; note?: string }) => {
    return request<{ status: string; message: string; data: any }>(`/api/loan/${loanId}/flag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  },

  resolveLoanFlag: async (loanId: number, resolution_note?: string) => {
    return request<{ status: string; message: string; data: any }>(`/api/loan/${loanId}/resolve-flag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resolution_note: resolution_note || 'Resolved by risk analyst.' }),
    });
  },

  // Simulation Stream & Persistence
  getSimulationStream: async () => {
    return request<SimulationStreamData>('/api/simulation/stream');
  },

  processNextSimRecord: async () => {
    return request<{ status: string; message: string; record: any }>('/api/simulation/process-next', {
      method: 'POST',
    });
  },

  getSimRecords: async (limit: number = 50) => {
    return request<any[]>(`/api/simulation/records?limit=${limit}`);
  },

  resetSimRecords: async () => {
    return request<{ status: string; message: string }>('/api/simulation/reset', {
      method: 'POST',
    });
  },

  getModelHealth: async () => {
    return request<ModelHealthData>('/api/model-health');
  },

  simulateRetraining: async () => {
    return request<any>('/api/model/simulate-retraining', {
      method: 'POST',
    });
  },

  // Case & Dataset CRUD
  getCases: async () => {
    return request<CaseSummary[]>('/api/cases');
  },

  getCaseDetail: async (caseId: number) => {
    return request<CaseDetailType>(`/api/cases/${caseId}`);
  },

  validateCsv: async (formData: FormData) => {
    const res = await fetch(`${API_BASE}/api/cases/validate`, {
      method: 'POST',
      body: formData,
    });
    return res.json();
  },

  uploadCase: async (formData: FormData) => {
    const token = localStorage.getItem('cp_token');
    const headers = new Headers();
    if (token) headers.set('Authorization', `Bearer ${token}`);

    const res = await fetch(`${API_BASE}/api/cases/upload`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  updateCase: async (caseId: number, caseName: string, notes: string) => {
    return request<{ status: string; message: string }>(`/api/cases/${caseId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ case_name: caseName, notes }),
    });
  },

  deleteCase: async (caseId: number) => {
    return request<{ status: string; message: string }>(`/api/cases/${caseId}`, {
      method: 'DELETE',
    });
  },

  // Sample Datasets
  getSampleDatasets: async () => {
    return request<SampleDataset[]>('/api/sample-datasets');
  },

  downloadSampleUrl: (filename: string) => `${API_BASE}/api/sample-datasets/${filename}`,
  downloadCaseCsvUrl: (caseId: number) => `${API_BASE}/api/cases/${caseId}/download-csv`,
};
