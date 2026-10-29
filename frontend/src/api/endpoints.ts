// Typed endpoint functions — one per backend route.

import { get, post } from './client';
import type { HealthResponse, Loan, QueueResponse, ScoreRequest, ScoreResponse } from './types';

export const fetchHealth = (signal?: AbortSignal): Promise<HealthResponse> =>
  get<HealthResponse>('/health', signal);

export const fetchLoans = (limit = 100, signal?: AbortSignal): Promise<Loan[]> =>
  get<Loan[]>(`/loans?limit=${limit}`, signal);

export const fetchQueue = (
  as_of_date: string,
  top = 50,
  signal?: AbortSignal
): Promise<QueueResponse> =>
  get<QueueResponse>(`/queue?as_of_date=${as_of_date}&top=${top}`, signal);

export const fetchScore = (req: ScoreRequest, signal?: AbortSignal): Promise<ScoreResponse> =>
  post<ScoreResponse>('/score', req, signal);
