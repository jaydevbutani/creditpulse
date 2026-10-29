// Base HTTP client with timeout, error normalisation, and abort support.

import type { ApiError } from './types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const DEFAULT_TIMEOUT_MS = 30_000;

export class ApiRequestError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly detail?: unknown
  ) {
    super(message);
    this.name = 'ApiRequestError';
  }

  toApiError(): ApiError {
    return { message: this.message, status: this.status, detail: this.detail };
  }
}

async function request<T>(
  path: string,
  options: RequestInit & { timeoutMs?: number } = {}
): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal: externalSignal, ...rest } = options;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  // If the caller passed their own signal, honour it too.
  if (externalSignal) {
    externalSignal.addEventListener('abort', () => controller.abort());
  }

  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      ...rest,
      signal: controller.signal,
    });

    clearTimeout(timer);

    if (!res.ok) {
      let detail: unknown = undefined;
      try {
        detail = await res.json();
      } catch {
        /* non-JSON error body */
      }

      const msg =
        typeof detail === 'object' && detail !== null && 'detail' in detail
          ? String((detail as { detail: unknown }).detail)
          : `HTTP ${res.status} ${res.statusText}`;

      throw new ApiRequestError(msg, res.status, detail);
    }

    return (await res.json()) as T;
  } catch (err) {
    clearTimeout(timer);

    if (err instanceof ApiRequestError) throw err;

    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiRequestError('Request timed out or was cancelled.', undefined);
    }

    throw new ApiRequestError(
      err instanceof Error ? err.message : 'Unknown network error',
      undefined
    );
  }
}

export const get = <T>(path: string, signal?: AbortSignal): Promise<T> =>
  request<T>(path, { method: 'GET', signal });

export const post = <T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> =>
  request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
