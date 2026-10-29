import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchHealth } from '../api/endpoints';
import type { HealthResponse, ApiError } from '../api/types';
import { ApiRequestError } from '../api/client';

interface UseHealthResult {
  data: HealthResponse | null;
  error: ApiError | null;
  loading: boolean;
  lastChecked: Date | null;
  refetch: () => void;
}

export function useHealth(pollMs = 0): UseHealthResult {
  const [data, setData]         = useState<HealthResponse | null>(null);
  const [error, setError]       = useState<ApiError | null>(null);
  const [loading, setLoading]   = useState(true);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [tick, setTick]         = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setLoading(true);
    fetchHealth(ctrl.signal)
      .then((d) => {
        if (ctrl.signal.aborted) return;
        setData(d);
        setError(null);
        setLastChecked(new Date());
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return;
        setError(err instanceof ApiRequestError ? err.toApiError() : { message: String(err) });
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });

    return () => ctrl.abort();
  }, [tick]);

  // Optional polling
  useEffect(() => {
    if (!pollMs) return;
    const id = setInterval(() => setTick((t) => t + 1), pollMs);
    return () => clearInterval(id);
  }, [pollMs]);

  const refetch = useCallback(() => setTick((t) => t + 1), []);

  return { data, error, loading, lastChecked, refetch };
}
