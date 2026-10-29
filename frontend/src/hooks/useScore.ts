import { useState, useCallback, useRef } from 'react';
import { fetchScore } from '../api/endpoints';
import type { ScoreRequest, ScoreResponse, ApiError } from '../api/types';
import { ApiRequestError } from '../api/client';

interface UseScoreResult {
  data: ScoreResponse | null;
  error: ApiError | null;
  loading: boolean;
  submit: (req: ScoreRequest) => void;
  reset: () => void;
}

export function useScore(): UseScoreResult {
  const [data, setData]       = useState<ScoreResponse | null>(null);
  const [error, setError]     = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const submit = useCallback((req: ScoreRequest) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setLoading(true);
    setData(null);
    setError(null);

    fetchScore(req, ctrl.signal)
      .then((d) => {
        if (ctrl.signal.aborted) return;
        setData(d);
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return;
        setError(err instanceof ApiRequestError ? err.toApiError() : { message: String(err) });
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, error, loading, submit, reset };
}
