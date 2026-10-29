import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchLoans } from '../api/endpoints';
import type { Loan, ApiError } from '../api/types';
import { ApiRequestError } from '../api/client';

interface UseLoansResult {
  data: Loan[];
  error: ApiError | null;
  loading: boolean;
  refetch: () => void;
}

export function useLoans(limit: number): UseLoansResult {
  const [data, setData]       = useState<Loan[]>([]);
  const [error, setError]     = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick]       = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setLoading(true);
    setError(null);
    fetchLoans(limit, ctrl.signal)
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

    return () => ctrl.abort();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [limit, tick]);

  const refetch = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, refetch };
}
