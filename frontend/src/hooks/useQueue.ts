import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchQueue } from '../api/endpoints';
import type { QueueResponse, ApiError } from '../api/types';
import { ApiRequestError } from '../api/client';

interface UseQueueResult {
  data: QueueResponse | null;
  error: ApiError | null;
  loading: boolean;
  refetch: () => void;
}

export function useQueue(asOfDate: string, top: number): UseQueueResult {
  const [data, setData]       = useState<QueueResponse | null>(null);
  const [error, setError]     = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [tick, setTick]       = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!asOfDate) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setLoading(true);
    setError(null);
    fetchQueue(asOfDate, top, ctrl.signal)
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
  }, [asOfDate, top, tick]);

  const refetch = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, refetch };
}
