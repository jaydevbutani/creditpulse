import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
  rows?: number;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ message = 'Loading…', rows = 4 }) => (
  <div className="flex flex-col items-center justify-center py-16 gap-3 text-slate-400">
    <Loader2 className="w-7 h-7 animate-spin text-navy-500" />
    <p className="text-sm">{message}</p>
  </div>
);

export const TableSkeleton: React.FC<{ rows?: number; cols?: number }> = ({
  rows = 6,
  cols = 5,
}) => (
  <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
    <div className="bg-slate-50 border-b border-slate-200 px-4 py-3 flex gap-4">
      {Array.from({ length: cols }).map((_, i) => (
        <div key={i} className="h-3.5 bg-slate-200 rounded animate-pulse flex-1" />
      ))}
    </div>
    {Array.from({ length: rows }).map((_, i) => (
      <div key={i} className="flex gap-4 px-4 py-3.5 border-b border-slate-100 last:border-0">
        {Array.from({ length: cols }).map((_, j) => (
          <div
            key={j}
            className="h-3 bg-slate-100 rounded animate-pulse flex-1"
            style={{ opacity: 1 - i * 0.1 }}
          />
        ))}
      </div>
    ))}
  </div>
);
