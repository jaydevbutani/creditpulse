import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import type { ApiError } from '../../api/types';

interface ErrorStateProps {
  error: ApiError;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({ error, onRetry }) => (
  <div className="flex flex-col items-center justify-center py-14 gap-4 text-center">
    <div className="w-12 h-12 rounded-full bg-red-50 flex items-center justify-center">
      <AlertCircle className="w-6 h-6 text-red-500" />
    </div>
    <div>
      <p className="font-semibold text-slate-800 mb-1">
        {error.status ? `Error ${error.status}` : 'Request failed'}
      </p>
      <p className="text-sm text-slate-500 max-w-sm">{error.message}</p>
    </div>
    {onRetry && (
      <button
        onClick={onRetry}
        className="flex items-center gap-2 px-4 py-2 bg-navy-800 text-white text-sm font-medium rounded-lg hover:bg-navy-700 transition-colors"
      >
        <RefreshCw className="w-4 h-4" />
        Try again
      </button>
    )}
  </div>
);
