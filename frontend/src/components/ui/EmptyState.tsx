import React from 'react';
import { InboxIcon } from 'lucide-react';

interface EmptyStateProps {
  message?: string;
  subtext?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  message = 'No data available',
  subtext,
}) => (
  <div className="flex flex-col items-center justify-center py-14 gap-3 text-center">
    <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center">
      <InboxIcon className="w-6 h-6 text-slate-400" />
    </div>
    <div>
      <p className="font-medium text-slate-600">{message}</p>
      {subtext && <p className="text-sm text-slate-400 mt-1">{subtext}</p>}
    </div>
  </div>
);
