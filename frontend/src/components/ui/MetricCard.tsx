import React from 'react';

interface MetricCardProps {
  label: string;
  value: React.ReactNode;
  subtext?: string;
  icon?: React.ReactNode;
  accent?: 'default' | 'green' | 'amber' | 'red' | 'navy';
  loading?: boolean;
}

const accentStyles: Record<string, string> = {
  default: 'border-slate-200',
  green:   'border-l-4 border-l-emerald-500 border-slate-200',
  amber:   'border-l-4 border-l-amber-500 border-slate-200',
  red:     'border-l-4 border-l-red-500 border-slate-200',
  navy:    'border-l-4 border-l-navy-700 border-slate-200',
};

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  icon,
  accent = 'default',
  loading = false,
}) => (
  <div className={`bg-white rounded-xl border shadow-sm p-5 ${accentStyles[accent]}`}>
    <div className="flex items-start justify-between">
      <div className="flex-1 min-w-0">
        <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2">
          {label}
        </p>
        {loading ? (
          <div className="h-7 w-24 bg-slate-100 animate-pulse rounded" />
        ) : (
          <p className="text-2xl font-bold text-navy-900 leading-none truncate">{value}</p>
        )}
        {subtext && !loading && (
          <p className="text-slate-400 text-xs mt-1.5">{subtext}</p>
        )}
      </div>
      {icon && (
        <div className="ml-3 text-slate-300 flex-shrink-0">{icon}</div>
      )}
    </div>
  </div>
);
