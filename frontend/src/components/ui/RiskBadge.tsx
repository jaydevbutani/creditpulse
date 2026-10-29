import React from 'react';
import { riskTier } from '../../utils/formatters';

interface RiskBadgeProps {
  risk: number; // [0, 1]
  size?: 'sm' | 'md';
}

const tierStyles = {
  low:    'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200',
  medium: 'bg-amber-50 text-amber-700 ring-1 ring-amber-200',
  high:   'bg-red-50 text-red-700 ring-1 ring-red-200',
};

const tierLabels = {
  low:    'Low',
  medium: 'Elevated',
  high:   'High',
};

export const RiskBadge: React.FC<RiskBadgeProps> = ({ risk, size = 'md' }) => {
  const tier = riskTier(risk);
  const pct  = (risk * 100).toFixed(1) + '%';
  const sizeClass = size === 'sm' ? 'text-xs px-1.5 py-0.5' : 'text-sm px-2.5 py-1';

  return (
    <span
      className={`inline-flex items-center gap-1 font-semibold rounded-full ${sizeClass} ${tierStyles[tier]}`}
      title={`${tierLabels[tier]} risk — ${pct} calibrated 6-month distress probability`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${
        tier === 'high' ? 'bg-red-500' : tier === 'medium' ? 'bg-amber-500' : 'bg-emerald-500'
      }`} />
      {pct}
    </span>
  );
};

interface RankBadgeProps {
  rank: number;
}

export const RankBadge: React.FC<RankBadgeProps> = ({ rank }) => (
  <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-navy-800 text-white text-xs font-bold">
    #{rank}
  </span>
);
