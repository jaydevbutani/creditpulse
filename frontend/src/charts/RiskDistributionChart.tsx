import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { QueueEntry } from '../api/types';
import { EmptyState } from '../components/ui/EmptyState';

interface RiskDistributionChartProps {
  entries: QueueEntry[];
}

const BINS = [
  { label: '0–1%',   min: 0,    max: 0.01 },
  { label: '1–2%',   min: 0.01, max: 0.02 },
  { label: '2–3%',   min: 0.02, max: 0.03 },
  { label: '3–4%',   min: 0.03, max: 0.04 },
  { label: '4–5%',   min: 0.04, max: 0.05 },
  { label: '>5%',    min: 0.05, max: Infinity },
];

const BIN_COLORS = ['#16a34a', '#65a30d', '#d97706', '#ea580c', '#dc2626', '#991b1b'];

const CustomTooltip: React.FC<{ active?: boolean; payload?: { value: number }[]; label?: string }> = ({
  active,
  payload,
  label,
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-lg px-3 py-2 text-sm">
      <p className="font-semibold text-navy-900">{label}</p>
      <p className="text-slate-600">{payload[0].value} loan{payload[0].value !== 1 ? 's' : ''}</p>
    </div>
  );
};

export const RiskDistributionChart: React.FC<RiskDistributionChartProps> = ({ entries }) => {
  const data = useMemo(
    () =>
      BINS.map((bin, i) => ({
        label: bin.label,
        count: entries.filter((e) => e.risk_6m >= bin.min && e.risk_6m < bin.max).length,
        color: BIN_COLORS[i],
      })),
    [entries]
  );

  if (!entries.length) {
    return <EmptyState message="No risk data to display" />;
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 11, fill: '#64748b' }}
          axisLine={{ stroke: '#e2e8f0' }}
          tickLine={false}
        />
        <YAxis
          allowDecimals={false}
          tick={{ fontSize: 11, fill: '#64748b' }}
          axisLine={false}
          tickLine={false}
          width={28}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: '#f8fafc' }} />
        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};
