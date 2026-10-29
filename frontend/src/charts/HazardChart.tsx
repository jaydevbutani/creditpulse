import React from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { EmptyState } from '../components/ui/EmptyState';

interface HazardChartProps {
  hazardScaled: number[];  // h1–h6 rescaled
  cumulativeIncidence: number[]; // 6 values; last = risk_6m
}

const Tip: React.FC<{ active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string }> = ({
  active, payload, label,
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-lg px-3 py-2 text-sm">
      <p className="font-semibold text-navy-900 mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {(p.value * 100).toFixed(2)}%
        </p>
      ))}
    </div>
  );
};

export const HazardChart: React.FC<HazardChartProps> = ({ hazardScaled, cumulativeIncidence }) => {
  if (!hazardScaled.length) return <EmptyState message="No hazard data" />;

  const data = hazardScaled.map((h, i) => ({
    month: `+${i + 1}m`,
    'Monthly hazard (scaled)': h,
    'Cumulative incidence': cumulativeIncidence[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="month" tick={{ fontSize: 11, fill: '#64748b' }} tickLine={false} />
        <YAxis
          tickFormatter={(v) => (v * 100).toFixed(1) + '%'}
          tick={{ fontSize: 11, fill: '#64748b' }}
          axisLine={false}
          tickLine={false}
          width={44}
        />
        <Tooltip content={<Tip />} cursor={{ fill: '#f8fafc' }} />
        <Legend
          wrapperStyle={{ fontSize: 11, color: '#64748b', paddingTop: 8 }}
          iconType="circle"
          iconSize={8}
        />
        <Bar dataKey="Monthly hazard (scaled)" fill="#2a6f97" radius={[3, 3, 0, 0]} />
        <Bar dataKey="Cumulative incidence" fill="#2ec4b6" radius={[3, 3, 0, 0]} opacity={0.8} />
      </BarChart>
    </ResponsiveContainer>
  );
};

interface TopRiskChartProps {
  entries: { loan_id: number; risk_6m: number; queue_rank: number }[];
}

export const TopRiskChart: React.FC<TopRiskChartProps> = ({ entries }) => {
  if (!entries.length) return <EmptyState message="No data to display" />;
  const top = entries.slice(0, 15).map((e) => ({
    name: `#${e.loan_id}`,
    'Risk (6m)': e.risk_6m,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={top} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} tickLine={false} />
        <YAxis
          tickFormatter={(v) => (v * 100).toFixed(1) + '%'}
          tick={{ fontSize: 11, fill: '#64748b' }}
          axisLine={false}
          tickLine={false}
          width={44}
        />
        <Tooltip
          formatter={(v: number) => [(v * 100).toFixed(2) + '%', '6-month risk']}
          contentStyle={{ fontSize: 12, borderRadius: 8 }}
        />
        <Line
          type="monotone"
          dataKey="Risk (6m)"
          stroke="#1d3557"
          strokeWidth={2}
          dot={{ r: 3, fill: '#1d3557' }}
          activeDot={{ r: 5 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};
