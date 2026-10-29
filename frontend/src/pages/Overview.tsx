import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, TrendingUp, AlertTriangle, BarChart2 } from 'lucide-react';
import { Topbar } from '../components/layout/Topbar';
import { MetricCard } from '../components/ui/MetricCard';
import { DateSelector } from '../components/ui/DateSelector';
import { RiskBadge, RankBadge } from '../components/ui/RiskBadge';
import { LoadingState, TableSkeleton } from '../components/ui/LoadingState';
import { ErrorState } from '../components/ui/ErrorState';
import { EmptyState } from '../components/ui/EmptyState';
import { RiskDistributionChart } from '../charts/RiskDistributionChart';
import { TopRiskChart } from '../charts/HazardChart';
import { useHealth } from '../hooks/useHealth';
import { useQueue } from '../hooks/useQueue';
import { DEFAULT_DATE } from '../utils/dateRange';
import { fmtMonths } from '../utils/formatters';

const THRESHOLD = 0.028466; // serving.json threshold_10pct_queue

export const Overview: React.FC = () => {
  const [date, setDate] = useState(DEFAULT_DATE);
  const { data: health, error: healthErr, loading: healthLoading, lastChecked, refetch: refetchHealth } = useHealth();
  const { data: queueData, error: queueErr, loading: queueLoading, refetch: refetchQueue } = useQueue(date, 100);

  const queue = queueData?.queue ?? [];

  const metrics = useMemo(() => {
    if (!queue.length) return null;
    const scored  = queue.length;
    const elevated = queue.filter((e) => e.risk_6m >= THRESHOLD).length;
    const maxRisk  = Math.max(...queue.map((e) => e.risk_6m));
    return { scored, elevated, maxRisk };
  }, [queue]);

  const connected = healthErr ? false : health ? true : null;

  const handleRefresh = () => {
    refetchHealth();
    refetchQueue();
  };

  return (
    <div className="flex-1 flex flex-col min-h-screen">
      <Topbar
        title="Overview"
        subtitle="Portfolio risk summary"
        connected={connected}
        lastChecked={lastChecked}
        onRefresh={handleRefresh}
        isRefreshing={healthLoading || queueLoading}
      />

      <div className="flex-1 px-8 py-6 space-y-6">
        {/* Date selector */}
        <div className="flex flex-wrap items-end gap-6">
          <DateSelector value={date} onChange={setDate} label="Monitoring cutoff date" />
          {queueData && (
            <p className="text-sm text-slate-500 pb-1">
              Model: <span className="font-mono text-xs text-navy-700">{queueData.model_version}</span>
            </p>
          )}
        </div>

        {/* Metric cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            label="Total loans in dataset"
            value={health?.loans ?? '—'}
            subtext="All loans known to the scoring service"
            icon={<Users className="w-6 h-6" />}
            loading={healthLoading}
          />
          <MetricCard
            label="Loans in review queue"
            value={queueLoading ? '—' : queue.length}
            subtext={`At-risk loans for ${date}`}
            icon={<BarChart2 className="w-6 h-6" />}
            accent="navy"
            loading={queueLoading}
          />
          <MetricCard
            label="Above alert threshold"
            value={queueLoading ? '—' : metrics?.elevated ?? 0}
            subtext={`risk ≥ ${(THRESHOLD * 100).toFixed(1)}% (10% queue fill)`}
            icon={<AlertTriangle className="w-6 h-6" />}
            accent={metrics && metrics.elevated > 0 ? 'amber' : 'green'}
            loading={queueLoading}
          />
          <MetricCard
            label="Highest 6-month risk"
            value={metrics ? (metrics.maxRisk * 100).toFixed(1) + '%' : '—'}
            subtext="Calibrated distress probability"
            icon={<TrendingUp className="w-6 h-6" />}
            accent={metrics && metrics.maxRisk >= 0.05 ? 'red' : 'default'}
            loading={queueLoading}
          />
        </div>

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <h3 className="text-sm font-semibold text-navy-900 mb-1">Risk distribution</h3>
            <p className="text-xs text-slate-400 mb-4">
              Number of loans per calibrated 6-month distress probability band
            </p>
            {queueLoading ? (
              <div className="h-56 flex items-center justify-center">
                <LoadingState message="Loading chart data…" />
              </div>
            ) : queueErr ? (
              <ErrorState error={queueErr} onRetry={refetchQueue} />
            ) : (
              <RiskDistributionChart entries={queue} />
            )}
          </div>

          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <h3 className="text-sm font-semibold text-navy-900 mb-1">Top 15 loans by risk</h3>
            <p className="text-xs text-slate-400 mb-4">
              Calibrated 6-month risk for the highest-ranked loans in the queue
            </p>
            {queueLoading ? (
              <div className="h-56 flex items-center justify-center">
                <LoadingState message="Loading chart data…" />
              </div>
            ) : queueErr ? (
              <ErrorState error={queueErr} onRetry={refetchQueue} />
            ) : (
              <TopRiskChart entries={queue} />
            )}
          </div>
        </div>

        {/* Priority review table */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm">
          <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-navy-900">Highest priority reviews</h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Ranked by estimated 6-month distress risk. Review ranking, not certainty of default.
              </p>
            </div>
          </div>

          {queueLoading ? (
            <TableSkeleton rows={10} cols={5} />
          ) : queueErr ? (
            <div className="p-6">
              <ErrorState error={queueErr} onRetry={refetchQueue} />
            </div>
          ) : queue.length === 0 ? (
            <EmptyState message="No at-risk loans for this date" subtext="Try a different monitoring date." />
          ) : (
            <PriorityTable queue={queue} />
          )}
        </div>
      </div>
    </div>
  );
};

// ─── Priority Table ──────────────────────────────────────────────────────────

interface PriorityTableProps {
  queue: { loan_id: number; queue_rank: number; risk_6m: number; queue_percentile: number; loan_age: number }[];
}

const PriorityTable: React.FC<PriorityTableProps> = ({ queue }) => {
  const navigate = useNavigate();
  const top = queue.slice(0, 20);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-slate-50 border-b border-slate-200">
            <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider w-16">Rank</th>
            <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">Loan ID</th>
            <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">6-Month Risk</th>
            <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">Percentile</th>
            <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">Loan Age</th>
            <th className="px-4 py-3 w-20"></th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {top.map((row) => (
            <tr key={row.loan_id} className="hover:bg-slate-50 transition-colors">
              <td className="px-5 py-3.5">
                <RankBadge rank={row.queue_rank} />
              </td>
              <td className="px-4 py-3.5 font-mono font-semibold text-navy-800">#{row.loan_id}</td>
              <td className="px-4 py-3.5">
                <RiskBadge risk={row.risk_6m} />
              </td>
              <td className="px-4 py-3.5 text-slate-600">
                top {row.queue_percentile.toFixed(1)}%
              </td>
              <td className="px-4 py-3.5 text-slate-500">{fmtMonths(row.loan_age)}</td>
              <td className="px-4 py-3.5">
                <button
                  onClick={() => navigate(`/scoring?loan_id=${row.loan_id}`)}
                  className="text-xs font-medium text-navy-600 hover:text-navy-800 hover:underline"
                >
                  Score →
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {queue.length > 20 && (
        <div className="px-5 py-3 border-t border-slate-100 text-xs text-slate-400">
          Showing top 20 of {queue.length} loans. Visit Risk Monitoring for the full queue.
        </div>
      )}
    </div>
  );
};
