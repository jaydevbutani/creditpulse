import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import { Topbar } from '../components/layout/Topbar';
import { DateSelector } from '../components/ui/DateSelector';
import { RiskBadge, RankBadge } from '../components/ui/RiskBadge';
import { TableSkeleton } from '../components/ui/LoadingState';
import { ErrorState } from '../components/ui/ErrorState';
import { EmptyState } from '../components/ui/EmptyState';
import { useHealth } from '../hooks/useHealth';
import { useQueue } from '../hooks/useQueue';
import { DEFAULT_DATE } from '../utils/dateRange';
import { fmtMonths } from '../utils/formatters';

const TOP_OPTIONS = [10, 25, 50, 100, 200];
const PAGE_SIZE = 25;

export const RiskMonitoring: React.FC = () => {
  const [date, setDate]     = useState(DEFAULT_DATE);
  const [top, setTop]       = useState(50);
  const [search, setSearch] = useState('');
  const [page, setPage]     = useState(1);
  const [sortKey, setSortKey] = useState<'queue_rank' | 'risk_6m' | 'loan_age'>('queue_rank');
  const [sortAsc, setSortAsc] = useState(true);

  const { data: health, error: healthErr } = useHealth();
  const { data: queueData, error, loading, refetch } = useQueue(date, top);

  const connected = healthErr ? false : health ? true : null;

  const filtered = useMemo(() => {
    const q = queueData?.queue ?? [];
    const s = search.trim();
    const list = s ? q.filter((e) => String(e.loan_id).includes(s)) : q;
    return [...list].sort((a, b) => {
      const d = a[sortKey] - b[sortKey];
      return sortAsc ? d : -d;
    });
  }, [queueData, search, sortKey, sortAsc]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const handleSort = (key: typeof sortKey) => {
    if (key === sortKey) setSortAsc((v) => !v);
    else { setSortKey(key); setSortAsc(true); }
    setPage(1);
  };

  const navigate = useNavigate();

  const SortIcon: React.FC<{ col: typeof sortKey }> = ({ col }) => {
    if (sortKey !== col) return <span className="text-slate-300 ml-1">↕</span>;
    return <span className="text-navy-600 ml-1">{sortAsc ? '↑' : '↓'}</span>;
  };

  return (
    <div className="flex-1 flex flex-col">
      <Topbar
        title="Risk Monitoring"
        subtitle="Review queue ranked by estimated 6-month distress risk"
        connected={connected}
        onRefresh={refetch}
        isRefreshing={loading}
      />

      <div className="px-8 py-6 space-y-5">
        {/* Controls */}
        <div className="flex flex-wrap items-end gap-5 bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
          <DateSelector value={date} onChange={(d) => { setDate(d); setPage(1); }} />

          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Top N loans</label>
            <select
              value={top}
              onChange={(e) => { setTop(Number(e.target.value)); setPage(1); }}
              className="px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white text-navy-900 font-medium shadow-sm focus:outline-none focus:ring-2 focus:ring-navy-500"
            >
              {TOP_OPTIONS.map((n) => (
                <option key={n} value={n}>{n} loans</option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1 flex-1 min-w-48">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Search by loan ID</label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                value={search}
                onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                placeholder="e.g. 5773"
                className="pl-9 pr-3 py-2 text-sm border border-slate-200 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-navy-500"
              />
            </div>
          </div>
        </div>

        {/* Notice */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-2.5 text-xs text-amber-800">
          <strong>Queue rank</strong> indicates relative priority for analyst review — not certainty that a loan will experience distress.
          Risk estimates are calibrated probabilities, not guarantees.
        </div>

        {/* Table */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
            <div>
              <span className="text-sm font-semibold text-navy-900">
                {loading ? 'Loading…' : `${filtered.length} loan${filtered.length !== 1 ? 's' : ''}`}
              </span>
              {queueData && (
                <span className="text-xs text-slate-400 ml-2">· {date} ·{' '}
                  <span className="font-mono">{queueData.model_version.split('-')[0]}</span>
                </span>
              )}
            </div>
            {totalPages > 1 && (
              <div className="flex items-center gap-2 text-sm">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="px-2.5 py-1 rounded border border-slate-200 disabled:opacity-40 hover:bg-slate-50"
                >
                  ‹
                </button>
                <span className="text-slate-500">
                  {page} / {totalPages}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  className="px-2.5 py-1 rounded border border-slate-200 disabled:opacity-40 hover:bg-slate-50"
                >
                  ›
                </button>
              </div>
            )}
          </div>

          {loading ? (
            <TableSkeleton rows={10} cols={5} />
          ) : error ? (
            <div className="p-6">
              <ErrorState error={error} onRetry={refetch} />
            </div>
          ) : paged.length === 0 ? (
            <EmptyState
              message={search ? `No loans matching "${search}"` : 'No at-risk loans for this date'}
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <th
                      className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider cursor-pointer select-none"
                      onClick={() => handleSort('queue_rank')}
                    >
                      Rank <SortIcon col="queue_rank" />
                    </th>
                    <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                      Loan ID
                    </th>
                    <th
                      className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider cursor-pointer select-none"
                      onClick={() => handleSort('risk_6m')}
                    >
                      6-Month Risk <SortIcon col="risk_6m" />
                    </th>
                    <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                      Percentile
                    </th>
                    <th
                      className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider cursor-pointer select-none"
                      onClick={() => handleSort('loan_age')}
                    >
                      Loan Age <SortIcon col="loan_age" />
                    </th>
                    <th className="px-4 py-3 w-20"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {paged.map((row) => (
                    <tr key={row.loan_id} className="hover:bg-slate-50 transition-colors">
                      <td className="px-5 py-3.5">
                        <RankBadge rank={row.queue_rank} />
                      </td>
                      <td className="px-4 py-3.5 font-mono font-semibold text-navy-800">
                        #{row.loan_id}
                      </td>
                      <td className="px-4 py-3.5">
                        <RiskBadge risk={row.risk_6m} />
                      </td>
                      <td className="px-4 py-3.5 text-slate-600">
                        top {row.queue_percentile.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3.5 text-slate-500">
                        {fmtMonths(row.loan_age)}
                      </td>
                      <td className="px-4 py-3.5">
                        <button
                          onClick={() => navigate(`/scoring?loan_id=${row.loan_id}`)}
                          className="text-xs font-medium text-navy-600 hover:text-navy-800 hover:underline whitespace-nowrap"
                        >
                          Score →
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
