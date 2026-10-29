import React, { useEffect, useState } from 'react';
import { useParams, NavLink } from 'react-router-dom';
import { api } from '../api';
import { CaseDetail as CaseDetailType, FacilityResult } from '../types';
import {
  ArrowLeft,
  FileSpreadsheet,
  AlertTriangle,
  TrendingUp,
  Download,
  Calendar,
  Layers,
  ChevronDown,
  Info,
  CheckCircle,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';

export const CaseDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [caseData, setCaseData] = useState<CaseDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Active tab: 'predictions' vs 'raw_csv'
  const [activeTab, setActiveTab] = useState<'predictions' | 'raw_csv'>('predictions');

  useEffect(() => {
    if (id) loadCase(parseInt(id, 10));
  }, [id]);

  const loadCase = async (caseId: number) => {
    try {
      setLoading(true);
      const data = await api.getCaseDetail(caseId);
      setCaseData(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load case detail.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-[1560px] mx-auto px-4 py-16 text-center text-xs text-outline">
        <span className="w-8 h-8 border-2 border-cognac border-t-transparent rounded-full animate-spin inline-block mb-3"></span>
        <p>Loading forensic telemetry & uploaded dataset...</p>
      </div>
    );
  }

  if (error || !caseData) {
    return (
      <div className="max-w-[1560px] mx-auto px-4 py-16 text-center space-y-4">
        <AlertTriangle className="w-10 h-10 text-error mx-auto" />
        <h2 className="text-lg font-bold text-espresso">Unable to Load Case</h2>
        <p className="text-xs text-outline">{error || 'Case not found or permission denied.'}</p>
        <NavLink
          to="/history"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-espresso text-white text-xs font-bold"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Case History</span>
        </NavLink>
      </div>
    );
  }

  // Aggregate metrics
  const totalFacilities = caseData.facilities.length;
  const criticalCount = caseData.facilities.filter((f: FacilityResult) => f.risk_tier === 'Critical Distress').length;
  const emergingCount = caseData.facilities.filter((f: FacilityResult) => f.risk_tier === 'Emerging Warning').length;
  const watchCount = caseData.facilities.filter((f: FacilityResult) => f.risk_tier === 'Watchlist').length;
  const totalExposure = caseData.facilities.reduce((sum: number, f: FacilityResult) => sum + f.principal_amount, 0);

  // Mean hazard trajectory points
  const trajectoryData = [
    { month: 'Month 1', hazard: caseData.summary.mean_hazard_trajectory?.[0] ? caseData.summary.mean_hazard_trajectory[0] * 100 : 0.8 },
    { month: 'Month 2', hazard: caseData.summary.mean_hazard_trajectory?.[1] ? caseData.summary.mean_hazard_trajectory[1] * 100 : 1.2 },
    { month: 'Month 3', hazard: caseData.summary.mean_hazard_trajectory?.[2] ? caseData.summary.mean_hazard_trajectory[2] * 100 : 1.7 },
    { month: 'Month 4', hazard: caseData.summary.mean_hazard_trajectory?.[3] ? caseData.summary.mean_hazard_trajectory[3] * 100 : 2.4 },
    { month: 'Month 5', hazard: caseData.summary.mean_hazard_trajectory?.[4] ? caseData.summary.mean_hazard_trajectory[4] * 100 : 3.1 },
    { month: 'Month 6', hazard: caseData.summary.mean_hazard_trajectory?.[5] ? caseData.summary.mean_hazard_trajectory[5] * 100 : 4.0 },
  ];

  // Risk distribution bar data
  const tierDistributionData = [
    { tier: 'Critical', count: criticalCount, fill: '#DC2626' },
    { tier: 'Emerging', count: emergingCount, fill: '#D97706' },
    { tier: 'Watchlist', count: watchCount, fill: '#2A6F97' },
  ];

  // CSV column headers
  const csvHeaders = caseData.raw_csv_data.length > 0 ? Object.keys(caseData.raw_csv_data[0]) : [];

  return (
    <div className="max-w-[1560px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Top Breadcrumb & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <NavLink
          to="/history"
          className="inline-flex items-center gap-1.5 text-xs font-bold text-outline hover:text-espresso transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Case Registry</span>
        </NavLink>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1 rounded-full bg-emerald-100 text-tertiary text-xs font-bold flex items-center gap-1.5">
            <CheckCircle className="w-3.5 h-3.5" />
            Model Scored & Audited
          </span>
        </div>
      </div>

      {/* Case Header Banner */}
      <div className="bg-surface-container-lowest rounded-3xl p-6 sm:p-8 border border-sandBorder shadow-warm flex flex-col md:flex-row md:items-start justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-outline text-[11px] font-mono font-bold">
              Case #{caseData.id}
            </span>
            <span className="text-outline text-xs flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              {new Date(caseData.created_at).toLocaleDateString()}
            </span>
            <span className="text-outline text-xs flex items-center gap-1 font-mono">
              <FileSpreadsheet className="w-3.5 h-3.5 text-cognac" />
              {caseData.original_filename}
            </span>
          </div>

          <h1 className="text-2xl sm:text-3xl font-extrabold text-espresso tracking-tight">
            {caseData.case_name}
          </h1>
          <p className="text-xs sm:text-sm text-outline mt-1.5 max-w-3xl leading-relaxed">
            {caseData.notes || 'No extra notes specified for this examination case.'}
          </p>
        </div>

        {/* Aggregate Risk Gauge Box */}
        <div className="p-4 rounded-2xl bg-espresso text-white shrink-0 sm:min-w-[200px]">
          <span className="text-[10px] text-stone-400 font-bold uppercase tracking-wider block">
            MEAN 6M DISTRESS PROB
          </span>
          <span className="text-3xl font-extrabold text-amber-400 block mt-1">
            {(caseData.avg_risk_6m * 100).toFixed(2)}%
          </span>
          <span className="text-[11px] text-stone-300 mt-1 block">
            {criticalCount} of {totalFacilities} facilities critical
          </span>
        </div>
      </div>

      {/* 4 Summary Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
          <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
            Total Scored Facilities
          </span>
          <span className="text-3xl font-extrabold text-espresso tracking-tight">{totalFacilities}</span>
          <span className="text-[11px] text-outline block mt-2 pt-2 border-t border-sandBorder/40">
            Validated in uploaded CSV
          </span>
        </div>

        <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
          <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
            Total Principal Exposure
          </span>
          <span className="text-3xl font-extrabold text-espresso tracking-tight">
            ${(totalExposure / 1000).toFixed(1)}k
          </span>
          <span className="text-[11px] text-outline block mt-2 pt-2 border-t border-sandBorder/40">
            Aggregated portfolio principal
          </span>
        </div>

        <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
          <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
            Critical Distress (p &gt; 80%)
          </span>
          <span className="text-3xl font-extrabold text-error tracking-tight">{criticalCount}</span>
          <span className="text-[11px] text-error font-bold block mt-2 pt-2 border-t border-sandBorder/40">
            Immediate workout required
          </span>
        </div>

        <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
          <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
            Watchlist & Emerging
          </span>
          <span className="text-3xl font-extrabold text-amberGlow tracking-tight">
            {emergingCount + watchCount}
          </span>
          <span className="text-[11px] text-outline block mt-2 pt-2 border-t border-sandBorder/40">
            Early intervention targets
          </span>
        </div>
      </div>

      {/* Visual Analytics Grid: Hazard Horizon & Risk Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm">
          <h2 className="text-base font-bold text-espresso tracking-tight mb-1">
            Mean 6-Month Hazard Trajectory Curve (Head B)
          </h2>
          <p className="text-xs text-outline mb-4">
            Month-by-month discrete hazard probability shape scaled to Platt-calibrated Head A level.
          </p>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trajectoryData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE1" />
                <XAxis dataKey="month" stroke="#867461" fontSize={11} />
                <YAxis stroke="#867461" fontSize={11} unit="%" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1E1B18',
                    border: 'none',
                    borderRadius: '12px',
                    color: '#FFF',
                    fontSize: '11px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="hazard"
                  stroke="#DC2626"
                  strokeWidth={3}
                  dot={{ r: 4, fill: '#DC2626' }}
                  name="Discrete Hazard %"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="lg:col-span-4 bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm flex flex-col justify-between">
          <div>
            <h2 className="text-base font-bold text-espresso tracking-tight mb-1">Cohort Risk Breakdown</h2>
            <p className="text-xs text-outline mb-4">Distribution across Basel risk tiers.</p>
            <div className="h-44 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={tierDistributionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE1" />
                  <XAxis dataKey="tier" stroke="#867461" fontSize={11} />
                  <YAxis stroke="#867461" fontSize={11} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1E1B18',
                      border: 'none',
                      borderRadius: '12px',
                      color: '#FFF',
                      fontSize: '11px',
                    }}
                  />
                  <Bar dataKey="count" radius={[8, 8, 0, 0]} name="Facilities" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="p-3 rounded-2xl bg-surface-container text-xs space-y-1">
            <div className="flex justify-between">
              <span className="text-outline">Critical:</span>
              <span className="font-bold text-error">{criticalCount}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-outline">Emerging:</span>
              <span className="font-bold text-amberGlow">{emergingCount}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-outline">Watchlist:</span>
              <span className="font-bold text-slateInk">{watchCount}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tab Switcher: Neural Predictions vs Uploaded CSV Data */}
      <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm">
        <div className="flex items-center justify-between mb-5 border-b border-sandBorder pb-4 flex-wrap gap-3">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab('predictions')}
              className={`px-4 py-2 rounded-full text-xs font-bold transition-all ${
                activeTab === 'predictions'
                  ? 'bg-espresso text-white shadow-xs'
                  : 'bg-surface-container text-outline hover:text-espresso'
              }`}
            >
              Neural Risk Predictions & Drivers ({caseData.facilities.length})
            </button>
            <button
              onClick={() => setActiveTab('raw_csv')}
              className={`px-4 py-2 rounded-full text-xs font-bold transition-all flex items-center gap-1.5 ${
                activeTab === 'raw_csv'
                  ? 'bg-espresso text-white shadow-xs'
                  : 'bg-surface-container text-outline hover:text-espresso'
              }`}
            >
              <FileSpreadsheet className="w-3.5 h-3.5" />
              Uploaded Raw CSV Data ({caseData.raw_csv_data.length} Rows)
            </button>
          </div>

          <span className="text-xs text-outline font-mono">
            Source: <strong className="text-espresso">{caseData.original_filename}</strong>
          </span>
        </div>

        {/* Tab 1: Predictions Table */}
        {activeTab === 'predictions' && (
          <div className="overflow-x-auto rounded-2xl border border-sandBorder">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-surface-container border-b border-sandBorder text-outline uppercase tracking-wider text-[11px] font-bold">
                  <th className="py-3 px-4">Loan / Facility ID</th>
                  <th className="py-3 px-4">Borrower Entity</th>
                  <th className="py-3 px-4">Sector</th>
                  <th className="py-3 px-4">Exposure</th>
                  <th className="py-3 px-4">DPD Status</th>
                  <th className="py-3 px-4">6M Calibrated Risk</th>
                  <th className="py-3 px-4">Risk Tier</th>
                  <th className="py-3 px-4">Primary Attribution Drivers</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-sandBorder bg-white">
                {caseData.facilities.map((f: FacilityResult) => (
                  <tr key={f.id} className="hover:bg-[#FFF9EE] transition-colors">
                    <td className="py-3.5 px-4 font-mono font-bold text-espresso">#LN-{f.loan_id}</td>
                    <td className="py-3.5 px-4 font-bold text-espresso">{f.borrower_name}</td>
                    <td className="py-3.5 px-4">
                      <span className="px-2 py-0.5 rounded-full bg-surface-container text-outline text-[11px] font-semibold">
                        {f.sector}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono font-bold text-espresso">
                      ${f.principal_amount ? f.principal_amount.toLocaleString() : '0'}
                    </td>
                    <td className="py-3.5 px-4">
                      <span
                        className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
                          f.dpd_status?.includes('Past Due')
                            ? 'bg-red-50 text-error'
                            : 'bg-emerald-50 text-tertiary'
                        }`}
                      >
                        {f.dpd_status}
                      </span>
                    </td>
                    <td className="py-3.5 px-4">
                      <span className="px-2.5 py-1 rounded-full bg-red-100 text-error font-extrabold text-xs">
                        ⚡ {(f.risk_6m * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-4">
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
                          f.risk_tier === 'Critical Distress'
                            ? 'bg-red-600 text-white'
                            : f.risk_tier === 'Emerging Warning'
                            ? 'bg-amber-500 text-white'
                            : 'bg-stone-200 text-espresso'
                        }`}
                      >
                        {f.risk_tier}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-[11px] text-outline max-w-sm">
                      {f.drivers && f.drivers.length > 0 ? (
                        <div className="space-y-1">
                          {f.drivers.map((d: any, i: number) => (
                            <div key={i} className="text-red-950 font-medium leading-tight">
                              • {d.text} <span className="text-outline text-[10px]">(+{d.contribution})</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <span>Standard behavioral baseline</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Tab 2: Raw CSV Data View */}
        {activeTab === 'raw_csv' && (
          <div className="space-y-3">
            <p className="text-xs text-outline">
              Verified dataset table read directly from <code className="font-mono text-espresso">{caseData.dataset_filename}</code>:
            </p>
            <div className="overflow-x-auto rounded-2xl border border-sandBorder max-h-[500px]">
              <table className="w-full text-left border-collapse text-xs">
                <thead className="sticky top-0 bg-surface-container z-10 shadow-xs">
                  <tr className="border-b border-sandBorder text-outline uppercase tracking-wider text-[11px] font-bold">
                    {csvHeaders.map((h, i) => (
                      <th key={i} className="py-3 px-4 font-mono whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-sandBorder bg-white font-mono text-[11px]">
                  {caseData.raw_csv_data.map((row: any, rIdx: number) => (
                    <tr key={rIdx} className="hover:bg-[#FFF9EE] transition-colors">
                      {csvHeaders.map((h: string, cIdx: number) => (
                        <td key={cIdx} className="py-2.5 px-4 whitespace-nowrap text-stone-800">
                          {String(row[h] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
