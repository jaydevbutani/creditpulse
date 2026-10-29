import React, { useEffect, useState } from 'react';
import { useSearchParams, NavLink } from 'react-router-dom';
import { api } from '../api';
import { LoanDetailData, CutoffMonth, StressTestResult } from '../types';
import {
  ArrowLeft,
  ArrowRight,
  Calendar,
  AlertCircle,
  Clock,
  TrendingUp,
  Info,
  CheckCircle2,
  FileSpreadsheet,
  Sliders,
  ShieldCheck,
  AlertTriangle,
  Send,
  BookmarkCheck,
  Eye,
  Check,
  Flag,
  UserCheck,
  Activity,
  Shield
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
  ReferenceLine,
  ComposedChart
} from 'recharts';

export const LoanDetail: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialLoanId = parseInt(searchParams.get('loan_id') || '5316', 10);
  const initialCutoff = searchParams.get('cutoff_mi') ? parseInt(searchParams.get('cutoff_mi')!, 10) : undefined;

  const [months, setMonths] = useState<CutoffMonth[]>([]);
  const [selectedCutoff, setSelectedCutoff] = useState<number | undefined>(initialCutoff);
  const [loanId, setLoanId] = useState<number>(initialLoanId);
  const [loanOptions, setLoanOptions] = useState<number[]>([]);
  
  const [data, setData] = useState<LoanDetailData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // What-If Scenario Stress Testing State (Requirement 19)
  const [showStressTest, setShowStressTest] = useState(false);
  const [inflowShock, setInflowShock] = useState(0); // -30 to +30 %
  const [outflowShock, setOutflowShock] = useState(0); // -30 to +30 %
  const [balShock, setBalShock] = useState(0); // -50 to +50 %
  const [paymentDelay, setPaymentDelay] = useState(0); // 0 to 3 months
  const [stressResult, setStressResult] = useState<StressTestResult | null>(null);
  const [runningStress, setRunningStress] = useState(false);

  // Review State & Notes (Requirements 15 & 20)
  const [reviewStatus, setReviewStatus] = useState<string>('normal');
  const [analystNote, setAnalystNote] = useState<string>('');
  const [savingReview, setSavingReview] = useState(false);
  const [saveSuccessMsg, setSaveSuccessMsg] = useState('');

  // Human Review Flagging State
  const [showFlagModal, setShowFlagModal] = useState(false);
  const [flagSeverity, setFlagSeverity] = useState<'Critical' | 'Concern' | 'Watch'>('Concern');
  const [flagReason, setFlagReason] = useState<string>('Cash-flow deficit & falling liquidity reserves');
  const [flagNote, setFlagNote] = useState<string>('');
  const [flaggingSaving, setFlaggingSaving] = useState(false);

  // Dynamic Temporal Slicer & Comparison State
  const [temporalWindow, setTemporalWindow] = useState<'1m' | '3m' | '6m' | '12m' | 'all'>('12m');
  const [comparePeriodsMode, setComparePeriodsMode] = useState<boolean>(false);
  const [compareSpan, setCompareSpan] = useState<number>(3); // 1, 2, 3, or 6 months per period

  useEffect(() => {
    loadMonthsAndQueue();
  }, []);

  useEffect(() => {
    if (loanId) {
      loadLoanData(loanId, selectedCutoff);
    }
  }, [loanId, selectedCutoff]);

  const loadMonthsAndQueue = async () => {
    try {
      const ms = await api.getCutoffMonths();
      setMonths(ms);
      const activeCutoff = selectedCutoff || (ms.length > 0 ? ms[ms.length - 1].cutoff_mi : undefined);
      if (activeCutoff && !selectedCutoff) {
        setSelectedCutoff(activeCutoff);
      }
      
      const mon = await api.getPortfolioMonitor({ cutoff_mi: activeCutoff });
      const lids = mon.queue.map((q) => q.loan_id);
      setLoanOptions(lids);
      if (!lids.includes(loanId) && lids.length > 0) {
        setLoanId(lids[0]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const loadLoanData = async (id: number, cutoff?: number) => {
    try {
      setLoading(true);
      setError('');
      const res = await api.getLoanDetail(id, cutoff);
      setData(res);
      setReviewStatus(res.review?.status || 'normal');
      setAnalystNote(res.review?.note || '');
      setStressResult(null); // reset stress scenario for new loan
    } catch (err: any) {
      setError(err.message || 'Loan data unavailable.');
    } finally {
      setLoading(false);
    }
  };

  const handleRunStressTest = async () => {
    if (!loanId) return;
    try {
      setRunningStress(true);
      const res = await api.runStressTest(loanId, {
        inflow_pct: inflowShock,
        outflow_pct: outflowShock,
        bal_pct: balShock,
        payment_delay_months: paymentDelay,
        cutoff_mi: selectedCutoff
      });
      setStressResult(res);
    } catch (err: any) {
      console.error(err);
    } finally {
      setRunningStress(false);
    }
  };

  const resetStressInputs = () => {
    setInflowShock(0);
    setOutflowShock(0);
    setBalShock(0);
    setPaymentDelay(0);
    setStressResult(null);
  };

  const handleSaveReview = async (newStatus: string) => {
    if (!loanId) return;
    try {
      setSavingReview(true);
      await api.setLoanReview(loanId, newStatus, analystNote);
      setReviewStatus(newStatus);
      setSaveSuccessMsg(`Facility #${loanId} updated to ${newStatus.toUpperCase()}`);
      setTimeout(() => setSaveSuccessMsg(''), 3500);
    } catch (err: any) {
      console.error(err);
    } finally {
      setSavingReview(false);
    }
  };

  const handleSaveFlag = async () => {
    if (!loanId) return;
    try {
      setFlaggingSaving(true);
      await api.flagLoan(loanId, {
        severity: flagSeverity,
        reason: flagReason,
        note: flagNote
      });
      setShowFlagModal(false);
      await loadLoanData(loanId, selectedCutoff);
      setSaveSuccessMsg(`Examiner review flag recorded [${flagSeverity}]: ${flagReason}`);
      setTimeout(() => setSaveSuccessMsg(''), 3500);
    } catch (err: any) {
      console.error(err);
    } finally {
      setFlaggingSaving(false);
    }
  };

  const handleResolveHumanFlag = async () => {
    if (!loanId) return;
    try {
      setFlaggingSaving(true);
      await api.resolveLoanFlag(loanId, 'Examiner verified risk profile within normal operating bounds.');
      await loadLoanData(loanId, selectedCutoff);
      setSaveSuccessMsg(`Facility #${loanId} human flag resolved.`);
      setTimeout(() => setSaveSuccessMsg(''), 3500);
    } catch (err: any) {
      console.error(err);
    } finally {
      setFlaggingSaving(false);
    }
  };

  return (
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Top Header & Breadcrumb */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-sandBorder">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <NavLink to="/" className="text-xs font-bold text-cognac hover:underline flex items-center gap-1">
              <ArrowLeft className="w-3.5 h-3.5" /> Back to Portfolio Monitor
            </NavLink>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-espresso tracking-tight">
            Loan Detail & Forensic Breakdown
          </h1>
          <p className="text-xs text-outline mt-0.5">
            Single-facility multi-task Transformer scoring with 8-month trajectory, discrete hazard timing, and behavioral drivers.
          </p>
        </div>

        {/* Selection Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 bg-surface-container-lowest px-3 py-1.5 rounded-full border border-sandBorder shadow-xs">
            <Calendar className="w-4 h-4 text-cognac" />
            <span className="text-xs font-bold text-espresso">Cutoff:</span>
            <select
              value={selectedCutoff}
              onChange={(e) => setSelectedCutoff(parseInt(e.target.value, 10))}
              className="text-xs font-bold text-espresso bg-transparent border-none focus:ring-0 cursor-pointer"
            >
              {months.map((m) => (
                <option key={m.cutoff_mi} value={m.cutoff_mi}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2 bg-surface-container-lowest px-3 py-1.5 rounded-full border border-sandBorder shadow-xs">
            <span className="text-xs font-bold text-espresso">Loan ID:</span>
            <select
              value={loanId}
              onChange={(e) => setLoanId(parseInt(e.target.value, 10))}
              className="text-xs font-bold text-espresso bg-transparent border-none focus:ring-0 cursor-pointer"
            >
              {loanOptions.map((lid) => (
                <option key={lid} value={lid}>
                  #{lid}
                </option>
              ))}
            </select>
          </div>

          {/* What-If Scenario Stress Testing Toggle Button */}
          <button
            onClick={() => {
              setShowStressTest(!showStressTest);
              if (!showStressTest && !stressResult) handleRunStressTest();
            }}
            className={`px-3 py-1.5 rounded-full text-xs font-bold border transition-all flex items-center gap-1.5 ${
              showStressTest
                ? 'bg-cognac text-white border-cognac shadow-xs'
                : 'bg-surface-container-lowest text-espresso border-sandBorder hover:bg-surface-container'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Scenario Stress Test</span>
          </button>
        </div>
      </div>

      {saveSuccessMsg && (
        <div className="p-3 rounded-2xl bg-emerald-50 border border-emerald-300 text-xs text-emerald-900 font-bold flex items-center gap-2 shadow-xs">
          <CheckCircle2 className="w-4 h-4 text-tertiary shrink-0" />
          <span>{saveSuccessMsg}</span>
        </div>
      )}

      {error ? (
        <div className="p-6 rounded-2xl bg-red-50 border border-red-200 text-error text-xs text-center space-y-2">
          <AlertCircle className="w-6 h-6 mx-auto" />
          <p>{error}</p>
        </div>
      ) : loading && !data ? (
        <div className="py-16 text-center text-xs text-outline">
          <span className="w-8 h-8 border-2 border-cognac border-t-transparent rounded-full animate-spin inline-block mb-3"></span>
          <p>Scoring loan and gathering behavioral tokens...</p>
        </div>
      ) : data ? (
        <div className="space-y-6">
          {/* 4 Headline Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
              <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
                6-Month Risk (Calibrated)
              </span>
              <span className="text-3xl font-extrabold text-error tracking-tight">
                {data.risk_percent}%
              </span>
              <span className="text-[11px] text-outline block mt-2 pt-2 border-t border-sandBorder/40">
                Platt-calibrated probability (Head A)
              </span>
            </div>

            <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
              <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
                Queue Rank
              </span>
              <span className="text-3xl font-extrabold text-espresso tracking-tight">
                #{data.queue_rank} <span className="text-sm font-bold text-outline">/ {data.queue_total}</span>
              </span>
              <span className="text-[11px] text-outline block mt-2 pt-2 border-t border-sandBorder/40">
                In month {data.cutoff_date}
              </span>
            </div>

            <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
              <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
                Percentile
              </span>
              <span className="text-3xl font-extrabold text-cognac tracking-tight">
                top {data.queue_percentile}%
              </span>
              <span className="text-[11px] text-outline block mt-2 pt-2 border-t border-sandBorder/40">
                Rank-based queue urgency
              </span>
            </div>

            <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
              <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
                Loan Age & Status
              </span>
              <div className="flex items-center justify-between">
                <span className="text-3xl font-extrabold text-espresso tracking-tight">
                  {data.loan_age_months}m
                </span>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
                  reviewStatus === 'escalated'
                    ? 'bg-red-100 text-error'
                    : reviewStatus === 'watchlist'
                    ? 'bg-amber-100 text-cognac'
                    : reviewStatus === 'reviewed'
                    ? 'bg-emerald-100 text-tertiary'
                    : 'bg-stone-100 text-stone-600'
                }`}>
                  {reviewStatus.toUpperCase()}
                </span>
              </div>
              <span className="text-[11px] text-outline block mt-2 pt-2 border-t border-sandBorder/40">
                Months since origination (Account #{data.account_id})
              </span>
            </div>
          </div>

          {/* Model Assessment vs Human Assessment - Strict Separation */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Model Assessment Card */}
            <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-sandBorder/60">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cognac" />
                  <span className="font-extrabold text-xs uppercase tracking-wider text-espresso">
                    Algorithmic Model Assessment
                  </span>
                </div>
                <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-outline font-mono text-[11px] font-bold">
                  {data.model_version || 'M6_multitask_ens5'}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-2xl bg-white border border-sandBorder">
                  <span className="text-[10px] text-outline font-bold uppercase block">Calibrated 6M Risk</span>
                  <span className="text-2xl font-extrabold text-espresso mt-0.5 block">{data.risk_percent}%</span>
                  <span className="text-[10px] text-outline">Platt-calibrated logit</span>
                </div>
                <div className="p-3 rounded-2xl bg-white border border-sandBorder">
                  <span className="text-[10px] text-outline font-bold uppercase block">Model Alert Status</span>
                  <span className={`text-base font-extrabold mt-1 block ${data.above_threshold ? 'text-error' : 'text-emerald-700'}`}>
                    {data.above_threshold ? '⚡ DISTRESS ALERT' : '✓ NORMAL'}
                  </span>
                  <span className="text-[10px] text-outline">Alert Threshold: 2.85%</span>
                </div>
              </div>

              {data.risk_trend && (
                <div className="p-3 rounded-2xl bg-white border border-sandBorder text-xs space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-espresso text-[11px]">Trajectory Direction:</span>
                    <span className={`font-mono font-bold text-[11px] ${
                      data.risk_trend.direction === 'Increasing' ? 'text-error' : data.risk_trend.direction === 'Decreasing' ? 'text-tertiary' : 'text-outline'
                    }`}>
                      {data.risk_trend.direction} ({data.risk_trend.delta_pp >= 0 ? `+${data.risk_trend.delta_pp} pp` : `${data.risk_trend.delta_pp} pp`})
                    </span>
                  </div>
                  <p className="text-[11px] text-stone-600 leading-relaxed">{data.risk_trend.description}</p>
                </div>
              )}
            </div>

            {/* Human Examiner Assessment Card */}
            <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-sandBorder/60">
                <div className="flex items-center gap-2">
                  <UserCheck className="w-4 h-4 text-cognac" />
                  <span className="font-extrabold text-xs uppercase tracking-wider text-espresso">
                    Human Examiner Assessment
                  </span>
                </div>
                <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
                  data.agreement === 'Confirmed Concern'
                    ? 'bg-red-100 text-error'
                    : data.agreement === 'Human Override'
                    ? 'bg-purple-100 text-purple-800'
                    : data.agreement === 'Model Alert'
                    ? 'bg-amber-100 text-cognac'
                    : 'bg-emerald-100 text-tertiary'
                }`}>
                  {data.agreement || 'Normal'}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-2xl bg-white border border-sandBorder">
                  <span className="text-[10px] text-outline font-bold uppercase block">Human Flag Status</span>
                  <span className={`text-base font-extrabold mt-1 block ${data.human_flag?.is_flagged ? 'text-error' : 'text-stone-700'}`}>
                    {data.human_flag?.is_flagged ? `🚩 FLAGGED (${data.human_flag.severity})` : 'UNFLAGGED'}
                  </span>
                  <span className="text-[10px] text-outline truncate block">
                    {data.human_flag?.flagged_by ? `By ${data.human_flag.flagged_by}` : 'Routine surveillance'}
                  </span>
                </div>
                <div className="p-3 rounded-2xl bg-white border border-sandBorder">
                  <span className="text-[10px] text-outline font-bold uppercase block">Agreement Quadrant</span>
                  <span className="text-sm font-extrabold text-espresso mt-1 block truncate">
                    {data.agreement}
                  </span>
                  <span className="text-[10px] text-outline">Model vs Examiner</span>
                </div>
              </div>

              {data.human_flag?.is_flagged ? (
                <div className="p-3 rounded-2xl bg-amber-50/70 border border-amber-300 text-xs space-y-1">
                  <strong className="font-bold text-espresso text-[11px] block">{data.human_flag.reason}</strong>
                  {data.human_flag.note && (
                    <p className="text-[11px] text-stone-700 italic leading-relaxed">{data.human_flag.note}</p>
                  )}
                  <div className="pt-2 flex items-center justify-between">
                    <button
                      onClick={() => setShowFlagModal(true)}
                      className="text-[11px] text-cognac font-bold hover:underline"
                    >
                      Edit Flag Severity / Reason
                    </button>
                    <button
                      onClick={handleResolveHumanFlag}
                      disabled={flaggingSaving}
                      className="px-3 py-1 rounded-full bg-white border border-emerald-300 text-emerald-800 text-[11px] font-bold hover:bg-emerald-600 hover:text-white transition-all"
                    >
                      Resolve Flag ✓
                    </button>
                  </div>
                </div>
              ) : (
                <div className="p-3 rounded-2xl bg-white border border-sandBorder text-xs flex items-center justify-between">
                  <span className="text-[11px] text-stone-600">No examiner distress concern flagged.</span>
                  <button
                    onClick={() => setShowFlagModal(true)}
                    className="px-3 py-1.5 rounded-full bg-espresso text-white text-xs font-bold hover:bg-stone-800 transition-all flex items-center gap-1.5"
                  >
                    <Flag className="w-3.5 h-3.5 text-amber-400" />
                    <span>Flag Facility</span>
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Decision Support Guidance Card */}
          {data.decision_support_guidance && (
            <div className="p-5 rounded-3xl bg-[#FAF6ED] border-2 border-cognac/30 shadow-warm space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-cognac" />
                  <span className="font-bold text-xs uppercase tracking-wider text-espresso">
                    Decision-Support Protocol ({data.decision_support_guidance.category})
                  </span>
                </div>
                <span className="text-[11px] font-bold text-outline">
                  Timeline: {data.decision_support_guidance.suggested_timeline}
                </span>
              </div>
              <p className="text-xs text-stone-800 font-semibold leading-relaxed">
                {data.decision_support_guidance.summary}
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                <div className="p-3 rounded-2xl bg-white border border-sandBorder text-xs">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-cognac block mb-0.5">
                    Recommended Action
                  </span>
                  <p className="text-stone-700 leading-relaxed font-medium">
                    {data.decision_support_guidance.recommended_action}
                  </p>
                </div>
                <div className="p-3 rounded-2xl bg-white border border-sandBorder text-xs">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-outline block mb-0.5">
                    Supervisory & CECL Consideration
                  </span>
                  <p className="text-stone-700 leading-relaxed font-medium">
                    {data.decision_support_guidance.supervisory_consideration}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* What-If Scenario Simulation / Stress Test Card (Requirement 19) */}
          {showStressTest && (
            <div className="bg-[#FFFDF9] rounded-3xl p-6 border border-amber-300 shadow-warm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-sandBorder/60">
                <div>
                  <div className="flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-cognac" />
                    <h2 className="text-base font-bold text-espresso tracking-tight">
                      What-If Scenario Stress Testing
                    </h2>
                    <span className="px-2 py-0.5 rounded-full bg-amber-100 text-cognac font-mono text-[10px] font-bold">
                      SCENARIO SIMULATION
                    </span>
                  </div>
                  <p className="text-xs text-outline mt-0.5">
                    Simulate cash flow or liquidity shocks to observe hypothetical distress elasticity. Original baseline prediction remains unchanged.
                  </p>
                </div>

                <button
                  onClick={resetStressInputs}
                  className="text-xs font-bold text-outline hover:text-espresso"
                >
                  Reset Shocks
                </button>
              </div>

              {/* Sliders Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-1">
                <div className="space-y-1.5 p-3 rounded-2xl bg-white border border-sandBorder">
                  <div className="flex justify-between text-xs font-bold">
                    <span className="text-outline">Monthly Inflow:</span>
                    <span className={inflowShock < 0 ? 'text-error' : inflowShock > 0 ? 'text-tertiary' : 'text-espresso'}>
                      {inflowShock > 0 ? `+${inflowShock}%` : `${inflowShock}%`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-30"
                    max="30"
                    step="5"
                    value={inflowShock}
                    onChange={(e) => setInflowShock(parseInt(e.target.value, 10))}
                    className="w-full accent-cognac cursor-pointer"
                  />
                  <span className="text-[10px] text-outline block">Operating cash receipt stress</span>
                </div>

                <div className="space-y-1.5 p-3 rounded-2xl bg-white border border-sandBorder">
                  <div className="flex justify-between text-xs font-bold">
                    <span className="text-outline">Monthly Outflow:</span>
                    <span className={outflowShock > 0 ? 'text-error' : outflowShock < 0 ? 'text-tertiary' : 'text-espresso'}>
                      {outflowShock > 0 ? `+${outflowShock}%` : `${outflowShock}%`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-30"
                    max="30"
                    step="5"
                    value={outflowShock}
                    onChange={(e) => setOutflowShock(parseInt(e.target.value, 10))}
                    className="w-full accent-cognac cursor-pointer"
                  />
                  <span className="text-[10px] text-outline block">Disbursement expenditure surge</span>
                </div>

                <div className="space-y-1.5 p-3 rounded-2xl bg-white border border-sandBorder">
                  <div className="flex justify-between text-xs font-bold">
                    <span className="text-outline">Available Balance:</span>
                    <span className={balShock < 0 ? 'text-error' : balShock > 0 ? 'text-tertiary' : 'text-espresso'}>
                      {balShock > 0 ? `+${balShock}%` : `${balShock}%`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-50"
                    max="50"
                    step="5"
                    value={balShock}
                    onChange={(e) => setBalShock(parseInt(e.target.value, 10))}
                    className="w-full accent-cognac cursor-pointer"
                  />
                  <span className="text-[10px] text-outline block">Liquidity reserve draw-down</span>
                </div>

                <div className="space-y-1.5 p-3 rounded-2xl bg-white border border-sandBorder">
                  <div className="flex justify-between text-xs font-bold">
                    <span className="text-outline">Payment Delay:</span>
                    <span className={paymentDelay > 0 ? 'text-error' : 'text-espresso'}>
                      +{paymentDelay} mo
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="3"
                    step="1"
                    value={paymentDelay}
                    onChange={(e) => setPaymentDelay(parseInt(e.target.value, 10))}
                    className="w-full accent-cognac cursor-pointer"
                  />
                  <span className="text-[10px] text-outline block">Hypothetical installment shock</span>
                </div>
              </div>

              {/* Stress Action & Results Panel */}
              <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <button
                  onClick={handleRunStressTest}
                  disabled={runningStress}
                  className="px-4 py-2 rounded-full bg-espresso text-white text-xs font-bold hover:bg-stone-800 transition-colors flex items-center gap-2 self-start"
                >
                  {runningStress ? 'Computing Scenario...' : 'Recalculate Stress Scenario'}
                </button>

                {stressResult && (
                  <div className="flex items-center gap-6 p-3 rounded-2xl bg-white border border-amber-300">
                    <div>
                      <span className="text-[10px] text-outline font-bold uppercase block">Baseline Risk</span>
                      <span className="text-base font-extrabold text-espresso">{stressResult.base_risk_percent}%</span>
                    </div>
                    <ArrowRight className="w-4 h-4 text-cognac" />
                    <div>
                      <span className="text-[10px] text-outline font-bold uppercase block">Stressed Risk</span>
                      <span className={`text-base font-extrabold ${stressResult.risk_delta_percent > 0 ? 'text-error' : 'text-tertiary'}`}>
                        {stressResult.stressed_risk_percent}%
                      </span>
                    </div>
                    <div className="pl-3 border-l border-sandBorder/50">
                      <span className="text-[10px] text-outline font-bold uppercase block">Scenario Shift</span>
                      <span className="text-xs font-bold text-error">
                        {stressResult.risk_delta_percent >= 0 ? `+${stressResult.risk_delta_percent}%` : `${stressResult.risk_delta_percent}%`}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Two Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Chart 1: Risk Trajectory (Last 8 Months) */}
            <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm">
              <h2 className="text-base font-bold text-espresso tracking-tight mb-1">
                Risk Trajectory (Last 8 Months)
              </h2>
              <p className="text-xs text-outline mb-4">
                Historical monthly probability progression vs 10% queue alert threshold.
              </p>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.trajectory}>
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
                    <ReferenceLine
                      y={data.trajectory[0]?.threshold || 2.8}
                      stroke="#DC2626"
                      strokeDasharray="4 4"
                      label={{ value: '10% queue threshold', fill: '#DC2626', fontSize: 10, position: 'top' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="risk_6m"
                      stroke="#B45309"
                      strokeWidth={3}
                      dot={{ r: 4, fill: '#B45309' }}
                      name="6-Month Risk %"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 2: When the Risk Falls: Next 6 Months */}
            <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm">
              <h2 className="text-base font-bold text-espresso tracking-tight mb-1">
                When the Risk Falls: Next 6 Months (Head B)
              </h2>
              <p className="text-xs text-outline mb-4">
                Monthly discrete hazard rate (bar) and cumulative incidence curve (line).
              </p>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={data.hazard_breakdown}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE1" />
                    <XAxis dataKey="horizon" stroke="#867461" fontSize={11} />
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
                    <Bar dataKey="monthly_hazard" fill="#B45309" radius={[6, 6, 0, 0]} name="Monthly Hazard %" />
                    <Line type="monotone" dataKey="cumulative_risk" stroke="#DC2626" strokeWidth={2} dot={{ r: 3, fill: '#DC2626' }} name="Cumulative %" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* "What Changed?" Forensic Section (Requirement 14) */}
          <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-cognac" />
              <h2 className="text-base font-bold text-espresso tracking-tight">
                What Changed? Behavioral Shift Diagnostics
              </h2>
            </div>
            <p className="text-xs text-outline">
              Automated behavioral change detection derived directly from the borrower's recent 12-month sequence reconciliation.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
              {data.what_changed?.map((item, idx) => (
                <div
                  key={idx}
                  className={`p-3.5 rounded-2xl border text-xs ${
                    item.severity === 'Critical'
                      ? 'bg-red-50 border-red-200'
                      : item.severity === 'High'
                      ? 'bg-[#FFF9EE] border-amber-200'
                      : 'bg-white border-sandBorder'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-espresso uppercase tracking-wider text-[10px]">
                      {item.category}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        item.severity === 'Critical'
                          ? 'bg-red-100 text-error'
                          : item.severity === 'High'
                          ? 'bg-amber-100 text-cognac'
                          : 'bg-emerald-50 text-tertiary'
                      }`}
                    >
                      {item.severity}
                    </span>
                  </div>
                  <p className="text-stone-800 leading-relaxed font-medium">{item.text}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Improved Attribution Explanations: Risk-Increasing vs Protective Factors (Requirement 13) */}
          <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
            <div>
              <h2 className="text-base font-bold text-espresso tracking-tight">
                Attribution Analysis (Integrated Gradients)
              </h2>
              <p className="text-xs text-outline mt-0.5">
                Feature attributions computed by path-integrated gradients across the multi-task Transformer seed ensemble.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-1">
              {/* Risk-Increasing Factors */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-error" />
                  <span className="text-xs font-bold uppercase tracking-wider text-error">
                    Risk-Increasing Attribution Drivers
                  </span>
                </div>
                <div className="space-y-2">
                  {(data.risk_increasing_drivers || data.drivers)?.map((d, i) => (
                    <div
                      key={i}
                      className="p-3.5 rounded-2xl bg-[#FFF9EE] border-l-4 border-cognac border-t border-r border-b border-sandBorder/70 text-xs flex items-center justify-between"
                    >
                      <div>
                        <span className="font-bold text-espresso text-sm block">{d.text}</span>
                        <span className="text-[11px] text-outline font-mono mt-0.5 block">
                          Feature: <code>{d.feature}</code> • Observed: {d.value?.toLocaleString()}
                        </span>
                      </div>
                      <span className="px-2.5 py-1 rounded-full bg-amber-100 text-cognac font-bold font-mono text-xs shrink-0">
                        Contrib: +{d.contribution}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Protective Factors */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-tertiary" />
                  <span className="text-xs font-bold uppercase tracking-wider text-tertiary">
                    Protective Mitigating Factors
                  </span>
                </div>
                <div className="space-y-2">
                  {data.protective_drivers && data.protective_drivers.length > 0 ? (
                    data.protective_drivers.map((d, i) => (
                      <div
                        key={i}
                        className="p-3.5 rounded-2xl bg-[#F6FAF7] border-l-4 border-emerald-600 border-t border-r border-b border-sandBorder/70 text-xs flex items-center justify-between"
                      >
                        <div>
                          <span className="font-bold text-espresso text-sm block">{d.text}</span>
                          <span className="text-[11px] text-outline font-mono mt-0.5 block">
                            Feature: <code>{d.feature}</code> • Observed: {d.value?.toLocaleString()}
                          </span>
                        </div>
                        <span className="px-2.5 py-1 rounded-full bg-emerald-100 text-tertiary font-bold font-mono text-xs shrink-0">
                          Contrib: {d.contribution}
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className="p-4 rounded-2xl bg-surface-container text-xs text-outline">
                      No significant protective offsets identified in this sequence.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Recommended Review Actions & Analyst Notes (Requirements 15 & 20) */}
          <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-espresso tracking-tight">
                  Recommended Surveillance Actions & Analyst Review
                </h2>
                <p className="text-xs text-outline mt-0.5">
                  Institutional workflow governance: Record formal review disposition and audit notes for supervisory record.
                </p>
              </div>
              <span className="text-xs font-mono text-outline">
                Current Status: <strong className="text-espresso font-bold">{reviewStatus.toUpperCase()}</strong>
              </span>
            </div>

            {/* Quick Action Buttons */}
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <button
                onClick={() => handleSaveReview('reviewed')}
                disabled={savingReview}
                className={`px-4 py-2 rounded-full text-xs font-bold transition-all flex items-center gap-1.5 ${
                  reviewStatus === 'reviewed'
                    ? 'bg-emerald-800 text-white shadow-xs'
                    : 'bg-emerald-50 text-emerald-900 hover:bg-emerald-100 border border-emerald-200'
                }`}
              >
                <Check className="w-3.5 h-3.5" />
                <span>Mark Reviewed</span>
              </button>

              <button
                onClick={() => handleSaveReview('watchlist')}
                disabled={savingReview}
                className={`px-4 py-2 rounded-full text-xs font-bold transition-all flex items-center gap-1.5 ${
                  reviewStatus === 'watchlist'
                    ? 'bg-amber-600 text-white shadow-xs'
                    : 'bg-amber-50 text-cognac hover:bg-amber-100 border border-amber-200'
                }`}
              >
                <BookmarkCheck className="w-3.5 h-3.5" />
                <span>Add to Watchlist</span>
              </button>

              <button
                onClick={() => handleSaveReview('escalated')}
                disabled={savingReview}
                className={`px-4 py-2 rounded-full text-xs font-bold transition-all flex items-center gap-1.5 ${
                  reviewStatus === 'escalated'
                    ? 'bg-red-700 text-white shadow-xs'
                    : 'bg-red-50 text-error hover:bg-red-100 border border-red-200'
                }`}
              >
                <AlertCircle className="w-3.5 h-3.5" />
                <span>Escalate Facility</span>
              </button>

              <button
                onClick={() => handleSaveReview('normal')}
                disabled={savingReview}
                className="px-3 py-2 rounded-full text-xs font-bold text-outline hover:text-espresso"
              >
                Reset to Normal
              </button>
            </div>

            {/* Analyst Note Input */}
            <div className="pt-2 space-y-2">
              <label className="text-xs font-bold text-espresso block">
                Analyst Working Note & Supervisory Comments:
              </label>
              <div className="flex gap-2">
                <textarea
                  rows={2}
                  value={analystNote}
                  onChange={(e) => setAnalystNote(e.target.value)}
                  placeholder="e.g., Customer ending balance has declined for three consecutive months. Verified payroll continuity and scheduled loan service capacity."
                  className="w-full text-xs p-3 rounded-2xl bg-surface-container-low border border-sandBorder focus:ring-1 focus:ring-cognac"
                />
                <button
                  onClick={() => handleSaveReview(reviewStatus)}
                  disabled={savingReview}
                  className="px-5 py-2 rounded-2xl bg-espresso text-white text-xs font-bold hover:bg-stone-800 transition-colors shrink-0 flex items-center gap-1.5"
                >
                  <Send className="w-3.5 h-3.5" />
                  <span>{savingReview ? 'Saving...' : 'Save Note'}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Monthly Behaviour Behind the Score */}
          {(() => {
            const allBehavior = data.behavior || [];
            const getSlicedBehavior = () => {
              if (temporalWindow === '1m') return allBehavior.slice(-1);
              if (temporalWindow === '3m') return allBehavior.slice(-3);
              if (temporalWindow === '6m') return allBehavior.slice(-6);
              if (temporalWindow === '12m') return allBehavior.slice(-12);
              return allBehavior;
            };

            const slicedBehavior = getSlicedBehavior();

            // Sliced Stats
            const totalInflow = slicedBehavior.reduce((sum, b) => sum + b.inflow, 0);
            const totalOutflow = slicedBehavior.reduce((sum, b) => sum + b.outflow, 0);
            const avgBal = slicedBehavior.length
              ? Math.round(slicedBehavior.reduce((sum, b) => sum + b.bal_eom, 0) / slicedBehavior.length)
              : 0;
            const minBal = slicedBehavior.length
              ? Math.min(...slicedBehavior.map((b) => b.bal_min))
              : 0;
            const totalDaysBelow = slicedBehavior.reduce((sum, b) => sum + b.days_below_zero, 0);
            const totalTx = slicedBehavior.reduce((sum, b) => sum + b.n_tx, 0);

            // Period Comparison
            // Period B = recent compareSpan months; Period A = prior compareSpan months
            const periodBRows = allBehavior.slice(-compareSpan);
            const periodARows = allBehavior.slice(-(compareSpan * 2), -compareSpan);

            const calcPeriodStats = (rows: typeof allBehavior) => {
              const count = rows.length;
              const range = count > 0 ? `${rows[0].month} – ${rows[rows.length - 1].month}` : 'N/A';
              const avgB = count ? Math.round(rows.reduce((sum, b) => sum + b.bal_eom, 0) / count) : 0;
              const inF = rows.reduce((sum, b) => sum + b.inflow, 0);
              const outF = rows.reduce((sum, b) => sum + b.outflow, 0);
              const minB = count ? Math.min(...rows.map((b) => b.bal_min)) : 0;
              const daysUnder = rows.reduce((sum, b) => sum + b.days_below_zero, 0);
              const tx = rows.reduce((sum, b) => sum + b.n_tx, 0);
              return { count, range, avgB, inF, outF, minB, daysUnder, tx, net: inF - outF };
            };

            const pA = calcPeriodStats(periodARows);
            const pB = calcPeriodStats(periodBRows);

            return (
              <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                  <div>
                    <h2 className="text-base font-bold text-espresso tracking-tight">
                      Monthly Behaviour Behind the Score
                    </h2>
                    <p className="text-xs text-outline mt-0.5">
                      Actual account transactions, temporal feature tokens, and balance dynamics from the Berka panel dataset.
                    </p>
                  </div>
                  <span className="px-3 py-1 rounded-full bg-surface-container text-outline font-mono text-xs font-semibold self-start md:self-auto">
                    Account #{data.account_id} · {allBehavior.length} Months Total
                  </span>
                </div>

                {/* Dynamic Temporal Slicer & Comparison Controls */}
                <div className="p-3.5 rounded-2xl bg-surface-container-low border border-sandBorder/60 flex flex-col md:flex-row md:items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-[11px] font-bold text-espresso uppercase tracking-wider mr-1">
                      Time Horizon:
                    </span>
                    {(['1m', '3m', '6m', '12m', 'all'] as const).map((w) => (
                      <button
                        key={w}
                        onClick={() => {
                          setTemporalWindow(w);
                          if (comparePeriodsMode) setComparePeriodsMode(false);
                        }}
                        className={`px-3 py-1 rounded-full text-xs font-bold transition-all ${
                          temporalWindow === w && !comparePeriodsMode
                            ? 'bg-espresso text-white shadow-xs'
                            : 'bg-white text-stone-700 hover:bg-stone-100 border border-sandBorder/60'
                        }`}
                      >
                        {w === '1m'
                          ? 'Last 1 Month'
                          : w === '3m'
                          ? 'Last 3 Months'
                          : w === '6m'
                          ? 'Last 6 Months'
                          : w === '12m'
                          ? 'Last 12 Months'
                          : 'All Available'}
                      </button>
                    ))}
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setComparePeriodsMode(!comparePeriodsMode)}
                      className={`px-3 py-1 rounded-full text-xs font-bold flex items-center gap-1.5 transition-all ${
                        comparePeriodsMode
                          ? 'bg-amber-600 text-white shadow-xs'
                          : 'bg-white text-espresso hover:bg-stone-100 border border-sandBorder/60'
                      }`}
                    >
                      <span>Compare Periods</span>
                      <span className="text-[10px] font-mono px-1 rounded bg-black/10">
                        {comparePeriodsMode ? 'ON' : 'OFF'}
                      </span>
                    </button>

                    {comparePeriodsMode && (
                      <select
                        value={compareSpan}
                        onChange={(e) => setCompareSpan(parseInt(e.target.value, 10))}
                        className="px-2.5 py-1 rounded-full bg-white text-xs font-bold text-espresso border border-sandBorder/80 cursor-pointer"
                      >
                        <option value={1}>1 Mo vs 1 Mo</option>
                        <option value={2}>2 Mo vs 2 Mo</option>
                        <option value={3}>3 Mo vs 3 Mo</option>
                        <option value={6}>6 Mo vs 6 Mo</option>
                      </select>
                    )}
                  </div>
                </div>

                {/* Sliced Summary Cards or Period Comparison Grid */}
                {!comparePeriodsMode ? (
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 text-xs">
                    <div className="p-3 rounded-2xl bg-white border border-sandBorder/60">
                      <span className="text-[10px] text-outline font-bold uppercase block">Avg Balance EOM</span>
                      <span className="font-mono font-bold text-espresso text-sm mt-0.5 block">{avgBal.toLocaleString()} CZK</span>
                      <span className="text-[10px] text-outline block mt-0.5">{slicedBehavior.length} months window</span>
                    </div>

                    <div className="p-3 rounded-2xl bg-white border border-sandBorder/60">
                      <span className="text-[10px] text-outline font-bold uppercase block">Total Inflow</span>
                      <span className="font-mono font-bold text-emerald-800 text-sm mt-0.5 block">+{totalInflow.toLocaleString()} CZK</span>
                      <span className="text-[10px] text-outline block mt-0.5">Aggregate deposits</span>
                    </div>

                    <div className="p-3 rounded-2xl bg-white border border-sandBorder/60">
                      <span className="text-[10px] text-outline font-bold uppercase block">Total Outflow</span>
                      <span className="font-mono font-bold text-red-800 text-sm mt-0.5 block">-{totalOutflow.toLocaleString()} CZK</span>
                      <span className="text-[10px] text-outline block mt-0.5">Aggregate payments</span>
                    </div>

                    <div className="p-3 rounded-2xl bg-white border border-sandBorder/60">
                      <span className="text-[10px] text-outline font-bold uppercase block">Net Cash Flow</span>
                      <span className={`font-mono font-bold text-sm mt-0.5 block ${totalInflow - totalOutflow >= 0 ? 'text-emerald-800' : 'text-error'}`}>
                        {(totalInflow - totalOutflow >= 0 ? '+' : '') + (totalInflow - totalOutflow).toLocaleString()} CZK
                      </span>
                      <span className="text-[10px] text-outline block mt-0.5">Inflow minus outflow</span>
                    </div>

                    <div className="p-3 rounded-2xl bg-white border border-sandBorder/60">
                      <span className="text-[10px] text-outline font-bold uppercase block">Min Balance</span>
                      <span className={`font-mono font-bold text-sm mt-0.5 block ${minBal < 0 ? 'text-error' : 'text-espresso'}`}>
                        {minBal.toLocaleString()} CZK
                      </span>
                      <span className="text-[10px] text-outline block mt-0.5">Trough liquidity</span>
                    </div>

                    <div className="p-3 rounded-2xl bg-white border border-sandBorder/60">
                      <span className="text-[10px] text-outline font-bold uppercase block">Days Below Zero</span>
                      <span className={`font-mono font-bold text-sm mt-0.5 block ${totalDaysBelow > 0 ? 'text-error' : 'text-espresso'}`}>
                        {totalDaysBelow} days
                      </span>
                      <span className="text-[10px] text-outline block mt-0.5">{totalTx} transactions</span>
                    </div>
                  </div>
                ) : (
                  /* Two-Period Comparative Diagnostic Cards */
                  <div className="p-4 rounded-2xl bg-amber-50/70 border border-amber-200 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-cognac" />
                        <span className="text-xs font-bold text-espresso uppercase tracking-wider">
                          Period A vs Period B Behavioral Comparison ({compareSpan}-Month Windows)
                        </span>
                      </div>
                      <span className="text-[11px] font-mono font-bold text-cognac">
                        Prior: {pA.range} → Recent: {pB.range}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                      {/* Metric 1: Average Balance */}
                      <div className="p-3 rounded-xl bg-white border border-sandBorder/60 space-y-1">
                        <span className="text-[10px] text-outline font-bold uppercase block">Avg Balance</span>
                        <div className="flex items-baseline justify-between font-mono">
                          <span className="text-stone-600">{pA.avgB.toLocaleString()} CZK</span>
                          <span className="text-espresso font-bold text-sm">→ {pB.avgB.toLocaleString()} CZK</span>
                        </div>
                        <div className="flex items-center justify-between text-[10px] pt-1 border-t border-sandBorder/30">
                          <span className="text-outline">Delta</span>
                          <span className={`font-mono font-bold ${pB.avgB - pA.avgB < 0 ? 'text-error' : 'text-emerald-700'}`}>
                            {pB.avgB - pA.avgB < 0 ? '' : '+'}{((pB.avgB - pA.avgB) / (Math.abs(pA.avgB) || 1) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      {/* Metric 2: Net Cashflow */}
                      <div className="p-3 rounded-xl bg-white border border-sandBorder/60 space-y-1">
                        <span className="text-[10px] text-outline font-bold uppercase block">Net Cash Flow (In - Out)</span>
                        <div className="flex items-baseline justify-between font-mono">
                          <span className="text-stone-600">{pA.net.toLocaleString()} CZK</span>
                          <span className={`font-bold text-sm ${pB.net < 0 ? 'text-error' : 'text-emerald-800'}`}>
                            → {pB.net.toLocaleString()} CZK
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-[10px] pt-1 border-t border-sandBorder/30">
                          <span className="text-outline">Shift</span>
                          <span className={`font-mono font-bold ${pB.net < pA.net ? 'text-error' : 'text-emerald-700'}`}>
                            {(pB.net - pA.net).toLocaleString()} CZK delta
                          </span>
                        </div>
                      </div>

                      {/* Metric 3: Overdraft Stress */}
                      <div className="p-3 rounded-xl bg-white border border-sandBorder/60 space-y-1">
                        <span className="text-[10px] text-outline font-bold uppercase block">Overdraft Days & Min Bal</span>
                        <div className="flex items-baseline justify-between font-mono">
                          <span className="text-stone-600">{pA.daysUnder}d (min {pA.minB.toLocaleString()})</span>
                          <span className={`font-bold text-sm ${pB.daysUnder > pA.daysUnder ? 'text-error' : 'text-espresso'}`}>
                            → {pB.daysUnder}d (min {pB.minB.toLocaleString()})
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-[10px] pt-1 border-t border-sandBorder/30">
                          <span className="text-outline">Overdraft Change</span>
                          <span className={`font-mono font-bold ${pB.daysUnder > pA.daysUnder ? 'text-error' : 'text-emerald-700'}`}>
                            {pB.daysUnder - pA.daysUnder > 0 ? `+${pB.daysUnder - pA.daysUnder} days stress` : 'No overdraft increase'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Behavioral Monthly Table */}
                <div className="overflow-x-auto rounded-2xl border border-sandBorder">
                  <table className="w-full text-left border-collapse text-xs font-mono">
                    <thead>
                      <tr className="bg-surface-container border-b border-sandBorder text-outline uppercase tracking-wider text-[11px] font-bold font-sans">
                        <th className="py-3 px-4">Month</th>
                        {comparePeriodsMode && <th className="py-3 px-2">Period</th>}
                        <th className="py-3 px-4">Inflow</th>
                        <th className="py-3 px-4">Outflow</th>
                        <th className="py-3 px-4">Bal EOM</th>
                        <th className="py-3 px-4">Bal Min</th>
                        <th className="py-3 px-4">Days &lt; 0</th>
                        <th className="py-3 px-4">Loan Payment</th>
                        <th className="py-3 px-4">Tx Count</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-sandBorder bg-white text-[11px]">
                      {(comparePeriodsMode ? [...periodARows, ...periodBRows] : slicedBehavior).map((b, i) => {
                        const isPeriodB = comparePeriodsMode && periodBRows.includes(b);
                        const isPeriodA = comparePeriodsMode && periodARows.includes(b);
                        return (
                          <tr
                            key={i}
                            className={`transition-colors ${
                              isPeriodB
                                ? 'bg-amber-50/60 font-medium hover:bg-amber-100/50'
                                : isPeriodA
                                ? 'bg-blue-50/50 hover:bg-blue-100/40'
                                : 'hover:bg-[#FFF9EE]'
                            }`}
                          >
                            <td className="py-2.5 px-4 font-bold text-espresso">{b.month}</td>
                            {comparePeriodsMode && (
                              <td className="py-2.5 px-2 font-sans">
                                {isPeriodB ? (
                                  <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-200 text-cognac">
                                    Period B (Recent)
                                  </span>
                                ) : isPeriodA ? (
                                  <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-blue-100 text-blue-800">
                                    Period A (Prior)
                                  </span>
                                ) : null}
                              </td>
                            )}
                            <td className="py-2.5 px-4 text-emerald-800">+{b.inflow.toLocaleString()}</td>
                            <td className="py-2.5 px-4 text-red-800">-{b.outflow.toLocaleString()}</td>
                            <td className="py-2.5 px-4 font-bold text-espresso">{b.bal_eom.toLocaleString()}</td>
                            <td className={`py-2.5 px-4 font-bold ${b.bal_min < 0 ? 'text-error' : 'text-stone-700'}`}>
                              {b.bal_min.toLocaleString()}
                            </td>
                            <td className={`py-2.5 px-4 font-bold ${b.days_below_zero > 0 ? 'text-error' : 'text-stone-700'}`}>
                              {b.days_below_zero}
                            </td>
                            <td className="py-2.5 px-4 text-stone-700">{b.loan_payment.toLocaleString()}</td>
                            <td className="py-2.5 px-4 text-stone-700">{b.n_tx}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })()}
        </div>
      ) : null}

      {/* Flag for Human Review Modal */}
      {showFlagModal && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl p-6 max-w-lg w-full space-y-4 shadow-xl border border-sandBorder animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between pb-2 border-b border-sandBorder/60">
              <div className="flex items-center gap-2">
                <Flag className="w-5 h-5 text-error" />
                <h3 className="font-extrabold text-base text-espresso">
                  Flag Facility #{loanId} for Examiner Review
                </h3>
              </div>
              <button
                onClick={() => setShowFlagModal(false)}
                className="text-stone-400 hover:text-stone-700 text-sm font-bold"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="font-bold text-espresso block mb-1">Severity Level</label>
                <div className="grid grid-cols-3 gap-2">
                  {(['Watch', 'Concern', 'Critical'] as const).map((sev) => (
                    <button
                      key={sev}
                      type="button"
                      onClick={() => setFlagSeverity(sev)}
                      className={`py-2 rounded-xl font-bold border transition-all ${
                        flagSeverity === sev
                          ? sev === 'Critical'
                            ? 'bg-red-600 text-white border-red-600 shadow-xs'
                            : sev === 'Concern'
                            ? 'bg-amber-600 text-white border-amber-600 shadow-xs'
                            : 'bg-stone-800 text-white border-stone-800 shadow-xs'
                          : 'bg-surface-container-low text-espresso border-sandBorder hover:bg-surface-container'
                      }`}
                    >
                      {sev}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="font-bold text-espresso block mb-1">Examiner Review Reason</label>
                <select
                  value={flagReason}
                  onChange={(e) => setFlagReason(e.target.value)}
                  className="w-full text-xs p-2.5 rounded-xl bg-surface-container-low border border-sandBorder text-espresso focus:ring-1 focus:ring-cognac"
                >
                  <option value="Cash-flow deficit & falling liquidity reserves">
                    Cash-flow deficit & falling liquidity reserves
                  </option>
                  <option value="Acute credit distress & multiple negative balance days">
                    Acute credit distress & multiple negative balance days
                  </option>
                  <option value="Customer sector volatility in fleet logistics">
                    Customer sector volatility in fleet logistics
                  </option>
                  <option value="Disbursement acceleration exceeding operating turnover">
                    Disbursement acceleration exceeding operating turnover
                  </option>
                  <option value="Qualitative debt service impairment / borrower dispute">
                    Qualitative debt service impairment / borrower dispute
                  </option>
                </select>
              </div>

              <div>
                <label className="font-bold text-espresso block mb-1">Examiner Diagnostic Working Note</label>
                <textarea
                  rows={3}
                  value={flagNote}
                  onChange={(e) => setFlagNote(e.target.value)}
                  placeholder="Record specific borrower telemetry, covenant breaches, or examiner oversight rationale..."
                  className="w-full text-xs p-2.5 rounded-xl bg-surface-container-low border border-sandBorder text-espresso focus:ring-1 focus:ring-cognac"
                />
              </div>
            </div>

            <div className="pt-2 flex justify-end gap-2 border-t border-sandBorder/40">
              <button
                onClick={() => setShowFlagModal(false)}
                className="px-4 py-2 rounded-full text-xs font-bold text-outline hover:text-espresso"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveFlag}
                disabled={flaggingSaving}
                className="px-5 py-2 rounded-full bg-espresso text-white text-xs font-bold hover:bg-stone-800 transition-colors shadow-xs"
              >
                {flaggingSaving ? 'Recording...' : 'Save Examiner Flag'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
