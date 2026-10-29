import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { PortfolioMonitorData, CutoffMonth, SimulationStreamData } from '../types';
import {
  Activity,
  AlertTriangle,
  TrendingUp,
  Clock,
  ArrowRight,
  Info,
  Calendar,
  CheckCircle2,
  ChevronRight,
  SlidersHorizontal,
  Layers,
  ArrowUpDown,
  Radio,
  Play,
  Pause,
  RotateCcw,
  ShieldAlert,
  Zap,
  Filter,
  Check,
  StepForward,
  UserCheck,
  Flag,
  ShieldCheck,
  RefreshCw
} from 'lucide-react';

export const PortfolioDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [months, setMonths] = useState<CutoffMonth[]>([]);
  const [selectedCutoff, setSelectedCutoff] = useState<number | undefined>(undefined);
  
  // Date/Month range state & presets (Requirements 2 & 4)
  const [rangePreset, setRangePreset] = useState<'1m' | '3m' | '6m' | '12m' | 'custom'>('1m');
  const [rangeStart, setRangeStart] = useState<number | undefined>(undefined);
  
  // Two-Period Comparison state (Requirement 3)
  const [compareMode, setCompareMode] = useState(false);
  const [compareCutoff, setCompareCutoff] = useState<number | undefined>(undefined);

  // Filters (Requirements 16 & 17)
  const [tierFilter, setTierFilter] = useState<string | null>(null);
  const [segmentFilter, setSegmentFilter] = useState<string | null>(null);
  const [warningFilter, setWarningFilter] = useState<string | null>(null);
  const [agreementFilter, setAgreementFilter] = useState<string | null>(null);

  // Live Simulation state (Requirements 9, 10, 11, 12)
  const [showSimulation, setShowSimulation] = useState(false);
  const [simRunning, setSimRunning] = useState(false);
  const [simSpeed, setSimSpeed] = useState<'1x' | '5x' | '10x'>('1x');
  const [driftScenario, setDriftScenario] = useState<'normal' | 'warning' | 'critical'>('normal');
  const [simData, setSimData] = useState<SimulationStreamData | null>(null);
  const [simPersistedRecords, setSimPersistedRecords] = useState<any[]>([]);
  const [simProcessing, setSimProcessing] = useState(false);
  const [lastProcessedRecord, setLastProcessedRecord] = useState<any | null>(null);

  const [data, setData] = useState<PortfolioMonitorData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadMonths();
  }, []);

  useEffect(() => {
    if (selectedCutoff !== undefined) {
      loadMonitor();
    }
  }, [selectedCutoff, rangeStart, compareMode, compareCutoff, tierFilter, segmentFilter, warningFilter]);

  // Periodic simulation poll when running
  useEffect(() => {
    let interval: any = null;
    if (showSimulation && simRunning) {
      const intervalMs = simSpeed === '10x' ? 1000 : simSpeed === '5x' ? 2500 : 5000;
      loadSimulationStream();
      interval = setInterval(() => {
        loadSimulationStream();
      }, intervalMs);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [showSimulation, simRunning, simSpeed, driftScenario]);

  const loadMonths = async () => {
    try {
      const ms = await api.getCutoffMonths();
      setMonths(ms);
      if (ms.length > 0) {
        const latest = ms[ms.length - 1].cutoff_mi;
        setSelectedCutoff(latest);
        if (ms.length >= 6) {
          setCompareCutoff(ms[ms.length - 7].cutoff_mi); // default 6-month prior for comparison
        }
      }
    } catch (err) {
      console.error(err);
    }
  };

  const loadMonitor = async () => {
    if (!selectedCutoff) return;
    try {
      setLoading(true);
      const res = await api.getPortfolioMonitor({
        cutoff_mi: selectedCutoff,
        range_start_mi: rangePreset !== '1m' ? rangeStart : undefined,
        compare_cutoff_mi: compareMode ? compareCutoff : undefined,
        tier_filter: tierFilter || undefined,
        segment_filter: segmentFilter || undefined,
        warning_filter: warningFilter || undefined,
      });
      setData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadSimulationStream = async () => {
    try {
      const res = await api.getSimulationStream();
      // Adjust simulation numbers dynamically based on scenario
      if (driftScenario === 'warning') {
        res.model_monitoring.performance_drift = 'Moderate';
        res.model_monitoring.calibration_drift = 'Warning';
        res.model_monitoring.current_pr_auc = 0.742;
        res.model_monitoring.status_text = 'Minor performance degradation detected — ongoing surveillance active';
      } else if (driftScenario === 'critical') {
        res.model_monitoring.performance_drift = 'Critical Drift';
        res.model_monitoring.calibration_drift = 'Severe Mismatch';
        res.model_monitoring.current_pr_auc = 0.688;
        res.model_monitoring.retraining_recommended = true;
        res.model_monitoring.status_text = '⚡ Behavioural Drift Exceeded Threshold — Retraining Recommended';
      }
      setSimData(res);
      const records = await api.getSimRecords(8);
      setSimPersistedRecords(records);
    } catch (err) {
      console.error(err);
    }
  };

  const handleProcessNextRecord = async () => {
    try {
      setSimProcessing(true);
      const res = await api.processNextSimRecord();
      setLastProcessedRecord(res.record);
      await loadSimulationStream();
    } catch (err) {
      console.error(err);
    } finally {
      setSimProcessing(false);
    }
  };

  const handleResetSimRecords = async () => {
    try {
      await api.resetSimRecords();
      setSimPersistedRecords([]);
      setLastProcessedRecord(null);
    } catch (err) {
      console.error(err);
    }
  };

  const handleResolveFlag = async (loanId: number) => {
    try {
      await api.resolveLoanFlag(loanId, 'Examiner verified risk profile within normal operating bounds.');
      await loadMonitor();
    } catch (err) {
      console.error(err);
    }
  };

  const handlePresetChange = (preset: '1m' | '3m' | '6m' | '12m' | 'custom') => {
    setRangePreset(preset);
    if (!selectedCutoff) return;

    if (preset === '1m') {
      setRangeStart(undefined);
    } else if (preset === '3m') {
      setRangeStart(selectedCutoff - 2);
    } else if (preset === '6m') {
      setRangeStart(selectedCutoff - 5);
    } else if (preset === '12m') {
      setRangeStart(selectedCutoff - 11);
    }
  };

  const handleInspectLoan = (loanId: number) => {
    navigate(`/loan-detail?loan_id=${loanId}&cutoff_mi=${data?.current_cutoff_mi}`);
  };

  const clearAllFilters = () => {
    setTierFilter(null);
    setSegmentFilter(null);
    setWarningFilter(null);
    setAgreementFilter(null);
  };

  const hasActiveFilters = tierFilter !== null || segmentFilter !== null || warningFilter !== null || agreementFilter !== null;

  return (
    <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Title & Multi-Month Range / Cutoff Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-2 border-b border-sandBorder">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-espresso tracking-tight">
              Portfolio Monitor
            </h1>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-outline text-[11px] font-mono font-bold">
              {data?.is_range
                ? `${data.range_start_date} → ${data.current_cutoff_date} (${rangePreset.toUpperCase()})`
                : `Cutoff: ${data?.current_cutoff_date || '—'}`}
            </span>
          </div>
          <p className="text-xs text-outline mt-0.5">
            Real-time decision support for human loan officers • Ranked review queue on calibrated 6-month risk (Head A)
          </p>
        </div>

        {/* Dynamic Period & Comparison Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Preset Buttons */}
          <div className="flex items-center bg-surface-container-lowest p-1 rounded-full border border-sandBorder shadow-xs text-xs font-bold text-espresso">
            {(['1m', '3m', '6m', '12m', 'custom'] as const).map((p) => (
              <button
                key={p}
                onClick={() => handlePresetChange(p)}
                className={`px-3 py-1 rounded-full transition-all text-xs ${
                  rangePreset === p
                    ? 'bg-espresso text-white shadow-xs'
                    : 'text-outline hover:text-espresso hover:bg-surface-container'
                }`}
              >
                {p === '1m' ? '1 Month' : p === '3m' ? '3 Months' : p === '6m' ? '6 Months' : p === '12m' ? '12 Months' : 'Custom'}
              </button>
            ))}
          </div>

          {/* Month Dropdown / Range Selectors */}
          <div className="flex items-center gap-2 bg-surface-container-lowest px-3 py-1.5 rounded-full border border-sandBorder shadow-xs">
            <Calendar className="w-4 h-4 text-cognac" />
            {rangePreset === 'custom' && (
              <>
                <span className="text-[11px] font-bold text-outline">From:</span>
                <select
                  value={rangeStart || months[0]?.cutoff_mi}
                  onChange={(e) => setRangeStart(parseInt(e.target.value, 10))}
                  className="text-xs font-bold text-espresso bg-transparent border-none focus:ring-0 cursor-pointer"
                >
                  {months.map((m) => (
                    <option key={m.cutoff_mi} value={m.cutoff_mi}>
                      {m.label}
                    </option>
                  ))}
                </select>
                <span className="text-[11px] font-bold text-outline">To:</span>
              </>
            )}
            <span className="text-xs font-bold text-espresso">Month-End:</span>
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

          {/* Two-Period Comparison Toggle */}
          <button
            onClick={() => setCompareMode(!compareMode)}
            className={`px-3 py-1.5 rounded-full text-xs font-bold border transition-all flex items-center gap-1.5 ${
              compareMode
                ? 'bg-cognac text-white border-cognac shadow-xs'
                : 'bg-surface-container-lowest text-espresso border-sandBorder hover:bg-surface-container'
            }`}
          >
            <ArrowUpDown className="w-3.5 h-3.5" />
            <span>Compare Periods</span>
          </button>

          {/* Live Data Simulation Toggle Button */}
          <button
            onClick={() => {
              setShowSimulation(!showSimulation);
              if (!showSimulation && !simData) loadSimulationStream();
            }}
            className={`px-3 py-1.5 rounded-full text-xs font-bold border transition-all flex items-center gap-1.5 ${
              showSimulation
                ? 'bg-emerald-800 text-white border-emerald-800 shadow-xs'
                : 'bg-surface-container-lowest text-tertiary border-emerald-300 hover:bg-emerald-50'
            }`}
          >
            <Radio className={`w-3.5 h-3.5 ${simRunning ? 'animate-pulse text-emerald-400' : ''}`} />
            <span>Live Stream Demo</span>
          </button>
        </div>
      </div>

      {/* Two-Period Comparison Drawer / Card (Requirement 3) */}
      {compareMode && (
        <div className="bg-[#FFFDF9] rounded-3xl p-5 border border-amber-300 shadow-warm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-sandBorder/60">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full bg-amber-200 text-cognac font-bold text-xs uppercase">
                Two-Period Comparative Analysis
              </span>
              <span className="text-xs text-outline font-medium">
                Comparing Period A vs Period B to evaluate portfolio drift & risk migration
              </span>
            </div>

            <div className="flex items-center gap-2 text-xs font-bold text-espresso">
              <span>Compare against Period B:</span>
              <select
                value={compareCutoff}
                onChange={(e) => setCompareCutoff(parseInt(e.target.value, 10))}
                className="text-xs font-bold text-espresso bg-surface-container-lowest px-2 py-1 rounded-lg border border-sandBorder"
              >
                {months.map((m) => (
                  <option key={m.cutoff_mi} value={m.cutoff_mi}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {data?.comparison ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-3.5 rounded-2xl bg-white border border-sandBorder">
                <span className="text-[11px] font-bold text-outline uppercase block">At-Risk Loans</span>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-xl font-extrabold text-espresso">{data.comparison.at_risk_loans_a}</span>
                  <span className="text-xs font-bold text-outline">vs {data.comparison.at_risk_loans_b}</span>
                </div>
                <div className="text-[11px] mt-1 pt-1 border-t border-sandBorder/30">
                  <span className={data.comparison.at_risk_loans_diff >= 0 ? 'text-error font-bold' : 'text-tertiary font-bold'}>
                    {data.comparison.at_risk_loans_diff >= 0 ? `+${data.comparison.at_risk_loans_diff}` : data.comparison.at_risk_loans_diff} net facilities
                  </span>
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-white border border-sandBorder">
                <span className="text-[11px] font-bold text-outline uppercase block">Mean 6M Calibrated Risk</span>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-xl font-extrabold text-cognac">{data.comparison.mean_risk_a}%</span>
                  <span className="text-xs font-bold text-outline">vs {data.comparison.mean_risk_b}%</span>
                </div>
                <div className="text-[11px] mt-1 pt-1 border-t border-sandBorder/30">
                  <span className={data.comparison.mean_risk_diff_pp >= 0 ? 'text-error font-bold' : 'text-tertiary font-bold'}>
                    {data.comparison.mean_risk_diff_pp >= 0 ? `+${data.comparison.mean_risk_diff_pp}` : data.comparison.mean_risk_diff_pp} pp shift
                  </span>
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-white border border-sandBorder">
                <span className="text-[11px] font-bold text-outline uppercase block">Above Alert Threshold</span>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-xl font-extrabold text-error">{data.comparison.above_threshold_a}</span>
                  <span className="text-xs font-bold text-outline">vs {data.comparison.above_threshold_b}</span>
                </div>
                <div className="text-[11px] mt-1 pt-1 border-t border-sandBorder/30">
                  <span className={data.comparison.above_threshold_diff >= 0 ? 'text-error font-bold' : 'text-tertiary font-bold'}>
                    {data.comparison.above_threshold_diff >= 0 ? `+${data.comparison.above_threshold_diff}` : data.comparison.above_threshold_diff} facilities
                  </span>
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-white border border-sandBorder">
                <span className="text-[11px] font-bold text-outline uppercase block">High-Risk Facilities (&ge;5%)</span>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-xl font-extrabold text-espresso">{data.comparison.high_risk_a}</span>
                  <span className="text-xs font-bold text-outline">vs {data.comparison.high_risk_b}</span>
                </div>
                <div className="text-[11px] mt-1 pt-1 border-t border-sandBorder/30">
                  <span className={data.comparison.high_risk_diff >= 0 ? 'text-error font-bold' : 'text-tertiary font-bold'}>
                    {data.comparison.high_risk_diff >= 0 ? `+${data.comparison.high_risk_diff}` : data.comparison.high_risk_diff} facilities
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-outline">Select two distinct months to compute comparative metrics.</p>
          )}

          {/* Why Risk Changed - Two-Period Diagnostic Breakdown */}
          {data?.why_risk_changed && data.why_risk_changed.length > 0 && (
            <div className="mt-3 pt-3 border-t border-amber-200/80">
              <div className="flex items-center gap-2 mb-2.5">
                <Info className="w-4 h-4 text-cognac" />
                <span className="text-xs font-bold text-espresso uppercase tracking-wider">
                  Why Did Risk Change? — Portfolio Period Diagnostic Breakdown
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {data.why_risk_changed.map((factor, idx) => (
                  <div key={idx} className="p-3 rounded-2xl bg-white border border-sandBorder text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-espresso">{factor.factor}</span>
                      <span className={`font-mono font-bold text-[11px] px-1.5 py-0.5 rounded ${
                        factor.impact === 'Risk Increasing'
                          ? 'bg-red-50 text-error'
                          : factor.impact === 'Protective'
                          ? 'bg-emerald-50 text-emerald-800'
                          : 'bg-stone-100 text-stone-700'
                      }`}>
                        {factor.delta}
                      </span>
                    </div>
                    <p className="text-[11px] text-stone-600 leading-relaxed">{factor.explanation}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Simulate New Bank Record (Simple, 10-Second Live Demonstration) */}
      {showSimulation && (
        <div className="bg-[#F8FAF9] rounded-3xl p-5 border border-emerald-300 shadow-warm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-emerald-200">
            <div>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <h3 className="font-bold text-sm text-espresso uppercase tracking-tight">
                  Test New Bank Record
                </h3>
                <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 font-mono text-[10px] font-bold">
                  LIVE DEMONSTRATION
                </span>
              </div>
              <p className="text-xs text-outline mt-0.5">
                Simulates an incoming account record: stored, evaluated over 12-month history by M6 Transformer, and updating calibrated risk.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleProcessNextRecord}
                disabled={simProcessing}
                className="px-4 py-2 rounded-full text-xs font-bold flex items-center gap-2 bg-espresso text-white hover:bg-black shadow-xs disabled:opacity-50 transition-all cursor-pointer"
                title="Process next incoming monthly record through the trained M6 Transformer"
              >
                {simProcessing ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <StepForward className="w-3.5 h-3.5 text-amber-400" />
                )}
                <span>Process New Record</span>
              </button>

              <button
                onClick={handleResetSimRecords}
                className="p-2 rounded-full bg-white border border-sandBorder text-stone-600 hover:text-error transition-colors"
                title="Reset simulation records"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* 4-Step Pipeline Sequence */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="p-3 rounded-2xl bg-white border border-sandBorder/70 flex items-center justify-between">
              <span className="text-stone-700 font-medium">1. Bank record received</span>
              <Check className="w-4 h-4 text-emerald-600" />
            </div>
            <div className="p-3 rounded-2xl bg-white border border-sandBorder/70 flex items-center justify-between">
              <span className="text-stone-700 font-medium">2. Record stored</span>
              <Check className="w-4 h-4 text-emerald-600" />
            </div>
            <div className="p-3 rounded-2xl bg-white border border-sandBorder/70 flex items-center justify-between">
              <span className="text-stone-700 font-medium">3. Model evaluates 12M history</span>
              <Check className="w-4 h-4 text-emerald-600" />
            </div>
            <div className="p-3 rounded-2xl bg-white border border-sandBorder/70 flex items-center justify-between">
              <span className="text-stone-700 font-medium">4. Risk updated</span>
              <Check className="w-4 h-4 text-emerald-600" />
            </div>
          </div>

          {/* Result Card: Previous vs New Risk & Reasons */}
          <div className="p-4 rounded-2xl bg-white border border-emerald-300 space-y-3 text-xs">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold text-espresso text-sm">
                    Loan #{lastProcessedRecord?.loan_id || 5316}
                  </span>
                  <span className="text-stone-700 font-semibold">
                    ({lastProcessedRecord?.borrower_name || 'Apex Logistics Corp'})
                  </span>
                  <span className="text-[11px] text-outline">
                    · {lastProcessedRecord?.event_type || 'Large Debit Disbursement (-18,500 CZK)'}
                  </span>
                </div>
                <p className="text-[11px] text-stone-600">
                  Operating Balance: <strong>{(lastProcessedRecord?.balance || -4200).toLocaleString()} CZK</strong> · Evaluated by M6 Multi-Task Transformer.
                </p>
              </div>

              <div className="flex items-center gap-4 border-t md:border-t-0 md:border-l border-sandBorder/60 pt-2 md:pt-0 md:pl-4">
                <div>
                  <span className="text-[10px] text-outline uppercase font-bold block">Previous Risk</span>
                  <span className="font-mono text-stone-600 text-sm font-semibold">
                    {((lastProcessedRecord?.previous_risk || 0.021) * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="text-stone-400 font-bold">→</div>
                <div>
                  <span className="text-[10px] text-outline uppercase font-bold block">New Model Risk</span>
                  <span className={`font-mono text-sm font-extrabold ${
                    (lastProcessedRecord?.predicted_risk || 0.048) >= 0.0285 ? 'text-error' : 'text-emerald-700'
                  }`}>
                    {((lastProcessedRecord?.predicted_risk || 0.048) * 100).toFixed(2)}%
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-outline uppercase font-bold block">Status</span>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                    (lastProcessedRecord?.predicted_risk || 0.048) >= 0.0285
                      ? 'bg-red-100 text-error'
                      : 'bg-emerald-100 text-emerald-800'
                  }`}>
                    {(lastProcessedRecord?.predicted_risk || 0.048) >= 0.0285 ? 'MODEL ALERT' : 'NORMAL'}
                  </span>
                </div>
              </div>
            </div>

            {/* Risk Changed Because: */}
            <div className="pt-2 border-t border-sandBorder/40">
              <span className="text-[11px] font-bold text-espresso block mb-1">
                Risk changed because:
              </span>
              <ul className="list-disc list-inside space-y-0.5 text-[11px] text-stone-700">
                {(lastProcessedRecord?.risk_reasons || [
                  'Balance decreased into overdraft (-4,200 CZK)',
                  'Cash-flow deficit deepened from large disbursement',
                  '3 overdraft days recorded in current cycle'
                ]).map((reason: string, rIdx: number) => (
                  <li key={rIdx}>{reason}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {loading && !data ? (
        <div className="py-16 text-center text-xs text-outline">
          <span className="w-8 h-8 border-2 border-cognac border-t-transparent rounded-full animate-spin inline-block mb-3"></span>
          <p>Loading monthly portfolio telemetry...</p>
        </div>
      ) : data ? (
        <div className="space-y-6">
          {/* 4 Primary Portfolio Cards (Requirement 17) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
              <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
                Total Loans
              </span>
              <span className="text-3xl font-extrabold text-espresso tracking-tight">
                {data.total_at_risk_loans}
              </span>
              <span className="text-[11px] text-outline block mt-2 pt-2 border-t border-sandBorder/40">
                Active facilities in month {data.current_cutoff_date}
              </span>
            </div>

            <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
              <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
                High Risk (Alert)
              </span>
              <span className="text-3xl font-extrabold text-error tracking-tight">
                {data.above_alert_threshold}
              </span>
              <span className="text-[11px] text-error font-medium block mt-2 pt-2 border-t border-sandBorder/40">
                Calibrated risk &ge; 2.85% alert threshold
              </span>
            </div>

            <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
              <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
                Watchlist
              </span>
              <span className="text-3xl font-extrabold text-cognac tracking-tight">
                {data.queue.filter((q) => q.risk_percent >= 2.0 && q.risk_percent < 4.0).length}
              </span>
              <span className="text-[11px] text-outline block mt-2 pt-2 border-t border-sandBorder/40">
                Emerging warning tier (2.0% &ndash; 4.0%)
              </span>
            </div>

            <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
              <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-1">
                Human Flagged
              </span>
              <span className="text-3xl font-extrabold text-purple-800 tracking-tight">
                {data.queue.filter((q) => q.human_flag?.is_flagged).length}
              </span>
              <span className="text-[11px] text-purple-700 font-medium block mt-2 pt-2 border-t border-sandBorder/40">
                Examiner qualitative oversight flags
              </span>
            </div>
          </div>

          {/* Month-on-Month Alert Badge */}
          <div className="p-3 rounded-2xl bg-[#FFF9EE] border border-amber-200 text-xs text-espresso flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded-full bg-amber-200 text-cognac font-bold text-[10px] uppercase">
                Month-on-Month Shift
              </span>
              <span>
                <strong className="font-bold text-espresso">{data.mom_entered_top_10}</strong> loans entered the top 10% queue,{' '}
                <strong className="font-bold text-espresso">{data.mom_left_top_10}</strong> left it.
              </span>
            </div>
            <span className="text-[11px] text-outline font-mono">Cutoff: {data.current_cutoff_date}</span>
          </div>

          {/* Human-Model Agreement Quadrant Matrix */}
          <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <div className="flex items-center gap-2">
                  <UserCheck className="w-4 h-4 text-cognac" />
                  <h2 className="text-sm font-bold text-espresso uppercase tracking-wider">
                    Human–Model Alignment Matrix
                  </h2>
                </div>
                <p className="text-xs text-outline mt-0.5">
                  Explicit separation of Transformer neural distress assessment (Head A threshold 2.85%) and human examiner judgment.
                </p>
              </div>
              {agreementFilter && (
                <button
                  onClick={() => setAgreementFilter(null)}
                  className="text-xs font-bold text-cognac hover:underline self-start sm:self-auto"
                >
                  Clear Quadrant Filter ({agreementFilter}) ✕
                </button>
              )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Confirmed Concern */}
              <div
                onClick={() => setAgreementFilter(agreementFilter === 'Confirmed Concern' ? null : 'Confirmed Concern')}
                className={`p-4 rounded-2xl border transition-all cursor-pointer ${
                  agreementFilter === 'Confirmed Concern'
                    ? 'bg-red-50 border-red-400 shadow-xs'
                    : 'bg-white border-sandBorder hover:border-red-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-error">Confirmed Concern</span>
                  <span className="w-2.5 h-2.5 rounded-full bg-red-600"></span>
                </div>
                <div className="flex items-baseline justify-between mt-2">
                  <span className="text-3xl font-extrabold text-error">
                    {data.agreement_summary?.confirmed_concern ?? 0}
                  </span>
                  <span className="text-[10px] font-bold text-outline">facilities</span>
                </div>
                <p className="text-[11px] text-stone-600 mt-2 pt-2 border-t border-sandBorder/40">
                  Both Transformer model and analyst flagged severe distress. Immediate supervisory workout required.
                </p>
              </div>

              {/* Model Alert */}
              <div
                onClick={() => setAgreementFilter(agreementFilter === 'Model Alert' ? null : 'Model Alert')}
                className={`p-4 rounded-2xl border transition-all cursor-pointer ${
                  agreementFilter === 'Model Alert'
                    ? 'bg-amber-50 border-amber-400 shadow-xs'
                    : 'bg-white border-sandBorder hover:border-amber-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-cognac">Model Alert</span>
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                </div>
                <div className="flex items-baseline justify-between mt-2">
                  <span className="text-3xl font-extrabold text-cognac">
                    {data.agreement_summary?.model_alert ?? 0}
                  </span>
                  <span className="text-[10px] font-bold text-outline">facilities</span>
                </div>
                <p className="text-[11px] text-stone-600 mt-2 pt-2 border-t border-sandBorder/40">
                  Transformer flagged risk &ge; 2.85%, human examiner has not yet flagged. In analyst review queue.
                </p>
              </div>

              {/* Human Override */}
              <div
                onClick={() => setAgreementFilter(agreementFilter === 'Human Override' ? null : 'Human Override')}
                className={`p-4 rounded-2xl border transition-all cursor-pointer ${
                  agreementFilter === 'Human Override'
                    ? 'bg-purple-50 border-purple-400 shadow-xs'
                    : 'bg-white border-sandBorder hover:border-purple-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-purple-800">Human Override</span>
                  <span className="w-2.5 h-2.5 rounded-full bg-purple-600"></span>
                </div>
                <div className="flex items-baseline justify-between mt-2">
                  <span className="text-3xl font-extrabold text-purple-900">
                    {data.agreement_summary?.human_override ?? 0}
                  </span>
                  <span className="text-[10px] font-bold text-outline">facilities</span>
                </div>
                <p className="text-[11px] text-stone-600 mt-2 pt-2 border-t border-sandBorder/40">
                  Model assessed low risk, but examiner flagged based on qualitative, sector, or off-balance indicators.
                </p>
              </div>

              {/* Concordant Normal */}
              <div
                onClick={() => setAgreementFilter(agreementFilter === 'Normal' ? null : 'Normal')}
                className={`p-4 rounded-2xl border transition-all cursor-pointer ${
                  agreementFilter === 'Normal'
                    ? 'bg-emerald-50 border-emerald-400 shadow-xs'
                    : 'bg-white border-sandBorder hover:border-emerald-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-800">Concordant Normal</span>
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-600"></span>
                </div>
                <div className="flex items-baseline justify-between mt-2">
                  <span className="text-3xl font-extrabold text-emerald-900">
                    {data.agreement_summary?.normal ?? 0}
                  </span>
                  <span className="text-[10px] font-bold text-outline">facilities</span>
                </div>
                <p className="text-[11px] text-stone-600 mt-2 pt-2 border-t border-sandBorder/40">
                  Both Transformer model and examiner agree facility is performing within standard operating tolerances.
                </p>
              </div>
            </div>
          </div>

          {/* Dedicated Human Review Queue (Active Examiner Flags) */}
          {data.human_review_queue && data.human_review_queue.length > 0 && (
            <div className="bg-surface-container-lowest rounded-3xl p-6 border border-amber-300/80 shadow-warm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2">
                    <Flag className="w-4 h-4 text-cognac" />
                    <h2 className="text-sm font-bold text-espresso uppercase tracking-wider">
                      Human Review Queue · Active Examiner Flags ({data.human_review_queue.length} Facilities)
                    </h2>
                  </div>
                  <p className="text-xs text-outline mt-0.5">
                    Supervisory queue tracking facilities manually flagged by risk examiners for workout restructuring, collateral audit, or cash-flow surveillance.
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto rounded-2xl border border-sandBorder">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="bg-surface-container border-b border-sandBorder text-outline uppercase tracking-wider text-[10px] font-bold">
                      <th className="py-2.5 px-3">Facility</th>
                      <th className="py-2.5 px-3">Sector</th>
                      <th className="py-2.5 px-3">Severity</th>
                      <th className="py-2.5 px-3">Examiner Reason & Note</th>
                      <th className="py-2.5 px-3">Flagged By</th>
                      <th className="py-2.5 px-3">Model 6M Risk</th>
                      <th className="py-2.5 px-3">Agreement</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-sandBorder bg-white">
                    {data.human_review_queue.map((hq) => (
                      <tr key={hq.loan_id} className="hover:bg-[#FFF9EE] transition-colors">
                        <td className="py-2.5 px-3 font-mono font-bold text-espresso">
                          #{hq.loan_id}
                        </td>
                        <td className="py-2.5 px-3 text-stone-700">
                          {hq.sector || 'Commercial'}
                        </td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            hq.severity === 'Critical'
                              ? 'bg-red-100 text-error'
                              : hq.severity === 'Concern'
                              ? 'bg-amber-100 text-cognac'
                              : 'bg-stone-100 text-stone-700'
                          }`}>
                            {hq.severity}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-stone-800 max-w-sm">
                          <strong className="block text-espresso text-[11px]">{hq.reason}</strong>
                          {hq.note && <span className="text-[10px] text-outline italic block truncate">{hq.note}</span>}
                        </td>
                        <td className="py-2.5 px-3 text-[11px] text-stone-600">
                          {hq.flagged_by || 'Dr. Alex Mercer'}
                        </td>
                        <td className="py-2.5 px-3 font-mono font-bold text-espresso">
                          {hq.risk_percent}%
                        </td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            hq.agreement === 'Confirmed Concern'
                              ? 'bg-red-50 text-error border border-red-200'
                              : hq.agreement === 'Human Override'
                              ? 'bg-purple-50 text-purple-800 border border-purple-200'
                              : 'bg-stone-100 text-stone-700'
                          }`}>
                            {hq.agreement}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right space-x-2">
                          <button
                            onClick={() => handleInspectLoan(hq.loan_id)}
                            className="px-2.5 py-1 rounded-full bg-surface-container hover:bg-espresso hover:text-white text-espresso text-[11px] font-bold transition-all"
                          >
                            Inspect
                          </button>
                          <button
                            onClick={() => handleResolveFlag(hq.loan_id)}
                            className="px-2.5 py-1 rounded-full bg-emerald-50 hover:bg-emerald-600 hover:text-white text-emerald-800 text-[11px] font-bold border border-emerald-200 transition-all"
                            title="Resolve human review flag"
                          >
                            Resolve
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Early Warning Signals & Portfolio Segmentation Row (Requirements 16, 17, 18) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Early Warning Signals Card */}
            <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-bold text-espresso uppercase tracking-wider">
                    Early Warning Signals
                  </h2>
                  <p className="text-xs text-outline mt-0.5">
                    Click any signal to isolate affected facilities in the review queue below.
                  </p>
                </div>
                {warningFilter && (
                  <button
                    onClick={() => setWarningFilter(null)}
                    className="text-[11px] font-bold text-cognac hover:underline"
                  >
                    Clear Filter
                  </button>
                )}
              </div>

              <div className="space-y-2 pt-1">
                {data.early_warnings?.map((sig) => {
                  const isSelected = warningFilter === sig.id;
                  return (
                    <div
                      key={sig.id}
                      onClick={() => setWarningFilter(isSelected ? null : sig.id)}
                      className={`p-3 rounded-2xl border transition-all cursor-pointer flex items-center justify-between ${
                        isSelected
                          ? 'bg-[#FFF9EE] border-cognac shadow-xs'
                          : 'bg-white border-sandBorder hover:border-sandBorder/80'
                      }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <span
                          className={`w-2.5 h-2.5 rounded-full ${
                            sig.severity === 'High'
                              ? 'bg-red-500'
                              : sig.severity === 'Medium'
                              ? 'bg-amber-500'
                              : 'bg-emerald-500'
                          }`}
                        ></span>
                        <div>
                          <span className="font-bold text-xs text-espresso block">{sig.signal}</span>
                          <span className="text-[11px] text-outline block">{sig.description}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-espresso text-xs font-mono font-bold">
                          {sig.facilities} facilities
                        </span>
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            sig.severity === 'High'
                              ? 'bg-red-50 text-error'
                              : sig.severity === 'Medium'
                              ? 'bg-amber-50 text-cognac'
                              : 'bg-emerald-50 text-tertiary'
                          }`}
                        >
                          {sig.severity}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Portfolio Segment Risk Heatmap Summary (Requirement 18) */}
            <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-bold text-espresso uppercase tracking-wider">
                    Segment Distress Heatmap
                  </h2>
                  <p className="text-xs text-outline mt-0.5">
                    Cross-segment facility count distribution across calibrated distress tiers.
                  </p>
                </div>
                {segmentFilter && (
                  <button
                    onClick={() => setSegmentFilter(null)}
                    className="text-[11px] font-bold text-cognac hover:underline"
                  >
                    Clear Filter
                  </button>
                )}
              </div>

              <div className="overflow-x-auto rounded-2xl border border-sandBorder pt-1">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-surface-container border-b border-sandBorder text-outline uppercase tracking-wider text-[10px] font-bold">
                      <th className="py-2.5 px-3">Segment</th>
                      <th className="py-2.5 px-2 text-center text-emerald-800">Low (&lt;2%)</th>
                      <th className="py-2.5 px-2 text-center text-stone-700">Mod (2-4%)</th>
                      <th className="py-2.5 px-2 text-center text-cognac">High (4-7%)</th>
                      <th className="py-2.5 px-2 text-center text-error">Crit (&ge;7%)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-sandBorder bg-white">
                    {data.segment_summary?.map((seg) => {
                      const isSelected = segmentFilter === seg.segment;
                      return (
                        <tr
                          key={seg.segment}
                          onClick={() => setSegmentFilter(isSelected ? null : seg.segment)}
                          className={`cursor-pointer transition-colors ${
                            isSelected ? 'bg-amber-50 font-bold' : 'hover:bg-[#FFF9EE]'
                          }`}
                        >
                          <td className="py-2.5 px-3 font-semibold text-espresso">{seg.segment}</td>
                          <td className="py-2.5 px-2 text-center font-mono text-emerald-700">{seg.low}</td>
                          <td className="py-2.5 px-2 text-center font-mono text-stone-700">{seg.moderate}</td>
                          <td className="py-2.5 px-2 text-center font-mono font-bold text-cognac">{seg.high}</td>
                          <td className="py-2.5 px-2 text-center font-mono font-bold text-error">
                            {seg.critical > 0 ? `⚡ ${seg.critical}` : '0'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Headline Evaluation (P1 out-of-time test) */}
          <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-espresso uppercase tracking-wider">
                Headline Evaluation (P1 Out-of-Time Test)
              </h2>
              <span className="text-[11px] font-mono text-outline">Model: {data.headline.model_name}</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1">
              <div className="p-3 rounded-2xl bg-surface-container-low">
                <span className="text-[11px] text-outline font-bold block">PR-AUC</span>
                <span className="text-2xl font-extrabold text-espresso mt-0.5 block">{data.headline.pr_auc}</span>
                <span className="text-[10px] text-outline block mt-1">95% CI {data.headline.pr_auc_ci}</span>
              </div>

              <div className="p-3 rounded-2xl bg-surface-container-low">
                <span className="text-[11px] text-outline font-bold block">Capture @ Top 10%</span>
                <span className="text-2xl font-extrabold text-cognac mt-0.5 block">{data.headline.capture_at_10}%</span>
                <span className="text-[10px] text-outline block mt-1">Of all defaulting event facilities</span>
              </div>

              <div className="p-3 rounded-2xl bg-surface-container-low">
                <span className="text-[11px] text-outline font-bold block">Brier Score</span>
                <span className="text-2xl font-extrabold text-espresso mt-0.5 block">{data.headline.brier_score}</span>
                <span className="text-[10px] text-outline block mt-1">Calibrated squared probability error</span>
              </div>
            </div>

            <p className="text-[11px] text-outline pt-2 border-t border-sandBorder/40">
              Test set: {data.headline.event_loans_test} event loans. Intervals are 95% loan-clustered bootstrap. Berka is 1993–1998 Czech banking data.
            </p>
          </div>

          {/* Review Queue Table with Segment Filters */}
          <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-bold text-espresso tracking-tight">
                  Review Queue · {data.current_cutoff_date}
                </h2>
                <p className="text-xs text-outline mt-0.5">
                  Showing facilities ranked by calibrated 6-month distress probability. Click any loan to open full loan detail.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {/* Active Filter Pills */}
                {hasActiveFilters && (
                  <button
                    onClick={clearAllFilters}
                    className="px-2.5 py-1 rounded-full bg-stone-200 text-espresso text-[11px] font-bold hover:bg-stone-300"
                  >
                    Reset Filters ✕
                  </button>
                )}

                {/* Risk Tier Quick Filter */}
                <select
                  value={tierFilter || ''}
                  onChange={(e) => setTierFilter(e.target.value || null)}
                  className="px-3 py-1 rounded-full bg-surface-container text-xs font-bold text-espresso border-none cursor-pointer"
                >
                  <option value="">All Risk Tiers</option>
                  <option value="Critical">Critical (&ge;7%)</option>
                  <option value="High">High (4-7%)</option>
                  <option value="Moderate">Moderate (2-4%)</option>
                  <option value="Low">Low (&lt;2%)</option>
                </select>

                <span className="px-3 py-1 rounded-full bg-surface-container text-espresso text-xs font-bold">
                  {((data?.queue || []).filter(r => !agreementFilter || r.agreement === agreementFilter)).length} Facilities In Queue
                </span>
              </div>
            </div>

            <div className="overflow-x-auto rounded-2xl border border-sandBorder">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-surface-container border-b border-sandBorder text-outline uppercase tracking-wider text-[11px] font-bold">
                    <th className="py-3 px-4">Rank</th>
                    <th className="py-3 px-4">Loan ID</th>
                    <th className="py-3 px-4">Sector</th>
                    <th className="py-3 px-4">6-Month Risk</th>
                    <th className="py-3 px-4">Δ vs Last Month</th>
                    <th className="py-3 px-4">Percentile</th>
                    <th className="py-3 px-4">Agreement</th>
                    <th className="py-3 px-4">Review State</th>
                    <th className="py-3 px-4">Top Driver</th>
                    <th className="py-3 px-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-sandBorder bg-white">
                  {(data.queue || [])
                    .filter((r) => !agreementFilter || r.agreement === agreementFilter)
                    .map((row) => (
                    <tr
                      key={row.loan_id}
                      onClick={() => handleInspectLoan(row.loan_id)}
                      className="hover:bg-[#FFF9EE] cursor-pointer transition-colors group"
                    >
                      <td className="py-3 px-4 font-mono font-bold text-espresso">
                        #{row.queue_rank}
                      </td>
                      <td className="py-3 px-4 font-mono font-bold text-espresso">
                        <div className="flex items-center gap-1.5">
                          <span>#{row.loan_id}</span>
                          {row.human_flag?.is_flagged && (
                            <span title={`Human Flag: ${row.human_flag.severity} - ${row.human_flag.reason}`}>
                              <Flag className="w-3.5 h-3.5 text-error fill-error inline" />
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-4 text-stone-700">
                        <span className="px-2 py-0.5 rounded-full bg-surface-container text-outline text-[10px] font-semibold">
                          {row.sector || 'Commercial'}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-2 rounded-full bg-sandBorder/40 overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                row.risk_percent > 7 ? 'bg-red-600' : row.risk_percent > 3 ? 'bg-amber-500' : 'bg-emerald-500'
                              }`}
                              style={{ width: `${Math.min(row.risk_percent * 6, 100)}%` }}
                            ></div>
                          </div>
                          <span className={`font-mono font-bold ${row.above_alert ? 'text-error' : 'text-espresso'}`}>
                            {row.risk_percent}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4 font-mono">
                        {row.delta_vs_last_month !== null ? (
                          <span className={row.delta_vs_last_month > 0 ? 'text-error' : 'text-tertiary'}>
                            {row.delta_vs_last_month > 0 ? `+${(row.delta_vs_last_month * 100).toFixed(2)}%` : `${(row.delta_vs_last_month * 100).toFixed(2)}%`}
                          </span>
                        ) : (
                          <span className="text-outline">—</span>
                        )}
                      </td>
                      <td className="py-3 px-4 font-mono text-stone-700">
                        top {row.queue_percentile}%
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          row.agreement === 'Confirmed Concern'
                            ? 'bg-red-100 text-error border border-red-200'
                            : row.agreement === 'Model Alert'
                            ? 'bg-amber-100 text-cognac'
                            : row.agreement === 'Human Override'
                            ? 'bg-purple-100 text-purple-800 border border-purple-200'
                            : 'bg-emerald-100 text-tertiary'
                        }`}>
                          {row.agreement || 'Normal'}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            row.review_status === 'escalated'
                              ? 'bg-red-100 text-error'
                              : row.review_status === 'watchlist'
                              ? 'bg-amber-100 text-cognac'
                              : row.review_status === 'reviewed'
                              ? 'bg-emerald-100 text-tertiary'
                              : 'bg-stone-100 text-stone-600'
                          }`}
                        >
                          {row.review_status ? row.review_status.toUpperCase() : 'PENDING'}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-[11px] text-stone-800 max-w-xs truncate">
                        {row.top_driver}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleInspectLoan(row.loan_id);
                          }}
                          className="px-3 py-1 rounded-full bg-surface-container group-hover:bg-espresso group-hover:text-white text-espresso text-xs font-bold transition-all"
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
