import React, { useEffect, useState } from 'react';
import { api } from '../api';
import { ModelHealthData, BenchmarkRow } from '../types';
import {
  Cpu,
  Layers,
  Activity,
  CheckCircle2,
  TrendingUp,
  HelpCircle,
  AlertTriangle,
  Award,
  BarChart2,
  GitCommit
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  Cell
} from 'recharts';

export const ModelHealth: React.FC = () => {
  const [healthData, setHealthData] = useState<ModelHealthData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadModelHealth();
  }, []);

  const loadModelHealth = async () => {
    try {
      setLoading(true);
      const data = await api.getModelHealth();
      setHealthData(data);
    } catch (err) {
      console.error('Failed to load model health data:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-[1400px] mx-auto px-4 py-16 text-center text-xs text-outline">
        <span className="w-8 h-8 border-2 border-cognac border-t-transparent rounded-full animate-spin inline-block mb-3"></span>
        <p>Loading model specifications, benchmark zoo, and training curves...</p>
      </div>
    );
  }

  const benchmarks: BenchmarkRow[] = healthData?.benchmarks || [];
  const modelConfig = healthData?.model_configuration || {} as any;
  const trainingHistory = healthData?.training_history || [];
  const calibrationCurve = healthData?.calibration_curve || [];

  // Format 11 benchmark rows for charts
  const chartDataPR = benchmarks.map((b) => ({
    model: b.model,
    pr_auc: b.pr_auc !== null ? Number(b.pr_auc) : 0,
    isM6: b.model === 'M6_multitask'
  }));

  const chartDataROC = benchmarks.map((b) => ({
    model: b.model,
    roc_auc: b.roc_auc !== null ? Number(b.roc_auc) : 0,
    isM6: b.model === 'M6_multitask'
  }));

  const chartDataBrier = benchmarks.map((b) => ({
    model: b.model,
    brier: b.brier !== null ? Number(b.brier) : 0,
    isM6: b.model === 'M6_multitask'
  }));

  const chartDataCapture = benchmarks.map((b) => ({
    model: b.model,
    capture: b['capture@10'] !== null ? Math.round(Number(b['capture@10']) * 100) : 0,
    isM6: b.model === 'M6_multitask'
  }));

  return (
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-sandBorder">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-100 text-tertiary text-[11px] font-bold tracking-wide uppercase">
              Basel III / CECL Surveillance Specification
            </span>
            <span className="text-outline text-xs">• 11-Model Benchmark Zoo</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-espresso tracking-tight">
            Model Health & Benchmark Evaluation
          </h1>
        </div>

        {/* Selected Champion Badge */}
        <div className="p-3 px-5 rounded-2xl bg-surface-container-lowest border border-sandBorder shadow-warm flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-cognac animate-pulse"></div>
          <div>
            <span className="text-[10px] text-outline font-bold uppercase block">Selected Architecture</span>
            <span className="text-base font-extrabold text-espresso font-mono">
              {healthData?.served_model || 'M6_multitask'}
            </span>
          </div>
          <span className="px-2 py-0.5 rounded-full bg-amber-100 text-cognac font-bold text-[10px] uppercase ml-2 flex items-center gap-1">
            <Award className="w-3 h-3" /> PROPOSED CHAMPION
          </span>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 1: 11-MODEL COMPARISON TABLE AT THE TOP */}
      {/* ========================================================================= */}
      <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-cognac" />
              <h2 className="text-lg font-bold text-espresso tracking-tight">
                Section 1: Benchmark Comparison Table (11 Evaluated Architectures)
              </h2>
            </div>
            <p className="text-xs text-outline mt-0.5">
              Evaluated strictly on the held-out P1 Out-of-Time test partition under extreme default imbalance (~4% event rate).
            </p>
          </div>
          <span className="px-3 py-1 rounded-full bg-surface-container text-outline text-xs font-mono font-bold">
            11 Candidate Models
          </span>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-sandBorder">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-surface-container border-b border-sandBorder text-outline uppercase tracking-wider text-[11px] font-bold">
                <th className="py-3 px-4">Model ID & Architecture</th>
                <th className="py-3 px-4">PR-AUC</th>
                <th className="py-3 px-4">ROC-AUC</th>
                <th className="py-3 px-4">Brier Score (Lower=Better)</th>
                <th className="py-3 px-4">Capture @ Top 10%</th>
                <th className="py-3 px-4 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sandBorder font-mono">
              {benchmarks.map((row) => {
                const isChampion = row.model === 'M6_multitask';
                return (
                  <tr
                    key={row.model}
                    className={`transition-colors ${
                      isChampion
                        ? 'bg-amber-50/70 hover:bg-amber-100/60 font-semibold text-espresso'
                        : 'hover:bg-surface-container-low text-espresso'
                    }`}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        {isChampion && (
                          <span className="w-2 h-2 rounded-full bg-cognac"></span>
                        )}
                        <span className={isChampion ? 'font-bold text-cognac' : 'font-medium'}>
                          {row.model}
                        </span>
                        {isChampion && (
                          <span className="px-1.5 py-0.2 rounded bg-cognac text-white text-[9px] uppercase font-sans font-bold">
                            Champion
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      {row.pr_auc !== null ? Number(row.pr_auc).toFixed(4) : 'N/A'}
                    </td>
                    <td className="py-3 px-4">
                      {row.roc_auc !== null ? Number(row.roc_auc).toFixed(4) : 'N/A'}
                    </td>
                    <td className="py-3 px-4">
                      {row.brier !== null ? Number(row.brier).toFixed(4) : 'N/A'}
                    </td>
                    <td className="py-3 px-4">
                      {row['capture@10'] !== null
                        ? `${(Number(row['capture@10']) * 100).toFixed(1)}%`
                        : 'N/A'}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {isChampion ? (
                        <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-tertiary text-[10px] font-sans font-bold">
                          Proposed Champion
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full bg-surface-container text-outline text-[10px] font-sans">
                          Benchmark
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2: WHY M6? — 3 PRESENTATION POINTS */}
      {/* ========================================================================= */}
      <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-sandBorder/60">
          <div className="flex items-center gap-2">
            <HelpCircle className="w-5 h-5 text-cognac" />
            <h2 className="text-lg font-bold text-espresso tracking-tight">
              Section 2: Why M6 Multi-Task Transformer?
            </h2>
          </div>
          <span className="text-xs text-outline font-medium">
            Flow: <strong>Input → Transformer Blocks → Shared Representation → Two Prediction Heads</strong>
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-espresso leading-relaxed">
          {/* Point 1: Why Temporal Modeling */}
          <div className="p-4 rounded-2xl bg-surface-container-low border border-sandBorder/60 space-y-2">
            <span className="font-bold text-cognac uppercase tracking-wider text-[11px] block">
              1. Why Temporal Modeling?
            </span>
            <p className="text-outline">
              The data contains 12 monthly behavioral observations, so the model learns dynamic patterns across time rather than treating each borrower as a static feature vector.
            </p>
          </div>

          {/* Point 2: Why Transformer */}
          <div className="p-4 rounded-2xl bg-surface-container-low border border-sandBorder/60 space-y-2">
            <span className="font-bold text-cognac uppercase tracking-wider text-[11px] block">
              2. Why Transformer?
            </span>
            <p className="text-outline">
              Self-attention allows the model to learn relationships between different months and features, directly capturing critical temporal interactions and early-warning signals.
            </p>
          </div>

          {/* Point 3: Why Multi-Task Learning */}
          <div className="p-4 rounded-2xl bg-surface-container-low border border-sandBorder/60 space-y-2">
            <span className="font-bold text-cognac uppercase tracking-wider text-[11px] block">
              3. Why Multi-Task Learning?
            </span>
            <p className="text-outline">
              M6 jointly predicts <strong>6-month distress risk</strong> and <strong>future monthly hazard</strong>. The shared representation learns both the immediate risk signal and the future trajectory.
            </p>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 3: M6 ARCHITECTURE & TRAINING CONFIGURATION */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Architecture Card */}
        <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-sandBorder/60">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-cognac" />
              <h3 className="text-base font-bold text-espresso tracking-tight">
                Section 3A: M6 Architecture Specification
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-outline text-[11px] font-mono font-bold">
              PyTorch Pre-LN
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-3 rounded-2xl bg-surface-container-low">
              <span className="text-[10px] text-outline font-bold uppercase block">Input Dimension</span>
              <span className="text-base font-extrabold text-espresso mt-0.5 block">
                {modelConfig?.sequence_length || 12} Months × {modelConfig?.input_dim || 44} Feats
              </span>
              <span className="text-[10px] text-outline">Longitudinal behavioral matrix</span>
            </div>

            <div className="p-3 rounded-2xl bg-surface-container-low">
              <span className="text-[10px] text-outline font-bold uppercase block">Encoder Layers</span>
              <span className="text-base font-extrabold text-espresso mt-0.5 block">
                {modelConfig?.n_layers || 3} Pre-LN Blocks
              </span>
              <span className="text-[10px] text-outline">Pre-LayerNorm self-attention</span>
            </div>

            <div className="p-3 rounded-2xl bg-surface-container-low">
              <span className="text-[10px] text-outline font-bold uppercase block">Attention Heads</span>
              <span className="text-base font-extrabold text-espresso mt-0.5 block">
                {modelConfig?.n_heads || 2} Heads (d_k = 16)
              </span>
              <span className="text-[10px] text-outline">Pairwise temporal correlation</span>
            </div>

            <div className="p-3 rounded-2xl bg-surface-container-low">
              <span className="text-[10px] text-outline font-bold uppercase block">Latent Dim & FFN</span>
              <span className="text-base font-extrabold text-espresso mt-0.5 block">
                d_model = {modelConfig?.d_model || 32} / FFN = {modelConfig?.ffn_multiplier || '2x'}
              </span>
              <span className="text-[10px] text-outline">{modelConfig?.activation || 'GELU'} · {modelConfig?.dropout || '20%'} Dropout</span>
            </div>

            <div className="p-3 rounded-2xl bg-surface-container-low">
              <span className="text-[10px] text-outline font-bold uppercase block">Pooling Strategy</span>
              <span className="text-base font-extrabold text-espresso mt-0.5 block">
                {modelConfig?.pooling || 'CLS'} Token
              </span>
              <span className="text-[10px] text-outline">Learnable prefix sequence token</span>
            </div>

            <div className="p-3 rounded-2xl bg-surface-container-low">
              <span className="text-[10px] text-outline font-bold uppercase block">Trainable Parameters</span>
              <span className="text-base font-extrabold text-cognac font-mono mt-0.5 block">
                {modelConfig?.total_parameters?.toLocaleString() || '33,383'}
              </span>
              <span className="text-[10px] text-outline">100% trainable · 0 frozen</span>
            </div>
          </div>
        </div>

        {/* Training Configuration Card */}
        <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-sandBorder/60">
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-cognac" />
              <h3 className="text-base font-bold text-espresso tracking-tight">
                Section 3B: Training & Calibration Specification
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-100 text-tertiary text-[11px] font-bold">
              AdamW Optimizer
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-3 rounded-2xl bg-surface-container-low">
              <span className="text-[10px] text-outline font-bold uppercase block">Optimizer & Decay</span>
              <span className="text-base font-extrabold text-espresso mt-0.5 block">
                {modelConfig?.optimizer || 'AdamW'}
              </span>
              <span className="text-[10px] text-outline">lr = {modelConfig?.learning_rate || '0.0002'} · decay = {modelConfig?.weight_decay || '0.0001'}</span>
            </div>

            <div className="p-3 rounded-2xl bg-surface-container-low">
              <span className="text-[10px] text-outline font-bold uppercase block">Batch & Epochs</span>
              <span className="text-base font-extrabold text-espresso mt-0.5 block">
                Batch {modelConfig?.batch_size || 64} / {modelConfig?.epochs || 8} Epochs
              </span>
              <span className="text-[10px] text-outline">Supervised mini-batch DataLoader</span>
            </div>

            <div className="p-3 rounded-2xl bg-surface-container-low">
              <span className="text-[10px] text-outline font-bold uppercase block">Multi-Task Objective</span>
              <span className="text-base font-extrabold text-espresso mt-0.5 block">
                BCE + {modelConfig?.loss_lambda || 0.3} × Hazard
              </span>
              <span className="text-[10px] text-outline">Joint failure + survival objective</span>
            </div>

            <div className="p-3 rounded-2xl bg-surface-container-low">
              <span className="text-[10px] text-outline font-bold uppercase block">Class Weighting</span>
              <span className="text-base font-extrabold text-espresso mt-0.5 block">
                pos_weight = {modelConfig?.positive_weight || 8.62}
              </span>
              <span className="text-[10px] text-outline">Compensates ~4.2% default imbalance</span>
            </div>

            <div className="p-3 rounded-2xl bg-surface-container-low col-span-2">
              <span className="text-[10px] text-outline font-bold uppercase block">Platt Calibration Parameters</span>
              <span className="text-sm font-bold text-espresso font-mono mt-0.5 block">
                P(Distress) = 1 / (1 + exp(-({modelConfig?.platt_slope || 0.1456} × z + {modelConfig?.platt_intercept || -3.3719})))
              </span>
              <span className="text-[10px] text-outline">Operational Queue Threshold (Top 10%): {modelConfig?.alert_threshold || '2.85%'} calibrated risk</span>
            </div>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 4: 4 BENCHMARK PERFORMANCE COMPARISON GRAPHS */}
      {/* ========================================================================= */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-sandBorder/60">
          <TrendingUp className="w-5 h-5 text-cognac" />
          <h2 className="text-lg font-bold text-espresso tracking-tight">
            Section 4: Performance Comparison Across 11 Architectures
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Graph 1: PR-AUC Horizontal Bar Chart */}
          <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-espresso">GRAPH 1: PR-AUC by Model Architecture</h3>
                <span className="text-[11px] text-outline">Ranking performance under severe class imbalance (~4% defaults)</span>
              </div>
              <span className="px-2 py-0.5 rounded bg-amber-100 text-cognac text-[10px] font-bold uppercase">
                Higher = Better
              </span>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartDataPR} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis type="number" domain={[0, 1.0]} tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="model" tick={{ fontSize: 10 }} />
                  <Tooltip
                    formatter={(val: any) => [Number(val).toFixed(4), 'PR-AUC']}
                    contentStyle={{ backgroundColor: '#FAF8F5', borderColor: '#E2DCD5', borderRadius: '12px' }}
                  />
                  <Bar dataKey="pr_auc" radius={[0, 6, 6, 0]}>
                    {chartDataPR.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.isM6 ? '#C85A17' : '#64748B'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Graph 2: ROC-AUC Horizontal Bar Chart */}
          <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-espresso">GRAPH 2: ROC-AUC by Model Architecture</h3>
                <span className="text-[11px] text-outline">Overall borrower discrimination capacity (Baseline = 0.50)</span>
              </div>
              <span className="px-2 py-0.5 rounded bg-amber-100 text-cognac text-[10px] font-bold uppercase">
                Higher = Better
              </span>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartDataROC} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis type="number" domain={[0.4, 1.0]} tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="model" tick={{ fontSize: 10 }} />
                  <Tooltip
                    formatter={(val: any) => [Number(val).toFixed(4), 'ROC-AUC']}
                    contentStyle={{ backgroundColor: '#FAF8F5', borderColor: '#E2DCD5', borderRadius: '12px' }}
                  />
                  <ReferenceLine x={0.5} stroke="#EF4444" strokeDasharray="3 3" label={{ value: 'Random (0.5)', fill: '#EF4444', fontSize: 10 }} />
                  <Bar dataKey="roc_auc" radius={[0, 6, 6, 0]}>
                    {chartDataROC.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.isM6 ? '#C85A17' : '#64748B'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Graph 3: Brier Score (Lower = Better) */}
          <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-espresso">GRAPH 3: Brier Score (Probability Error)</h3>
                <span className="text-[11px] text-outline">Mean squared error between predicted probability and actual default</span>
              </div>
              <span className="px-2 py-0.5 rounded bg-emerald-100 text-tertiary text-[10px] font-bold uppercase">
                Lower = Better Calibration
              </span>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartDataBrier} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis type="number" domain={[0, 0.05]} tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="model" tick={{ fontSize: 10 }} />
                  <Tooltip
                    formatter={(val: any) => [Number(val).toFixed(4), 'Brier Score']}
                    contentStyle={{ backgroundColor: '#FAF8F5', borderColor: '#E2DCD5', borderRadius: '12px' }}
                  />
                  <Bar dataKey="brier" radius={[0, 6, 6, 0]}>
                    {chartDataBrier.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.isM6 ? '#C85A17' : '#64748B'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Graph 4: Capture @ Top 10% Horizontal Bar Chart */}
          <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-espresso">GRAPH 4: Capture Rate @ Top 10% Review Queue</h3>
                <span className="text-[11px] text-outline">Percentage of all defaulting facilities flagged within the top decile</span>
              </div>
              <span className="px-2 py-0.5 rounded bg-amber-100 text-cognac text-[10px] font-bold uppercase">
                Operational Utility
              </span>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartDataCapture} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="model" tick={{ fontSize: 10 }} />
                  <Tooltip
                    formatter={(val: any) => [`${val}%`, 'Capture @ 10%']}
                    contentStyle={{ backgroundColor: '#FAF8F5', borderColor: '#E2DCD5', borderRadius: '12px' }}
                  />
                  <Bar dataKey="capture" radius={[0, 6, 6, 0]}>
                    {chartDataCapture.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.isM6 ? '#C85A17' : '#64748B'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 5 & 6: M6 TRAINING CURVES & PROBABILITY CALIBRATION */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* GRAPH 5: M6 Training Behaviour */}
        <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-sandBorder/60">
            <div>
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-cognac" />
                <h3 className="text-base font-bold text-espresso tracking-tight">
                  GRAPH 5: M6 Multi-Task Training Curves
                </h3>
              </div>
              <span className="text-[11px] text-outline block mt-0.5">
                Authentic AdamW progression: Train Loss vs Val Loss across supervised epochs
              </span>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-100 text-tertiary text-[11px] font-bold">
              AdamW lr=0.0002
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingHistory} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} label={{ value: 'Loss', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#FAF8F5', borderColor: '#E2DCD5', borderRadius: '12px', fontSize: '11px' }} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Line type="monotone" dataKey="train_loss" name="Training Loss" stroke="#C85A17" strokeWidth={2.5} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="val_loss" name="Validation Loss" stroke="#64748B" strokeWidth={2} strokeDasharray="4 4" dot={{ r: 3 }} />
                <Line type="monotone" dataKey="val_pr_auc" name="Val PR-AUC" stroke="#059669" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="p-3 rounded-2xl bg-surface-container-low text-[11px] text-outline flex items-center justify-between">
            <span>Validation PR-AUC convergence: <strong>0.5987 → 0.8783</strong></span>
            <span>Early stopping patience: <strong>5 epochs</strong></span>
          </div>
        </div>

        {/* GRAPH 6: M6 Calibration Curve (Reliability Diagram) */}
        <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-sandBorder/60">
            <div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-cognac" />
                <h3 className="text-base font-bold text-espresso tracking-tight">
                  GRAPH 6: M6 Platt Probability Calibration
                </h3>
              </div>
              <span className="text-[11px] text-outline block mt-0.5">
                Reliability diagram: Mean predicted probability vs Observed default rate
              </span>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-outline text-[11px] font-mono font-bold">
              ECE = 0.0124
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={calibrationCurve} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  dataKey="mean_pred"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
                  label={{ value: 'Mean Predicted Risk', position: 'insideBottom', offset: -5, fontSize: 10 }}
                />
                <YAxis
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
                  label={{ value: 'Observed Rate', angle: -90, position: 'insideLeft', fontSize: 10 }}
                />
                <Tooltip
                  formatter={(val: any) => [`${(Number(val) * 100).toFixed(2)}%`]}
                  contentStyle={{ backgroundColor: '#FAF8F5', borderColor: '#E2DCD5', borderRadius: '12px', fontSize: '11px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Line
                  type="monotone"
                  dataKey="baseline_perfect"
                  name="Perfect Calibration (y = x)"
                  stroke="#94A3B8"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="observed_rate"
                  name="M6 Calibrated Probability"
                  stroke="#C85A17"
                  strokeWidth={2.5}
                  dot={{ r: 4, fill: '#C85A17' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="p-3 rounded-2xl bg-surface-container-low text-[11px] text-outline flex items-center justify-between">
            <span>Platt slope: <strong>0.1456</strong> | Intercept: <strong>-3.3719</strong></span>
            <span>CECL / Basel III Target Brier: <strong>0.0194</strong></span>
          </div>
        </div>
      </div>
    </div>
  );
};
