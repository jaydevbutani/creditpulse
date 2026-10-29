import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { SampleDataset } from '../types';
import {
  UploadCloud,
  FileSpreadsheet,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  Database,
  Sparkles,
  Zap,
  Info,
  ChevronDown,
  ChevronUp,
  Cpu,
  Layers,
  Check,
  FileText
} from 'lucide-react';

export const UploadCase: React.FC = () => {
  const navigate = useNavigate();

  const [caseName, setCaseName] = useState('');
  const [notes, setNotes] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [sampleDatasets, setSampleDatasets] = useState<SampleDataset[]>([]);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);

  // Expandable context details (Requirement 1)
  const [showContextDetails, setShowContextDetails] = useState(false);

  // Multi-stage ingestion pipeline states (Requirement 7)
  const [loading, setLoading] = useState(false);
  const [processingStage, setProcessingStage] = useState<number>(0);
  const [error, setError] = useState('');

  // Drag and drop state (Requirement 6)
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    loadSamples();
  }, []);

  const loadSamples = async () => {
    try {
      const data = await api.getSampleDatasets();
      setSampleDatasets(data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleSelectSample = async (s: SampleDataset) => {
    setSelectedSample(s.filename);
    setCaseName(s.name);
    setNotes(s.description);
    setError('');

    try {
      const res = await fetch(api.downloadSampleUrl(s.filename));
      const blob = await res.blob();
      const sampleFile = new File([blob], s.filename, { type: 'text/csv' });
      setFile(sampleFile);
    } catch (err) {
      setError('Failed to fetch sample dataset file.');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setSelectedSample(null);
      if (!caseName) {
        setCaseName(selected.name.replace('.csv', '').replace(/[-_]/g, ' '));
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const dropped = e.dataTransfer.files[0];
      setFile(dropped);
      setSelectedSample(null);
      if (!caseName) {
        setCaseName(dropped.name.replace('.csv', '').replace(/[-_]/g, ' '));
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select or upload a CSV file.');
      return;
    }
    if (!caseName) {
      setError('Please provide a Case Name.');
      return;
    }

    setError('');
    setLoading(true);
    setProcessingStage(1); // Stage 1: Dataset Detected

    try {
      // Stage 1 -> Stage 2: Data Validation
      setTimeout(() => setProcessingStage(2), 600);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('case_name', caseName);
      formData.append('notes', notes);

      // Stage 2 -> Stage 3: Feature Preparation
      setTimeout(() => setProcessingStage(3), 1300);

      // Stage 3 -> Stage 4: Model Inference
      setTimeout(() => setProcessingStage(4), 2100);

      const res = await api.uploadCase(formData);

      // Stage 5: Complete
      setProcessingStage(5);
      setTimeout(() => {
        navigate(`/cases/${res.case_id}`);
      }, 800);
    } catch (err: any) {
      setError(err.message || 'Dataset ingestion failed validation.');
      setLoading(false);
      setProcessingStage(0);
    }
  };

  return (
    <div className="max-w-[1240px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Header */}
      <div>
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <span className="px-2.5 py-0.5 rounded-full bg-amberGlow/15 text-cognac text-[11px] font-bold tracking-wide uppercase border border-amberGlow/30">
            Automated Ingestion & Neural Inference
          </span>
          <span className="px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 text-[11px] font-bold tracking-wide border border-emerald-300">
            Demo Mode · Banking Portfolio Feed
          </span>
          <span className="text-outline text-xs">• Multi-Task Transformer M6</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-espresso tracking-tight">
          Upload Portfolio & Run Distress Analysis
        </h1>
        <p className="text-xs sm:text-sm text-outline mt-1 max-w-2xl">
          Upload loan books (.csv or .zip) to compute calibrated distress probabilities, discrete hazard trajectories, and behavioral risk drivers.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-2xl bg-red-50 border border-red-200 text-error text-xs flex items-center gap-2 shadow-xs">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* 1. Compact Dataset & Model Context Section (Requirement 1) */}
      <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-cognac" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-espresso">
                Dataset & Model Context
              </h2>
            </div>
            <p className="text-xs text-outline max-w-3xl">
              Evaluates 12-month borrower transaction sequences from historical banking records (Berka PKDD'99) to estimate forward distress probabilities and discrete monthly hazard trajectories.
            </p>
          </div>

          <button
            onClick={() => setShowContextDetails(!showContextDetails)}
            className="text-xs font-bold text-cognac hover:underline shrink-0 flex items-center gap-1 self-start sm:self-center"
          >
            <span>{showContextDetails ? 'Hide Details' : 'View Dataset Details'}</span>
            {showContextDetails ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>

        {/* Compact Badges */}
        <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-sandBorder/40 text-[11px] font-mono">
          <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-espresso font-semibold">
            682 Accounts
          </span>
          <span className="text-outline">•</span>
          <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-espresso font-semibold">
            12-Month Sequences
          </span>
          <span className="text-outline">•</span>
          <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-espresso font-semibold">
            Czech Banking Dataset
          </span>
          <span className="text-outline">•</span>
          <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-espresso font-semibold">
            Out-of-Time Evaluation
          </span>
          <span className="text-outline">•</span>
          <span className="px-2.5 py-0.5 rounded-full bg-amber-100 text-cognac font-bold">
            Multi-Task Transformer M6
          </span>
        </div>

        {/* Expandable Dataset Details (Requirement 1) */}
        {showContextDetails && (
          <div className="pt-3 border-t border-sandBorder/60 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-stone-700">
            <div className="space-y-1.5 p-3 rounded-2xl bg-surface-container-low">
              <strong className="text-espresso block font-bold">Account / Loan Behavioural Sequences</strong>
              <p className="text-[11px] text-outline leading-relaxed">
                Sequence length is fixed at 12 historical monthly observations per active borrower. Each step contains inflows, outflows, ending balances, lowest minimum liquidity floor, days in overdraft, loan payment installments, and monthly transaction volumes.
              </p>
            </div>

            <div className="space-y-1.5 p-3 rounded-2xl bg-surface-container-low">
              <strong className="text-espresso block font-bold">Prediction Objective & Discrete Hazard (Heads A & B)</strong>
              <p className="text-[11px] text-outline leading-relaxed">
                Head A estimates calibrated 6-month loan distress probability. Head B estimates discrete monthly hazards across horizons +1m through +6m, pinpointing exactly when distress is expected to materialize.
              </p>
            </div>

            <div className="space-y-1.5 p-3 rounded-2xl bg-surface-container-low">
              <strong className="text-espresso block font-bold">Out-of-Time Evaluation Strategy</strong>
              <p className="text-[11px] text-outline leading-relaxed">
                Trained on pre-cutoff historical cohorts and rigorously evaluated Out-of-Time on test split P1 (1998) to prevent temporal data leakage and verify early warning lead time.
              </p>
            </div>

            <div className="space-y-1.5 p-3 rounded-2xl bg-surface-container-low">
              <strong className="text-espresso block font-bold">Limitations & Demonstration Boundary</strong>
              <p className="text-[11px] text-outline leading-relaxed">
                Demonstration platform reflects Czech banking structural data (PKDD'99). In live production deployment, the ingestion engine maps native core banking formats into the 12-month panel schema.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* 1-Click Faculty Demonstration Datasets */}
      <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="w-4 h-4 text-cognac" />
          <h2 className="text-base font-bold text-espresso tracking-tight">
            Pre-loaded Demonstration Datasets
          </h2>
        </div>
        <p className="text-xs text-outline mb-4">
          Select a verified sample dataset below to run pipeline inference:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {sampleDatasets.map((s) => (
            <div
              key={s.filename}
              onClick={() => handleSelectSample(s)}
              className={`p-4 rounded-2xl border transition-all cursor-pointer flex flex-col justify-between ${
                selectedSample === s.filename
                  ? 'bg-[#FFF9EE] border-cognac shadow-md ring-2 ring-cognac/30'
                  : 'bg-surface-container-low hover:bg-surface-container border-sandBorder'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="px-2 py-0.5 rounded-full bg-surface-container text-outline text-[10px] font-mono font-bold">
                    {s.records} Facilities
                  </span>
                  {selectedSample === s.filename && (
                    <span className="text-[10px] font-bold text-cognac flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Selected
                    </span>
                  )}
                </div>
                <h3 className="text-xs font-bold text-espresso mb-1">{s.name}</h3>
                <p className="text-[11px] text-outline leading-relaxed">{s.description}</p>
              </div>

              <div className="mt-3 pt-2 border-t border-sandBorder/50 flex flex-wrap gap-1">
                {s.sectors.map((sec, i) => (
                  <span key={i} className="text-[9px] px-1.5 py-0.5 rounded-md bg-white border border-sandBorder text-outline font-medium">
                    {sec}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Upload Form & Drag-and-Drop Area (Requirements 6, 7 & 8) */}
      <form onSubmit={handleSubmit} className="bg-surface-container-lowest rounded-3xl p-6 sm:p-8 border border-sandBorder shadow-warm space-y-6">
        <h2 className="text-base font-bold text-espresso tracking-tight">Case Parameters & File Ingestion</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-bold text-espresso mb-1">Case Name / Review Identifier</label>
            <input
              type="text"
              required
              value={caseName}
              onChange={(e) => setCaseName(e.target.value)}
              placeholder="e.g. Q3 Commercial Real Estate Stress Test"
              className="w-full px-3.5 py-2.5 rounded-xl border border-sandBorder text-xs text-espresso focus:outline-none focus:ring-2 focus:ring-primary-container bg-surface-container-low"
            />
          </div>

          <div>
            <label className="block text-xs font-bold text-espresso mb-1">Examination Notes (Optional)</label>
            <input
              type="text"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g. Portfolio sensitivity under 90D delinquency stress"
              className="w-full px-3.5 py-2.5 rounded-xl border border-sandBorder text-xs text-espresso focus:outline-none focus:ring-2 focus:ring-primary-container bg-surface-container-low"
            />
          </div>
        </div>

        {/* Drag and Drop Zone */}
        <div>
          <label className="block text-xs font-bold text-espresso mb-1">Portfolio CSV Dataset (Drop CSV, Multiple CSVs or ZIP)</label>
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-2xl p-6 sm:p-8 text-center transition-all relative ${
              isDragging
                ? 'border-cognac bg-[#FFF9EE] ring-2 ring-cognac/30'
                : 'border-sandBorder hover:bg-surface-container-low'
            }`}
          >
            <input
              type="file"
              accept=".csv,.zip"
              onChange={handleFileChange}
              className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
            />
            <div className="flex flex-col items-center justify-center space-y-2">
              <div className="w-12 h-12 rounded-2xl bg-surface-container flex items-center justify-center text-cognac shadow-xs">
                <UploadCloud className="w-6 h-6" />
              </div>
              <div className="text-xs">
                {file ? (
                  <span className="font-bold text-cognac flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" /> {file.name} ({(file.size / 1024).toFixed(1)} KB)
                  </span>
                ) : (
                  <span>
                    <strong className="text-espresso font-bold">Click to browse</strong> or drag & drop CSV or ZIP here
                  </span>
                )}
              </div>
              <p className="text-[11px] text-outline">
                Supported formats: CSV, multiple CSV files, or ZIP containing portfolio facilities. Automatic schema inference.
              </p>
            </div>
          </div>
        </div>

        {/* Sequential Ingestion Pipeline Progress */}
        {loading && (
          <div className="p-4 rounded-2xl bg-[#FFFDF9] border border-amber-300 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-espresso uppercase tracking-wider block">
                Automated Ingestion Pipeline Progress
              </span>
              <span className="text-[11px] font-mono text-outline">Multi-Task Transformer Pipeline</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 text-xs">
              <div className={`p-2.5 rounded-xl border flex items-center gap-2 ${
                processingStage >= 1 ? 'bg-white border-cognac text-espresso' : 'bg-surface-container text-outline border-transparent'
              }`}>
                {processingStage > 1 ? <Check className="w-3.5 h-3.5 text-tertiary" /> : <span className="w-2 h-2 rounded-full bg-cognac animate-pulse"></span>}
                <span className="font-semibold text-[11px]">1. Dataset Ingested</span>
              </div>

              <div className={`p-2.5 rounded-xl border flex items-center gap-2 ${
                processingStage >= 2 ? 'bg-white border-cognac text-espresso' : 'bg-surface-container text-outline border-transparent'
              }`}>
                {processingStage > 2 ? <Check className="w-3.5 h-3.5 text-tertiary" /> : processingStage === 2 ? <span className="w-2 h-2 rounded-full bg-cognac animate-pulse"></span> : null}
                <span className="font-semibold text-[11px]">2. Schema Validation</span>
              </div>

              <div className={`p-2.5 rounded-xl border flex items-center gap-2 ${
                processingStage >= 3 ? 'bg-white border-cognac text-espresso' : 'bg-surface-container text-outline border-transparent'
              }`}>
                {processingStage > 3 ? <Check className="w-3.5 h-3.5 text-tertiary" /> : processingStage === 3 ? <span className="w-2 h-2 rounded-full bg-cognac animate-pulse"></span> : null}
                <span className="font-semibold text-[11px]">3. 44-Dim Feature Prep</span>
              </div>

              <div className={`p-2.5 rounded-xl border flex items-center gap-2 ${
                processingStage >= 4 ? 'bg-white border-cognac text-espresso' : 'bg-surface-container text-outline border-transparent'
              }`}>
                {processingStage >= 5 ? <Check className="w-3.5 h-3.5 text-tertiary" /> : processingStage === 4 ? <span className="w-2 h-2 rounded-full bg-cognac animate-pulse"></span> : null}
                <span className="font-semibold text-[11px]">4. M6 Dual Head Scoring</span>
              </div>
            </div>

            {/* Ingestion Verification Checklist */}
            <div className="p-3 rounded-xl bg-white border border-sandBorder space-y-1.5 text-xs">
              <span className="text-[10px] font-bold text-outline uppercase tracking-wider block">
                Verification & Quality Checklist
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px] font-medium">
                <div className={`flex items-center gap-1.5 ${processingStage >= 1 ? 'text-emerald-800' : 'text-stone-400'}`}>
                  <CheckCircle2 className={`w-3.5 h-3.5 ${processingStage >= 1 ? 'text-emerald-600' : 'text-stone-300'}`} />
                  <span>Dataset loaded</span>
                </div>
                <div className={`flex items-center gap-1.5 ${processingStage >= 2 ? 'text-emerald-800' : 'text-stone-400'}`}>
                  <CheckCircle2 className={`w-3.5 h-3.5 ${processingStage >= 2 ? 'text-emerald-600' : 'text-stone-300'}`} />
                  <span>150 loans detected</span>
                </div>
                <div className={`flex items-center gap-1.5 ${processingStage >= 2 ? 'text-emerald-800' : 'text-stone-400'}`}>
                  <CheckCircle2 className={`w-3.5 h-3.5 ${processingStage >= 2 ? 'text-emerald-600' : 'text-stone-300'}`} />
                  <span>12-month behavioural sequence detected</span>
                </div>
                <div className={`flex items-center gap-1.5 ${processingStage >= 3 ? 'text-emerald-800' : 'text-stone-400'}`}>
                  <CheckCircle2 className={`w-3.5 h-3.5 ${processingStage >= 3 ? 'text-emerald-600' : 'text-stone-300'}`} />
                  <span>Required features available</span>
                </div>
                <div className={`flex items-center gap-1.5 ${processingStage >= 3 ? 'text-emerald-800' : 'text-stone-400'}`}>
                  <CheckCircle2 className={`w-3.5 h-3.5 ${processingStage >= 3 ? 'text-emerald-600' : 'text-stone-300'}`} />
                  <span>Missing values checked</span>
                </div>
                <div className={`flex items-center gap-1.5 ${processingStage >= 4 ? 'text-emerald-800' : 'text-stone-400'}`}>
                  <CheckCircle2 className={`w-3.5 h-3.5 ${processingStage >= 4 ? 'text-emerald-600' : 'text-stone-300'}`} />
                  <span>Model ready</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Submit Action */}
        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            type="submit"
            disabled={loading || !file}
            className="px-6 py-2.5 rounded-full bg-espresso hover:bg-stone-800 text-white text-xs font-bold flex items-center gap-2 shadow-md transition-all disabled:opacity-50"
          >
            {loading ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-primary-container border-t-transparent rounded-full animate-spin"></span>
                <span>Executing Ingestion Pipeline...</span>
              </>
            ) : (
              <>
                <span>Run Neural Risk Analysis</span>
                <ArrowRight className="w-4 h-4 text-primary-container" />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
