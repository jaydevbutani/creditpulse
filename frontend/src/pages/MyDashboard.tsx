import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { api } from '../api';
import { useAuth } from '../context/AuthContext';
import { CaseSummary } from '../types';
import {
  Briefcase,
  AlertOctagon,
  FileSpreadsheet,
  CheckCircle,
  Plus,
  ArrowRight,
  TrendingDown,
  Clock,
  ShieldCheck,
} from 'lucide-react';

export const MyDashboard: React.FC = () => {
  const { user } = useAuth();
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadUserCases();
  }, []);

  const loadUserCases = async () => {
    try {
      setLoading(true);
      const data = await api.getCases();
      setCases(data);
    } catch (err) {
      console.error('Failed to load user cases', err);
    } finally {
      setLoading(false);
    }
  };

  const totalFacilities = cases.reduce((acc, c) => acc + c.total_facilities, 0);
  const totalCritical = cases.reduce((acc, c) => acc + c.critical_count, 0);
  const avgRisk =
    cases.length > 0 ? (cases.reduce((acc, c) => acc + c.avg_risk_6m, 0) / cases.length) * 100 : 0;

  return (
    <div className="max-w-[1560px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Welcome & Context Banner */}
      <div className="bg-espresso text-white rounded-3xl p-6 sm:p-8 border border-stone-800 shadow-dark-card flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-stone-800 text-amber-400 text-xs font-bold uppercase tracking-wider mb-2">
            <ShieldCheck className="w-3.5 h-3.5" />
            ISOLATED USER WORKSPACE
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
            Welcome, {user?.full_name || 'Risk Examiner'}
          </h1>
          <p className="text-xs sm:text-sm text-stone-300 mt-1 max-w-xl">
            {user?.department || 'Institutional Risk'} • Role: <strong className="text-white">{user?.role}</strong>. Your uploaded datasets and cases are completely private and isolated to your institutional account.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <NavLink
            to="/upload"
            className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-cognac hover:bg-[#964304] text-white text-xs font-bold transition-all shadow-md"
          >
            <Plus className="w-4 h-4" />
            <span>Upload New Portfolio CSV</span>
          </NavLink>
        </div>
      </div>

      {/* User Portfolio KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
          <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-2">Monitored Cases</span>
          <span className="text-3xl sm:text-4xl font-extrabold text-espresso tracking-tight">{cases.length}</span>
          <div className="mt-2 text-xs text-outline pt-2 border-t border-sandBorder/40">
            Personal examination history
          </div>
        </div>

        <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
          <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-2">Total Facilities Scored</span>
          <span className="text-3xl sm:text-4xl font-extrabold text-espresso tracking-tight">{totalFacilities}</span>
          <div className="mt-2 text-xs text-outline pt-2 border-t border-sandBorder/40">
            Across uploaded cohorts
          </div>
        </div>

        <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
          <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-2">Critical Flagged Facilities</span>
          <span className="text-3xl sm:text-4xl font-extrabold text-error tracking-tight">{totalCritical}</span>
          <div className="mt-2 text-xs text-error font-bold pt-2 border-t border-sandBorder/40">
            Distress prob &gt; 80%
          </div>
        </div>

        <div className="bg-surface-container-lowest rounded-3xl p-5 border border-sandBorder shadow-warm">
          <span className="text-xs font-bold text-outline uppercase tracking-wider block mb-2">Mean Portfolio Distress</span>
          <span className="text-3xl sm:text-4xl font-extrabold text-cognac tracking-tight">{avgRisk.toFixed(2)}%</span>
          <div className="mt-2 text-xs text-outline pt-2 border-t border-sandBorder/40">
            Calibrated 6-month hazard
          </div>
        </div>
      </div>

      {/* Uploaded Cases Section */}
      <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-lg font-bold text-espresso tracking-tight">Your Portfolio Cases</h2>
            <p className="text-xs text-outline mt-0.5">
              Datasets and evaluation cases uploaded under your profile.
            </p>
          </div>
          <NavLink
            to="/history"
            className="text-xs font-bold text-cognac hover:underline flex items-center gap-1"
          >
            <span>Manage All Cases</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </NavLink>
        </div>

        {loading ? (
          <div className="py-12 text-center text-xs text-outline">Loading your cases...</div>
        ) : cases.length === 0 ? (
          <div className="py-12 text-center space-y-3 bg-surface-container-low rounded-2xl p-6 border border-dashed border-sandBorder">
            <FileSpreadsheet className="w-8 h-8 text-outline mx-auto" />
            <h3 className="text-sm font-bold text-espresso">No datasets uploaded yet</h3>
            <p className="text-xs text-outline max-w-sm mx-auto">
              Upload a portfolio CSV or select a sample dataset below for evaluation.
            </p>
            <NavLink
              to="/upload"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-espresso text-white text-xs font-bold"
            >
              <Plus className="w-3.5 h-3.5" />
              Upload Now
            </NavLink>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {cases.map((c) => (
              <NavLink
                key={c.id}
                to={`/cases/${c.id}`}
                className="p-5 rounded-2xl bg-surface-container-low hover:bg-[#FFF9EE] border border-sandBorder transition-all flex flex-col justify-between group shadow-xs"
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="px-2.5 py-0.5 rounded-full bg-emerald-100 text-tertiary text-[10px] font-bold">
                      {c.status}
                    </span>
                    <span className="text-[10px] font-mono text-outline">
                      {new Date(c.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <h3 className="text-base font-bold text-espresso group-hover:text-cognac transition-colors">
                    {c.case_name}
                  </h3>
                  <p className="text-xs text-outline line-clamp-2 mt-1">
                    {c.notes || 'No notes provided.'}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-sandBorder flex items-center justify-between text-xs">
                  <div>
                    <span className="text-outline text-[11px] block">Facilities</span>
                    <span className="font-extrabold text-espresso">{c.total_facilities}</span>
                  </div>
                  <div>
                    <span className="text-outline text-[11px] block">Critical</span>
                    <span className="font-extrabold text-error">{c.critical_count}</span>
                  </div>
                  <div className="flex items-center gap-1 text-cognac font-bold text-xs">
                    <span>Inspect</span>
                    <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </NavLink>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
