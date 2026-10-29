import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Shield, Zap, Lock, UserCheck, ArrowRight, AlertCircle } from 'lucide-react';

export const Login: React.FC = () => {
  const { login, loginAsRole, register } = useAuth();
  const navigate = useNavigate();

  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleRoleQuickLogin = async (role: 'analyst' | 'examiner') => {
    setError('');
    setSubmitting(true);
    try {
      await loginAsRole(role);
      navigate('/');
    } catch (err: any) {
      setError(err.message || 'Quick login failed');
    } finally {
      setSubmitting(false);
    }
  };

  const handleStandardSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      if (isRegister) {
        await register({
          username,
          email,
          password,
          full_name: fullName,
        });
      } else {
        await login(username, password);
      }
      navigate('/');
    } catch (err: any) {
      setError(err.message || 'Authentication failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 sm:p-6 lg:p-8 bg-background">
      <main className="w-full max-w-[690px] flex flex-col items-center">
        {/* Top Brand Header Block */}
        <header className="flex flex-col items-center text-center mb-6">
          <div className="relative flex items-center justify-center w-12 h-12 mb-3.5 rounded-2xl bg-gradient-to-b from-[#2D2622] to-[#1C1714] text-amber-400 shadow-md shadow-amber-950/15 border border-[#3E3530]">
            <Shield className="w-6 h-6 stroke-[1.8]" />
            <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-amber-500"></span>
            </span>
          </div>

          <div className="flex items-center justify-center gap-2 mb-1.5 flex-wrap">
            <h1 className="text-xl sm:text-[22px] font-bold text-espresso tracking-tight">
              CreditPulse Sovereign Risk
            </h1>
            <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-mono font-medium bg-[#EFE9DD] text-[#786D5F] border border-[#DDD5C7]">
              v4.2.1 BASEL III
            </span>
          </div>

          <p className="text-xs sm:text-[13px] text-[#6E6457] font-medium tracking-normal max-w-lg">
            Institutional Loan Distress Early-Warning & Forensic Attribution Platform
          </p>
        </header>

        {error && (
          <div className="w-full mb-4 p-3 rounded-xl bg-red-50 border border-red-200 text-error text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Quick Role Selection Card (Matches Stitch Screenshot 2) */}
        <section className="w-full bg-[#26201D] text-[#EFEBE4] rounded-2xl p-5 sm:p-6 shadow-xl shadow-stone-900/10 border border-[#38302B] mb-5">
          <div className="mb-4">
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-[#3D332D] text-[#F3BA63] text-[10px] font-bold tracking-wider uppercase mb-2 border border-[#4E413A]">
              <Zap className="w-3 h-3 text-[#E59837]" />
              INSTITUTIONAL EXAMINATION
            </div>
            <h2 className="text-[17px] font-bold text-white tracking-tight leading-snug">
              Instant Role-Based Examination Login
            </h2>
            <p className="text-xs text-[#A89F93] mt-0.5 font-normal">
              Bypass single sign-on to inspect deterministic distress telemetry graphs and diagnostic suites.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-1">
            {/* Role 1: Standard Risk Analyst */}
            <div className="bg-[#FAF7F2] rounded-xl p-3.5 flex flex-col justify-between text-espresso border border-[#EDE5D8] shadow-sm">
              <div>
                <div className="flex items-start gap-2.5 mb-2">
                  <div className="w-7 h-7 rounded-lg bg-[#F1E8DB] flex items-center justify-center text-cognac shrink-0 mt-0.5">
                    <UserCheck className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-xs font-bold text-espresso leading-tight">Standard Risk Analyst</h3>
                    <code className="text-[10px] text-outline font-mono">desk_officer</code>
                  </div>
                </div>
                <p className="text-[11px] text-[#5C5349] leading-relaxed mb-3">
                  Telemetry inspection, node timeline & sovereign facility portfolio replays.
                </p>
              </div>
              <button
                type="button"
                disabled={submitting}
                onClick={() => handleRoleQuickLogin('analyst')}
                className="w-full py-2 px-3 rounded-lg bg-cognac hover:bg-[#964304] text-white text-xs font-bold flex items-center justify-center gap-1.5 transition-colors shadow-sm disabled:opacity-50"
              >
                <span>LOGIN AS RISK ANALYST</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Role 2: Senior Risk Examiner */}
            <div className="bg-[#FAF7F2] rounded-xl p-3.5 flex flex-col justify-between text-espresso border border-[#EDE5D8] shadow-sm">
              <div>
                <div className="flex items-start gap-2.5 mb-2">
                  <div className="w-7 h-7 rounded-lg bg-[#F1E8DB] flex items-center justify-center text-cognac shrink-0 mt-0.5">
                    <Shield className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-xs font-bold text-espresso leading-tight">Senior Risk Examiner</h3>
                    <code className="text-[10px] text-outline font-mono">lead_auditor</code>
                  </div>
                </div>
                <p className="text-[11px] text-[#5C5349] leading-relaxed mb-3">
                  Full root-cause attribution, covenant rubric audits & stress test synthesis.
                </p>
              </div>
              <button
                type="button"
                disabled={submitting}
                onClick={() => handleRoleQuickLogin('examiner')}
                className="w-full py-2 px-3 rounded-lg bg-[#E6DFD5] hover:bg-[#D8CFBF] text-espresso text-xs font-bold flex items-center justify-center gap-1.5 transition-colors shadow-sm disabled:opacity-50"
              >
                <span>LOGIN AS EXAMINER</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </section>

        {/* Traditional Credentials Login Box */}
        <div className="w-full bg-[#FFFFFF] rounded-2xl p-6 border border-sandBorder shadow-warm">
          <div className="flex items-center justify-center gap-3 mb-5 border-b border-sandBorder pb-3">
            <button
              type="button"
              onClick={() => setIsRegister(false)}
              className={`pb-1 px-4 text-xs font-bold transition-all ${
                !isRegister
                  ? 'border-b-2 border-cognac text-espresso'
                  : 'text-outline hover:text-espresso'
              }`}
            >
              Sign In with Account
            </button>
            <button
              type="button"
              onClick={() => setIsRegister(true)}
              className={`pb-1 px-4 text-xs font-bold transition-all ${
                isRegister
                  ? 'border-b-2 border-cognac text-espresso'
                  : 'text-outline hover:text-espresso'
              }`}
            >
              Register New Examiner
            </button>
          </div>

          <form onSubmit={handleStandardSubmit} className="space-y-4">
            {isRegister && (
              <div>
                <label className="block text-xs font-bold text-espresso mb-1">Full Legal Name</label>
                <input
                  type="text"
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="e.g. Dr. Jordan Vance, CFA"
                  className="w-full px-3.5 py-2 rounded-xl border border-sandBorder text-xs text-espresso focus:outline-none focus:ring-2 focus:ring-primary-container"
                />
              </div>
            )}

            <div>
              <label className="block text-xs font-bold text-espresso mb-1">Username or Institutional ID</label>
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g. analyst or examiner"
                className="w-full px-3.5 py-2 rounded-xl border border-sandBorder text-xs text-espresso focus:outline-none focus:ring-2 focus:ring-primary-container"
              />
            </div>

            {isRegister && (
              <div>
                <label className="block text-xs font-bold text-espresso mb-1">Institutional Email</label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="e.g. j.vance@centralbank.org"
                  className="w-full px-3.5 py-2 rounded-xl border border-sandBorder text-xs text-espresso focus:outline-none focus:ring-2 focus:ring-primary-container"
                />
              </div>
            )}

            <div>
              <label className="block text-xs font-bold text-espresso mb-1">Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full px-3.5 py-2 rounded-xl border border-sandBorder text-xs text-espresso focus:outline-none focus:ring-2 focus:ring-primary-container"
              />
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="w-full py-2.5 rounded-full bg-espresso hover:bg-stone-800 text-white text-xs font-bold flex items-center justify-center gap-2 transition-all shadow-md disabled:opacity-50"
            >
              <span>{isRegister ? 'Register Institutional Account' : 'Sign In with Institutional Account'}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>
        </div>

        {/* Footer Telemetry Status */}
        <footer className="mt-8 flex flex-wrap items-center justify-center gap-4 text-[11px] font-mono text-outline">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span>Cluster: us-east-sovereign-4</span>
          </div>
          <span>• Latency: 14ms</span>
          <span>• Basel III / CECL Rubrics</span>
          <div className="flex items-center gap-1 text-emerald-700 font-semibold">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
            <span>Platform Status: Operational</span>
          </div>
        </footer>
      </main>
    </div>
  );
};
