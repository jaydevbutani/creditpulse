import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, Search, UploadCloud, FolderKanban, Cpu, LogOut } from 'lucide-react';

export const Navbar: React.FC = () => {
  const { user, logout, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="w-full sticky top-0 z-50">
      {/* Top Telemetry Ticker Bar */}
      <div className="bg-surface-container border-b border-sandBorder/60 py-1.5 px-4 sm:px-6 lg:px-8">
        <div className="max-w-[1560px] mx-auto flex items-center justify-between text-xs">
          <div className="flex items-center gap-3">
            <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full bg-surface-container-lowest border border-sandBorder shadow-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-[11px] font-semibold text-tertiary">
                Decision Support Active • Berka (PKDD'99) 1993–1998 • Multi-Task Transformer M6
              </span>
            </div>
            <span className="hidden sm:inline text-outline text-[11px]">
              Trained on Czech Bank accounts (682 loans, 12-month sequence tokens)
            </span>
          </div>
          <div className="hidden md:flex items-center gap-4 text-[11px] text-outline font-mono">
            <span>
              Evaluation Split: <strong className="text-tertiary font-semibold">P1 Out-of-Time</strong>
            </span>
            <span>
              Engine: <strong className="text-espresso font-semibold">M6_multitask_ens5</strong>
            </span>
          </div>
        </div>
      </div>

      {/* Main Header Bar */}
      <header className="bg-surface/90 backdrop-blur-xl border-b border-sandBorder/70 shadow-warm-sm">
        <div className="max-w-[1560px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-6">
          {/* Brand Logo */}
          <div className="flex items-center gap-6">
            <NavLink to="/" className="flex items-center gap-2.5 group">
              <div className="w-9 h-9 rounded-xl bg-espresso flex items-center justify-center text-primary-container shadow-md group-hover:scale-105 transition-transform">
                <Activity className="w-5 h-5" />
              </div>
              <div className="flex flex-col">
                <span className="font-extrabold text-lg text-espresso tracking-tight leading-none">CreditPulse</span>
                <span className="text-[10px] font-mono text-outline font-medium tracking-wide">TEMPORAL DISTRESS MONITOR</span>
              </div>
            </NavLink>

            {/* Navigation Pills */}
            <nav className="hidden lg:flex items-center gap-1.5 pl-4">
              <NavLink
                to="/"
                end
                className={({ isActive }) =>
                  `px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
                    isActive
                      ? 'bg-espresso text-surface-container-lowest shadow-sm'
                      : 'text-outline hover:text-espresso hover:bg-surface-container'
                  }`
                }
              >
                <Activity className="w-3.5 h-3.5 text-primary-container" />
                Portfolio Monitor
              </NavLink>

              <NavLink
                to="/loan-detail"
                className={({ isActive }) =>
                  `px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
                    isActive
                      ? 'bg-espresso text-surface-container-lowest shadow-sm'
                      : 'text-outline hover:text-espresso hover:bg-surface-container'
                  }`
                }
              >
                <Search className="w-3.5 h-3.5" />
                Loan Detail
              </NavLink>

              <NavLink
                to="/upload"
                className={({ isActive }) =>
                  `px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
                    isActive
                      ? 'bg-espresso text-surface-container-lowest shadow-sm'
                      : 'text-outline hover:text-espresso hover:bg-surface-container'
                  }`
                }
              >
                <UploadCloud className="w-3.5 h-3.5" />
                Upload & Analyze
              </NavLink>

              <NavLink
                to="/history"
                className={({ isActive }) =>
                  `px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
                    isActive
                      ? 'bg-espresso text-surface-container-lowest shadow-sm'
                      : 'text-outline hover:text-espresso hover:bg-surface-container'
                  }`
                }
              >
                <FolderKanban className="w-3.5 h-3.5" />
                Case History
              </NavLink>

              <NavLink
                to="/model-health"
                className={({ isActive }) =>
                  `px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
                    isActive
                      ? 'bg-espresso text-surface-container-lowest shadow-sm'
                      : 'text-outline hover:text-espresso hover:bg-surface-container'
                  }`
                }
              >
                <Cpu className="w-3.5 h-3.5" />
                Model Health
              </NavLink>
            </nav>
          </div>

          {/* Right Section: User Profile & Actions */}
          <div className="flex items-center gap-3">
            {isAuthenticated && user ? (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 pl-1.5 py-1 pr-3 bg-surface-container-lowest rounded-full border border-sandBorder shadow-xs">
                  <div className="w-7 h-7 rounded-full bg-espresso text-amberGlow flex items-center justify-center font-bold text-xs">
                    {user.full_name ? user.full_name.substring(0, 2).toUpperCase() : 'US'}
                  </div>
                  <div className="flex flex-col text-left">
                    <span className="text-xs font-bold text-espresso leading-none">{user.full_name}</span>
                    <span className="text-[10px] text-outline font-medium">{user.role}</span>
                  </div>
                </div>

                <button
                  onClick={handleLogout}
                  title="Sign Out"
                  className="w-8 h-8 rounded-full bg-surface-container hover:bg-red-50 border border-sandBorder flex items-center justify-center text-outline hover:text-error transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <NavLink
                to="/login"
                className="px-4 py-1.5 rounded-full bg-espresso text-white text-xs font-bold hover:bg-stone-800 transition-colors"
              >
                Sign In
              </NavLink>
            )}
          </div>
        </div>
      </header>
    </div>
  );
};
