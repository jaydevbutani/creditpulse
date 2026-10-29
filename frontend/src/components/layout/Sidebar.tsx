import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  AlertTriangle,
  Search,
  Calculator,
  ActivitySquare,
  Info,
  TrendingUp,
} from 'lucide-react';

const navItems = [
  { to: '/',           label: 'Overview',                icon: LayoutDashboard },
  { to: '/monitoring', label: 'Risk Monitoring',         icon: AlertTriangle },
  { to: '/explorer',   label: 'Loan Explorer',           icon: Search },
  { to: '/scoring',    label: 'Individual Scoring',      icon: Calculator },
  { to: '/status',     label: 'Model & System Status',   icon: ActivitySquare },
  { to: '/about',      label: 'About CreditPulse',       icon: Info },
];

export const Sidebar: React.FC = () => (
  <aside className="fixed inset-y-0 left-0 w-64 bg-navy-900 flex flex-col z-30 shadow-2xl">
    {/* Brand */}
    <div className="px-6 pt-7 pb-6 border-b border-navy-800">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 bg-teal-400 rounded-lg flex items-center justify-center flex-shrink-0">
          <TrendingUp className="w-5 h-5 text-navy-900" strokeWidth={2.5} />
        </div>
        <div>
          <p className="text-white font-semibold text-sm leading-tight">CreditPulse</p>
          <p className="text-navy-300 text-xs leading-tight mt-0.5">Risk Intelligence</p>
        </div>
      </div>
    </div>

    {/* Navigation */}
    <nav className="flex-1 px-3 py-4 overflow-y-auto">
      <p className="text-navy-400 text-xs font-semibold uppercase tracking-widest px-3 mb-3">
        Navigation
      </p>
      <ul className="space-y-0.5">
        {navItems.map(({ to, label, icon: Icon }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-navy-700 text-white'
                    : 'text-navy-300 hover:bg-navy-800 hover:text-white'
                }`
              }
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span>{label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>

    {/* Disclaimer */}
    <div className="px-4 pb-5 pt-4 border-t border-navy-800">
      <p className="text-navy-400 text-xs leading-relaxed">
        Decision support for human analysts only. Risk estimates are not guarantees and must not be
        used as automated lending decisions.
      </p>
    </div>
  </aside>
);
