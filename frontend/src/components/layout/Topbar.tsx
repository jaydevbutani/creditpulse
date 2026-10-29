import React from 'react';
import { RefreshCw, Wifi, WifiOff, Loader2 } from 'lucide-react';

interface TopbarProps {
  title: string;
  subtitle?: string;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  connected?: boolean | null; // null = unknown / loading
  lastChecked?: Date | null;
}

export const Topbar: React.FC<TopbarProps> = ({
  title,
  subtitle,
  onRefresh,
  isRefreshing,
  connected,
  lastChecked,
}) => {
  const connLabel =
    connected === null
      ? 'Connecting…'
      : connected
      ? 'Backend connected'
      : 'Backend unreachable';

  const connColor =
    connected === null
      ? 'text-amber-500'
      : connected
      ? 'text-emerald-500'
      : 'text-red-500';

  return (
    <header className="sticky top-0 z-20 bg-white/95 backdrop-blur border-b border-slate-200 px-8 py-4 flex items-center justify-between">
      <div>
        <h1 className="text-navy-900 font-semibold text-xl leading-tight">{title}</h1>
        {subtitle && <p className="text-slate-500 text-sm mt-0.5">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-5">
        {/* Connection indicator */}
        <div className={`flex items-center gap-1.5 text-sm font-medium ${connColor}`}>
          {connected === null ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : connected ? (
            <Wifi className="w-4 h-4" />
          ) : (
            <WifiOff className="w-4 h-4" />
          )}
          <span className="hidden sm:inline">{connLabel}</span>
        </div>

        {lastChecked && (
          <span className="text-xs text-slate-400 hidden md:inline">
            Last checked {lastChecked.toLocaleTimeString()}
          </span>
        )}

        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="flex items-center gap-1.5 text-sm text-navy-700 hover:text-navy-900 font-medium px-3 py-1.5 rounded-lg border border-slate-200 hover:border-navy-300 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        )}
      </div>
    </header>
  );
};
