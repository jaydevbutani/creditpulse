import React from 'react';
import { generateMonthEnds, DATA_START, DATA_END } from '../../utils/dateRange';

interface DateSelectorProps {
  value: string;
  onChange: (date: string) => void;
  label?: string;
  disabled?: boolean;
}

const MONTH_ENDS = generateMonthEnds();

export const DateSelector: React.FC<DateSelectorProps> = ({
  value,
  onChange,
  label = 'Monitoring date',
  disabled = false,
}) => (
  <div className="flex flex-col gap-1">
    <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
      {label}
    </label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className="px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white text-navy-900 font-medium shadow-sm
                 focus:outline-none focus:ring-2 focus:ring-navy-500 focus:border-transparent
                 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {MONTH_ENDS.map((d) => (
        <option key={d} value={d}>
          {d}
        </option>
      ))}
    </select>
    <p className="text-xs text-slate-400">
      Supported range: {DATA_START} – {DATA_END}
    </p>
  </div>
);
