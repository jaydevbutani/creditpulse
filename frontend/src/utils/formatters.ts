// Utility helpers for formatting values shown in the UI.

/** Format a calibrated probability [0,1] as a percentage string: 0.044 → "4.4%" */
export function fmtRisk(value: number): string {
  return (value * 100).toFixed(1) + '%';
}

/** Format a percentile for display: 0.29 → "top 0.3%" */
export function fmtPercentile(p: number): string {
  return `top ${p.toFixed(1)}%`;
}

/** Format CZK amount with thousands separators */
export function fmtCZK(value: number): string {
  return new Intl.NumberFormat('cs-CZ', {
    style: 'currency',
    currency: 'CZK',
    maximumFractionDigits: 0,
  }).format(value);
}

/** Format a YYYY-MM-DD or YYYY-MM date string for display */
export function fmtDate(d: string): string {
  if (!d) return '—';
  // Handle YYYY-MM
  if (/^\d{4}-\d{2}$/.test(d)) return d;
  const parts = d.split('-');
  if (parts.length === 3) {
    const [y, m, day] = parts;
    return `${day} ${monthName(parseInt(m, 10))} ${y}`;
  }
  return d;
}

function monthName(m: number): string {
  return ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m - 1] ?? '';
}

/** Format a duration in months */
export function fmtMonths(m: number): string {
  return `${m} month${m !== 1 ? 's' : ''}`;
}

/** Classify a risk_6m probability into a tier */
export function riskTier(risk: number): 'low' | 'medium' | 'high' {
  if (risk >= 0.05) return 'high';
  if (risk >= 0.025) return 'medium';
  return 'low';
}

/** Clamp a number to [min, max] */
export function clamp(x: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, x));
}
