// Date-range helpers for the Berka dataset (1993-01-31 to 1998-12-31).
// These bounds are derived from the backend config and /health response.

export const DATA_START = '1993-01-31';
export const DATA_END   = '1998-12-31';

/** Return the last day of a given year-month (YYYY-MM) as a YYYY-MM-DD string. */
function lastDayOfMonth(year: number, month: number): string {
  const d = new Date(year, month, 0); // day=0 → last day of previous month
  return `${year}-${String(month).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

/** Generate an array of month-end dates from DATA_START to DATA_END. */
export function generateMonthEnds(): string[] {
  const result: string[] = [];
  // Start from 1993-01
  for (let year = 1993; year <= 1998; year++) {
    const endMonth = year === 1998 ? 12 : 12;
    for (let month = 1; month <= endMonth; month++) {
      result.push(lastDayOfMonth(year, month));
    }
  }
  return result;
}

/** Check whether a YYYY-MM-DD string is within the supported data range. */
export function isInRange(dateStr: string): boolean {
  return dateStr >= DATA_START && dateStr <= DATA_END;
}

/** A reasonable default date for demonstrations — last data point. */
export const DEFAULT_DATE = DATA_END;
