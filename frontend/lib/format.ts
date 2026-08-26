/**
 * How a number is written on screen.
 *
 * ONE RULE THROUGHOUT: the unit comes from the evidence record, never
 * from the component. A figure the engine published as `INR_CR` renders
 * in crore wherever it appears, and a component that decided for itself
 * that a number "looked like lakhs" would be quietly restating a value
 * the engine was careful about.
 */

export function formatValue(
  value: number | string | null,
  unit: string | null,
): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;

  switch ((unit ?? "").toLowerCase()) {
    case "pct":
      return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
    case "pt":
      return `${value > 0 ? "+" : ""}${value.toFixed(1)} pt`;
    case "inr_cr":
      return `₹${value.toFixed(2)} Cr`;
    case "inr_l":
      return `₹${value.toFixed(0)} L`;
    case "inr":
      return `₹${Math.round(value).toLocaleString("en-IN")}`;
    case "ratio":
      return value.toFixed(2);
    case "count":
      return Math.round(value).toLocaleString("en-IN");
    case "hours":
      return `${value.toFixed(0)} h`;
    default:
      return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
}

/** A sparkline's own label. Compact, because it sits under a small chart. */
export function formatCompact(value: number | null, unit: string | null): string {
  if (value === null) return "—";
  const u = (unit ?? "").toLowerCase();
  if (u === "inr_cr" || u === "inr") {
    if (Math.abs(value) >= 1e7) return `₹${(value / 1e7).toFixed(1)} Cr`;
    if (Math.abs(value) >= 1e5) return `₹${(value / 1e5).toFixed(1)} L`;
    return `₹${Math.round(value).toLocaleString("en-IN")}`;
  }
  if (u === "pct") return `${value.toFixed(1)}%`;
  if (u === "pt") return `${value.toFixed(1)} pt`;
  return value.toFixed(1);
}

export function formatPercentChange(value: number | null): string {
  if (value === null) return "n/a";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}

/**
 * Elapsed time to a verdict.
 *
 * Rendered in whatever unit is honest for the size, because this is the
 * measurement that replaced an asserted "11 minutes" (resolved defect 6).
 * A real thirteen seconds should read as thirteen seconds.
 */
export function formatElapsed(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  const seconds = ms / 1000;
  if (seconds < 1) return `${Math.round(ms)} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${Math.round(seconds - minutes * 60)} s`;
}

export function formatHours(hours: number | null | undefined): string {
  if (hours === null || hours === undefined) return "—";
  if (hours < 1) return `${Math.round(hours * 60)} min`;
  if (hours < 48) return `${hours.toFixed(1)} h`;
  return `${(hours / 24).toFixed(1)} days`;
}

export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return `${date.toISOString().replace("T", " ").slice(0, 16)} UTC`;
}

export function titleCase(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** A percentage for a bar width, clamped so a rounding error cannot
 *  produce a bar wider than its track. */
export function share(part: number, whole: number): number {
  if (!whole) return 0;
  return Math.min(100, Math.max(0, (Math.abs(part) / Math.abs(whole)) * 100));
}
