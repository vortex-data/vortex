// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * Shared display formatting helpers for the read UI.
 *
 * These mirror the v3 server-rendered HTML layer so the Next.js port reads the
 * same as the system it replaces. The summary card's time values arrive in
 * nanoseconds (the `Summary` wire shape from `lib/summary.ts`, matching the
 * Rust `value_ns` columns), so [`formatTimeNs`] is the nanosecond-input port of
 * `server/src/html/summary.rs::format_time_ns`.
 */

/**
 * Format a nanosecond duration into a compact `s` / `ms` / `us` / `ns` string,
 * the port of `server/src/html/summary.rs::format_time_ns`.
 *
 * The tier thresholds and the two-decimal (`s` / `ms` / `us`) vs zero-decimal
 * (`ns`) precision match the Rust source exactly; the unit suffix is a leading
 * space plus the ASCII unit (`us`, not `μs`), again matching v3.
 */
export function formatTimeNs(ns: number): string {
  const abs = Math.abs(ns);
  if (abs >= 1_000_000_000) {
    return `${(ns / 1_000_000_000).toFixed(2)} s`;
  }
  if (abs >= 1_000_000) {
    return `${(ns / 1_000_000).toFixed(2)} ms`;
  }
  if (abs >= 1_000) {
    return `${(ns / 1_000).toFixed(2)} us`;
  }
  return `${ns.toFixed(0)} ns`;
}
