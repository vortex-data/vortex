// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * Server-side commit-window cap used by every chart query, the TypeScript port
 * of `server/src/api/window.rs`.
 *
 * Visual downsampling is the client's job; this module only decides how many of
 * the most recent commits (by `commits.timestamp`) a chart loads.
 */

/** Default window when no `?n=` is supplied. */
export const DEFAULT_COMMIT_WINDOW = 100;

/**
 * Largest numeric `?n=` accepted before clamping. `?n=all` remains the explicit
 * opt-in for full history, with no server-side cap.
 */
export const MAX_NUMERIC_COMMIT_WINDOW = 1000;

/**
 * Resolved commit window. `Last(n)` keeps the most recent `n` commits; `All`
 * returns every commit ever ingested.
 */
export type CommitWindow = { kind: 'all' } | { kind: 'last'; n: number };

/**
 * Parse the `?n=...` query parameter. `null`/`undefined` and malformed values
 * fall back to the default window. `"all"` (any case, surrounding whitespace
 * trimmed) means unbounded. Numeric values are floored to `1` (so `?n=0`
 * becomes `?n=1`) and clamped down to [`MAX_NUMERIC_COMMIT_WINDOW`]. A value
 * that overflows `u32` (the Rust parse type) is treated as malformed and falls
 * back to the default, matching `window.rs`.
 */
export function parseCommitWindow(raw: string | null | undefined): CommitWindow {
  if (raw === null || raw === undefined) {
    return { kind: 'last', n: DEFAULT_COMMIT_WINDOW };
  }
  const trimmed = raw.trim();
  if (/^all$/i.test(trimmed)) {
    return { kind: 'all' };
  }
  // Rust's `str::parse::<u32>()` accepts an optional leading `+` then ASCII
  // digits, and rejects signs, decimals, and overflow. Anything it would reject
  // falls back to the default window.
  if (!/^\+?\d+$/.test(trimmed)) {
    return { kind: 'last', n: DEFAULT_COMMIT_WINDOW };
  }
  const value = Number(trimmed);
  const U32_MAX = 4_294_967_295;
  if (value > U32_MAX) {
    return { kind: 'last', n: DEFAULT_COMMIT_WINDOW };
  }
  const clamped = Math.min(Math.max(value, 1), MAX_NUMERIC_COMMIT_WINDOW);
  return { kind: 'last', n: clamped };
}

/**
 * The `LIMIT` value for the most-recent-commits subquery, or `null` for an
 * unbounded `All` window (no `LIMIT` clause). Mirrors `window.rs::limit_param`.
 */
export function commitWindowLimit(window: CommitWindow): number | null {
  return window.kind === 'all' ? null : window.n;
}

/**
 * Render the window as the value the URL carries (`"100"` / `"all"`). Ported
 * ahead of its consumer: v3's `CommitWindow::url_value` is used by the HTML
 * toolbar and permalink links, which arrive with the PR-4.4.b client islands.
 * Until then this is exercised only by its unit test.
 */
export function commitWindowUrlValue(window: CommitWindow): string {
  return window.kind === 'all' ? 'all' : String(window.n);
}
