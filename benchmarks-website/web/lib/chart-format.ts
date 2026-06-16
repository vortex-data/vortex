// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * Pure chart helpers shared by the PR-4.4.b client islands, the TypeScript port
 * of the side-effect-free layer of `server/static/chart-init.js`: formatting,
 * the display-unit picker, LTTB downsampling, payload normalization onto the
 * full-history x-axis, visible-range math, and the filter predicates.
 *
 * Everything here is pure (no DOM, no module state), so it is importable from
 * server components, client components, and node-environment vitest alike. The
 * stateful counterparts (fetch queues, filter stores) live in
 * `lib/chart-store.ts`; the Chart.js wiring lives in `components/Chart.tsx`.
 */

import type { ChartHistory, ChartResponse, CommitPoint, SeriesTag, UnitKind } from '@/lib/queries';

// ---------------------------------------------------------------------------
// Constants (ported verbatim from `chart-init.js`).
// ---------------------------------------------------------------------------

/** One frame at ~60fps; throttle for slider drag. */
export const ZOOM_THROTTLE_MS = 16;
/** Pan/zoom rebuild throttle, looser than the slider. */
export const PAN_THROTTLE_MS = 50;
/** The explicit full-history upgrade window (`?n=all`). */
export const FETCH_N = 'all';
/** Initial visible window (last 100 of the fetched commits). */
export const DEFAULT_VISIBLE = 100;
/** The initial per-chart fetch window (`?n=100`). */
export const CHART_FETCH_N = String(DEFAULT_VISIBLE);
/** Per-tab cap for initial latest-100 chart requests. */
export const HYDRATION_CONCURRENCY = 4;
/** Per-tab cap for the per-group bundle fetches (`/api/group/{slug}?n=100`).
 * One in-flight bundle covers a whole group, so the cap bounds how many groups
 * fetch at once on Expand All without serializing the top groups. */
export const BUNDLE_CONCURRENCY = 3;
/** Per-tab cap for background `?n=all` warmup requests. */
export const FULL_HISTORY_CONCURRENCY = 2;
/** Priority for a full-history fetch promoted by direct user interaction. */
export const INTERACTION_FULL_PRIORITY = 1_000_000;
/** A silent hover-dwell prefetch outranks idle background work but yields to a
 * direct user interaction (chip click, pan/zoom into the unloaded region). */
export const HOVER_PREFETCH_PRIORITY = 500_000;
/** How long the pointer must rest on one chart card before the silent
 * full-history prefetch starts, so a mouse sweep across the page fetches
 * nothing while a deliberate hover has data ready by the time the user acts. */
export const HOVER_DWELL_MS = 600;
/** Per-fetch timeout (ms) for the chart `?n=100` / `?n=all` requests. A stalled
 * request aborts at this bound instead of spinning the loading indicator
 * forever. 30s is generous headroom over a cold Vercel function first-hit
 * (~7.8s measured) so a slow-but-live request is not falsely aborted. */
export const FETCH_TIMEOUT_MS = 30000;
/** `IntersectionObserver` root margin for landing-page lazy hydration: a chart
 * begins hydrating slightly before it scrolls into view so it is rarely blank
 * by the time the user reaches it. */
export const LAZY_HYDRATION_ROOT_MARGIN = '300px 0px';

/**
 * Hard cap on how many distinct commit indices (x-positions) a chart renders at
 * once, shared across every series. Below the cap every commit with data
 * renders raw; above it the per-commit max-across-series is LTTB-downsampled to
 * exactly this many representatives. Chart cards are ~600-900px on desktop and
 * Chart.js draws ~2px markers, so 500 points is about as dense as the eye can
 * resolve.
 */
export const MAX_VISIBLE_POINTS = 500;

// ---------------------------------------------------------------------------
// Palette + small formatting helpers.
// ---------------------------------------------------------------------------

/** The fixed series palette; series are colored by sorted-name index. */
export const PALETTE = [
  '#2563eb',
  '#dc2626',
  '#16a34a',
  '#ea580c',
  '#7c3aed',
  '#0891b2',
  '#ca8a04',
  '#db2777',
  '#65a30d',
  '#475569',
] as const;

/** Color for the `i`-th series (wraps around the palette). */
export function colorFor(i: number): string {
  return PALETTE[i % PALETTE.length];
}

/** First 7 characters of a commit SHA. */
export function shortSha(sha: unknown): string {
  return typeof sha === 'string' ? sha.slice(0, 7) : String(sha);
}

/** The `YYYY-MM-DD` prefix of an ISO timestamp, or `''` for non-strings. */
export function shortDate(ts: unknown): string {
  if (typeof ts !== 'string') {
    return '';
  }
  return ts.slice(0, 10);
}

/** Truncate `s` to at most `max` characters, ellipsizing the overflow. */
export function truncate(s: unknown, max: number): string {
  if (typeof s !== 'string') {
    return '';
  }
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

/** Everything before the first newline of `s`, or `''` for non-strings. */
export function firstLine(s: unknown): string {
  if (typeof s !== 'string') {
    return '';
  }
  const nl = s.indexOf('\n');
  return nl >= 0 ? s.slice(0, nl) : s;
}

/**
 * Extract the PR number from a squash-merge subject. Vortex commits to
 * `develop` are squash-merged from PRs and the squash subject ends with
 * `(#NNNN)`; returning just the number lets callers build either a PR or a
 * commit URL.
 */
export function parsePrNumber(message: unknown): string | null {
  if (typeof message !== 'string') {
    return null;
  }
  const m = message.match(/\(#(\d+)\)/);
  return m ? m[1] : null;
}

/**
 * HTML-escape a string for interpolation into tooltip markup. The external
 * tooltip builds `innerHTML` strings (a direct port of `chart-init.js`), so
 * every dynamic value passes through here first.
 */
export function escapeHtml(s: unknown): string {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** The x-axis label for one commit slot: the short SHA, or `''` for a virtual
 * (not yet loaded) slot. */
export function labelForCommit(commit: CommitPoint | null | undefined): string {
  return commit && commit.sha ? shortSha(commit.sha) : '';
}

/**
 * The chronologically preceding raw value for a tooltip row's delta, or `null`
 * when no earlier measurement exists. `commits[]` is sorted oldest-first by
 * SQL, so the predecessor of index `idx` lives at `idx - 1` (BAN-pinned: a
 * "fix" flipping this walk to `idx + 1` has been reverted before); the walk
 * continues back across null-valued slots so series that did not run on every
 * commit still get a meaningful baseline.
 */
export function predecessorValue(
  rawData: readonly (number | null | undefined)[],
  idx: number,
): number | null {
  let prevIdx = idx - 1;
  while (prevIdx >= 0) {
    const pv = rawData[prevIdx];
    if (pv !== null && pv !== undefined && !Number.isNaN(pv)) {
      return pv;
    }
    prevIdx -= 1;
  }
  return null;
}

/**
 * The single search param value, or `null` when the param is absent or
 * repeated. Shared by the landing and chart permalink pages for `?engine=`,
 * `?format=`, and `?n=`.
 */
export function singleSearchParam(v: string | string[] | undefined): string | null {
  return typeof v === 'string' ? v : null;
}

/** The short human label for a wire `unit_kind`, ported from the Rust
 * `UnitKind::label` (used by the chart permalink page's meta line). */
export function unitKindLabel(unitKind: UnitKind): string {
  switch (unitKind) {
    case 'time_ns':
      return 'ns';
    case 'bytes':
      return 'bytes';
    case 'ratio':
      return 'ratio';
    case 'count':
      return 'count';
    case 'throughput_mb_s':
      return 'MB/s';
  }
}

// ---------------------------------------------------------------------------
// Display unit picker. The wire payload's `unit_kind` says what the values are
// (`time_ns`, `bytes`, ...); this helper turns that plus the magnitude of the
// loaded values into a `(multiplier, suffix, axisLabel, decimals)` tuple. The
// chart locks that tuple on construction (and again after the lazy `?n=all`
// refetch swaps the payload) so the y-axis stays stable while the user
// pans/zooms; recomputing per-frame would shift the unit out from under them.
// ---------------------------------------------------------------------------

/** A locked display-unit tuple applied to axis ticks and tooltip values. */
export interface DisplayUnit {
  multiplier: number;
  suffix: string;
  axisLabel: string;
  decimals: number;
}

/** The no-scaling fallback for unknown unit kinds. */
export const IDENTITY_UNIT: DisplayUnit = {
  multiplier: 1,
  suffix: '',
  axisLabel: '',
  decimals: 2,
};

/**
 * Median of the finite, nonzero `|v|` in `values`. Zeros and NaNs are not
 * informative for the magnitude pick (a chart with all zeros is not readable
 * anyway), so they are skipped; if every value is filtered out, returns `null`
 * and callers fall back to the kind's smallest display unit.
 */
export function magnitudeReference(values: readonly (number | null | undefined)[]): number | null {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }
  const sample: number[] = [];
  for (const v of values) {
    if (v === null || v === undefined || typeof v !== 'number' || !Number.isFinite(v)) {
      continue;
    }
    const a = Math.abs(v);
    if (a === 0) {
      continue;
    }
    sample.push(a);
  }
  if (sample.length === 0) {
    return null;
  }
  sample.sort((a, b) => a - b);
  const mid = Math.floor(sample.length / 2);
  return sample.length % 2 ? sample[mid] : (sample[mid - 1] + sample[mid]) / 2;
}

/**
 * Concatenate every series' non-null finite values. The picker works off the
 * merged distribution so a chart with one very fast and one very slow series
 * still picks the unit that keeps the larger magnitudes readable.
 */
export function collectAllValues(payload: Pick<ChartResponse, 'series'> | null): number[] {
  const out: number[] = [];
  const series = payload?.series ?? {};
  for (const arr of Object.values(series)) {
    if (!Array.isArray(arr)) {
      continue;
    }
    for (const v of arr) {
      if (v !== null && v !== undefined && Number.isFinite(v)) {
        out.push(v);
      }
    }
  }
  return out;
}

/** Steps: ns to µs (1e3) to ms (1e6) to s (1e9), picked by the median's
 * magnitude so the y-axis tick numbers fit in 1-4 digits. */
function pickTimeUnit(ref: number | null): {
  multiplier: number;
  suffix: string;
  decimals: number;
} {
  if (ref === null || ref < 1e3) {
    return { multiplier: 1, suffix: 'ns', decimals: 0 };
  }
  if (ref < 1e6) {
    return { multiplier: 1e-3, suffix: 'µs', decimals: 2 };
  }
  if (ref < 1e9) {
    return { multiplier: 1e-6, suffix: 'ms', decimals: 2 };
  }
  return { multiplier: 1e-9, suffix: 's', decimals: 2 };
}

/** Binary multiples to match how DuckDB and on-disk file sizes are typically
 * reported. Steps: B, KiB (1024), MiB, GiB, TiB. */
function pickBytesUnit(ref: number | null): {
  multiplier: number;
  suffix: string;
  decimals: number;
} {
  const k = 1024;
  if (ref === null || ref < k) {
    return { multiplier: 1, suffix: 'B', decimals: 0 };
  }
  if (ref < k * k) {
    return { multiplier: 1 / k, suffix: 'KiB', decimals: 2 };
  }
  if (ref < k * k * k) {
    return { multiplier: 1 / (k * k), suffix: 'MiB', decimals: 2 };
  }
  if (ref < k * k * k * k) {
    return { multiplier: 1 / (k * k * k), suffix: 'GiB', decimals: 2 };
  }
  return { multiplier: 1 / (k * k * k * k), suffix: 'TiB', decimals: 2 };
}

/**
 * Pick the display unit for a payload: `unit_kind` selects the family, the
 * magnitude of `values` selects the step within it. Dimensionless kinds
 * (`ratio`, `count`) get no scaling, no suffix, and no axis title, so a
 * "1.2x speedup" axis is not misread via an axis-title-driven label. Unknown
 * kinds (forward-compat with a future server enum) fall back to identity.
 */
export function pickDisplayUnit(
  unitKind: UnitKind | string | undefined,
  values: readonly (number | null | undefined)[],
): DisplayUnit {
  const ref = magnitudeReference(values);
  if (unitKind === 'time_ns') {
    const t = pickTimeUnit(ref);
    return { ...t, axisLabel: `Time (${t.suffix})` };
  }
  if (unitKind === 'bytes') {
    const b = pickBytesUnit(ref);
    return { ...b, axisLabel: `Size (${b.suffix})` };
  }
  if (unitKind === 'throughput_mb_s') {
    return { multiplier: 1, suffix: 'MB/s', axisLabel: 'Throughput (MB/s)', decimals: 2 };
  }
  if (unitKind === 'ratio' || unitKind === 'count') {
    return { multiplier: 1, suffix: '', axisLabel: '', decimals: unitKind === 'count' ? 0 : 2 };
  }
  return IDENTITY_UNIT;
}

/**
 * Tooltip formatter: applies the chart's locked display unit so the tooltip
 * value matches the y-axis tick numbers exactly. Raw `null`/`NaN` collapse to
 * an em dash so a missing data point reads as a clear gap rather than a
 * literal `0`.
 */
export function formatDisplayValue(
  rawValue: number | null | undefined,
  displayUnit: DisplayUnit | null | undefined,
): string {
  if (rawValue === null || rawValue === undefined || Number.isNaN(rawValue)) {
    return '—';
  }
  const u = displayUnit ?? IDENTITY_UNIT;
  const scaled = rawValue * u.multiplier;
  const text = Number.isFinite(scaled) ? scaled.toFixed(u.decimals) : '—';
  return u.suffix ? `${text} ${u.suffix}` : text;
}

// ---------------------------------------------------------------------------
// Throttle.
// ---------------------------------------------------------------------------

/**
 * Throttle `fn` to at most one call per `ms`; the trailing call is preserved so
 * the final slider position is honored. (`requestAnimationFrame` is
 * conceptually similar but this wants a hard ceiling regardless of when the
 * browser schedules a frame.)
 */
export function throttle<A extends unknown[]>(
  fn: (...args: A) => void,
  ms: number,
): (...args: A) => void {
  let lastRan = 0;
  let pending: ReturnType<typeof setTimeout> | null = null;
  let pendingArgs: A;
  return (...args: A) => {
    const now = Date.now();
    pendingArgs = args;
    if (now - lastRan >= ms) {
      lastRan = now;
      fn(...pendingArgs);
    } else if (!pending) {
      const wait = ms - (now - lastRan);
      pending = setTimeout(() => {
        lastRan = Date.now();
        pending = null;
        fn(...pendingArgs);
      }, wait);
    }
  };
}

// ---------------------------------------------------------------------------
// Range-strip window clamp.
// ---------------------------------------------------------------------------

/**
 * Clamp a requested `[rawMin, rawMax]` visible-commit window to the valid index
 * range `[0, maxIdx]`. A minimum span of one commit is enforced ONLY when the
 * chart has more than one commit; on a single-commit chart (`maxIdx === 0`) the
 * minimum span collapses to zero so the window stays pinned at `[0, 0]` rather
 * than extending one slot past the only label. This mirrors the `n <= 1` guards
 * on the strip's drag and pixel-to-index paths, which the bare-track-click path
 * does not apply before requesting a window.
 */
export function clampRangeWindow(
  maxIdx: number,
  rawMin: number,
  rawMax: number,
): { min: number; max: number } {
  const minRange = Math.min(1, maxIdx);
  const min = Math.max(0, Math.min(maxIdx - minRange, rawMin));
  const max = Math.max(min + minRange, Math.min(maxIdx, rawMax));
  return { min, max };
}

// ---------------------------------------------------------------------------
// LTTB (Largest-Triangle-Three-Buckets) downsampler.
// ---------------------------------------------------------------------------

/**
 * Return the indices into `xs` / `ys` to keep, including index 0 and `n - 1`.
 * `xs` must be strictly increasing. When `threshold >= n` or `threshold < 3`,
 * returns `[0, 1, ..., n - 1]` unchanged.
 *
 * Algorithm: <https://skemman.is/handle/1946/15343>. Per bucket, pick the point
 * that forms the largest triangle with the previously kept point and the
 * average of the next bucket.
 */
export function lttbIndices(
  xs: readonly number[],
  ys: readonly number[],
  threshold: number,
): number[] {
  const n = xs.length;
  if (threshold >= n || threshold < 3) {
    const all = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      all[i] = i;
    }
    return all;
  }
  const out = new Array<number>(threshold);
  out[0] = 0;
  const bucket = (n - 2) / (threshold - 2);
  let a = 0;
  for (let bi = 0; bi < threshold - 2; bi++) {
    // Average of the *next* bucket, the "C" point in the triangle.
    const nextStart = Math.floor((bi + 1) * bucket) + 1;
    const nextEnd = Math.min(n, Math.floor((bi + 2) * bucket) + 1);
    const count = Math.max(1, nextEnd - nextStart);
    let ax = 0;
    let ay = 0;
    for (let j = nextStart; j < nextEnd; j++) {
      ax += xs[j];
      ay += ys[j];
    }
    ax /= count;
    ay /= count;

    // Search this bucket for the point with the largest triangle area against
    // `(a, avg_next)`.
    const rangeStart = Math.floor(bi * bucket) + 1;
    const rangeEnd = Math.floor((bi + 1) * bucket) + 1;
    const pax = xs[a];
    const pay = ys[a];
    let maxArea = -1;
    let maxIdx = rangeStart;
    for (let k = rangeStart; k < rangeEnd; k++) {
      const area = Math.abs((pax - ax) * (ys[k] - pay) - (pax - xs[k]) * (ay - pay)) * 0.5;
      if (area > maxArea) {
        maxArea = area;
        maxIdx = k;
      }
    }
    out[bi + 1] = maxIdx;
    a = maxIdx;
  }
  out[threshold - 1] = n - 1;
  return out;
}

// ---------------------------------------------------------------------------
// Payload normalization onto the full-history x-axis.
// ---------------------------------------------------------------------------

/**
 * A [`ChartResponse`] normalized onto the full-history x-axis: `commits` and
 * every series span `history.total_commits` slots, with `null` placeholders for
 * the virtual (not yet loaded) prefix of a bounded window.
 */
export interface NormalizedChartPayload extends Omit<ChartResponse, 'commits'> {
  history: ChartHistory;
  commits: (CommitPoint | null)[];
}

/**
 * Clamp a payload's `history` into a self-consistent placement: `loaded` and
 * `total` are non-negative integers with `loaded <= total`, `start` fits the
 * loaded window inside the total span, and `complete` is honored from the wire
 * or derived from full coverage.
 */
export function canonicalHistory(
  payload: Pick<ChartResponse, 'commits'> & { history?: Partial<ChartHistory> | null },
): ChartHistory {
  const commits = Array.isArray(payload?.commits) ? payload.commits : [];
  const history = payload?.history ?? {};
  let loaded = Number.isFinite(history.loaded_commits)
    ? (history.loaded_commits as number)
    : commits.length;
  let total = Number.isFinite(history.total_commits)
    ? (history.total_commits as number)
    : commits.length;
  let start = Number.isFinite(history.start_index) ? (history.start_index as number) : 0;
  loaded = Math.max(0, Math.floor(loaded));
  total = Math.max(loaded, Math.floor(total));
  start = Math.max(0, Math.min(Math.floor(start), Math.max(0, total - loaded)));
  return {
    total_commits: total,
    start_index: start,
    loaded_commits: loaded,
    complete: history.complete === true || (start === 0 && loaded === total),
  };
}

/**
 * Normalize a chart payload onto the full-history x-axis: a bounded latest-100
 * window is padded with `null` commit slots and `null` series values so the
 * x-axis spans every commit ever ingested, and panning into the virtual prefix
 * can promote the `?n=all` upgrade. Idempotent without v3's `__bench_normalized`
 * marker: an array that already spans `total_commits` slots (a complete payload
 * or a previously normalized bounded one) takes the fast path unchanged, while
 * a raw bounded payload has `commits.length === loaded_commits < total_commits`
 * and gets padded.
 */
export function normalizeChartPayload(payload: ChartResponse): NormalizedChartPayload {
  const commits = Array.isArray(payload.commits) ? payload.commits : [];
  const history = canonicalHistory(payload);
  if (history.total_commits === commits.length) {
    return { ...payload, history, commits };
  }

  const total = history.total_commits;
  const start = history.start_index;
  const normalizedCommits = new Array<CommitPoint | null>(total).fill(null);
  for (let ci = 0; ci < commits.length && start + ci < total; ci++) {
    normalizedCommits[start + ci] = commits[ci];
  }

  const rawSeries = payload.series ?? {};
  const normalizedSeries: Record<string, (number | null)[]> = {};
  for (const [name, values] of Object.entries(rawSeries)) {
    const out = new Array<number | null>(total).fill(null);
    if (Array.isArray(values)) {
      for (let vi = 0; vi < values.length && start + vi < total; vi++) {
        out[start + vi] = values[vi];
      }
    }
    normalizedSeries[name] = out;
  }

  return { ...payload, history, commits: normalizedCommits, series: normalizedSeries };
}

/** Whether `[min, max]` reaches outside the loaded window of an incomplete
 * payload (the signal to promote the `?n=all` upgrade). */
export function rangeTouchesUnloadedHistory(
  payload: Pick<NormalizedChartPayload, 'history'> | null,
  min: number,
  max: number,
): boolean {
  const history = payload?.history;
  if (!history || history.complete) {
    return false;
  }
  const start = history.start_index || 0;
  const end = start + (history.loaded_commits || 0) - 1;
  return Math.floor(min) < start || Math.ceil(max) > end;
}

// ---------------------------------------------------------------------------
// Visible-range math.
// ---------------------------------------------------------------------------

/** A visible x-range in commit-index space; `undefined` bounds mean "let
 * Chart.js use the full axis". */
export interface VisibleRange {
  min: number | undefined;
  max: number | undefined;
}

/**
 * Resolve the visible `[min, max]` for `scope` visible commits out of
 * `commitCount`. Invariant: when `currentRange` is supplied AND the chart is
 * already panned away from the right edge, a scope change preserves the visible
 * CENTER instead of snapping to the most recent N commits. With no
 * `currentRange` (initial render) or a view that already covers everything or
 * sits flush with the newest commit, anchor to the right, which is the right
 * default at first load and after "show all".
 */
export function visibleRange(
  commitCount: number,
  scope: number | 'all',
  currentRange?: { min: number | undefined; max: number | undefined } | null,
): VisibleRange {
  if (commitCount <= 0) {
    return { min: undefined, max: undefined };
  }
  const maxIdx = commitCount - 1;
  if (scope === 'all' || !Number.isFinite(scope) || scope <= 0 || scope >= commitCount) {
    return { min: 0, max: maxIdx };
  }
  const width = scope;
  const rightAnchored = { min: Math.max(0, maxIdx - (width - 1)), max: maxIdx };
  if (!currentRange) {
    return rightAnchored;
  }
  const curMin = Number.isFinite(currentRange.min) ? (currentRange.min as number) : 0;
  const curMax = Number.isFinite(currentRange.max) ? (currentRange.max as number) : maxIdx;
  const coversAll = curMin <= 0 && curMax >= maxIdx;
  // Half-commit tolerance: pan/zoom can leave fractional drift even when the
  // user is effectively still flush with the newest commit.
  const atRightEdge = curMax >= maxIdx - 0.5;
  if (coversAll || atRightEdge) {
    return rightAnchored;
  }
  const center = (curMin + curMax) / 2;
  const halfWidth = (width - 1) / 2;
  let newMin = Math.round(center - halfWidth);
  let newMax = newMin + (width - 1);
  if (newMin < 0) {
    newMin = 0;
    newMax = width - 1;
  } else if (newMax > maxIdx) {
    newMax = maxIdx;
    newMin = maxIdx - (width - 1);
  }
  return { min: newMin, max: newMax };
}

// ---------------------------------------------------------------------------
// Filter state + predicates.
// ---------------------------------------------------------------------------

/**
 * The distinct engines and formats observed across the fact tables; the chip
 * universe of the global filter bar. Collected server-side by
 * `queries.ts::collectFilterUniverse` and passed to the header as a prop.
 */
export interface FilterUniverse {
  engines: string[];
  formats: string[];
}

/**
 * The active (visible) chip sets of the global filter. Every chip active means
 * no filter is applied; the URL `?engine=`/`?format=` allowlists are translated
 * into active sets against the universe on hydration.
 */
export interface GlobalFilterState {
  engines: string[];
  formats: string[];
}

/**
 * Parse one `?engine=` / `?format=` CSV param into a deduplicated, trimmed
 * allowlist, the TypeScript port of the Axum server's `parse_csv`. Empty
 * entries (e.g. trailing commas) are dropped; an absent or entirely empty param
 * means "no filter active" and is encoded as an empty array.
 */
export function parseFilterCsv(raw: string | null | undefined): string[] {
  if (raw === null || raw === undefined) {
    return [];
  }
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of raw.split(',')) {
    const trimmed = part.trim();
    if (trimmed.length === 0 || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    out.push(trimmed);
  }
  return out;
}

/**
 * Translate a URL allowlist into the active chip set. An empty allowlist means
 * "no filter", so every chip in the universe is active. A non-empty allowlist
 * is taken verbatim, even if a chip has since been added or removed from the
 * universe, which keeps stale URLs deterministic.
 */
export function seedActiveFromAllowlist(
  allowlist: readonly string[],
  universe: readonly string[],
): string[] {
  return allowlist.length === 0 ? [...universe] : [...allowlist];
}

/**
 * Whether a series passes the global filter. A series is hidden when its
 * engine/format dimension is filtered (the active set is a strict subset of
 * the universe) AND its tag is not in the active set. Series without an engine
 * tag (e.g. compression-time `format:op` series) are unaffected by the engine
 * filter, symmetric for format, so hiding an engine does not nuke charts that
 * have no engine dimension.
 */
export function seriesPassesFilter(
  meta: SeriesTag | undefined,
  active: GlobalFilterState,
  universe: FilterUniverse,
): boolean {
  const m = meta ?? {};
  if (
    m.engine &&
    active.engines.length < universe.engines.length &&
    !active.engines.includes(m.engine)
  ) {
    return false;
  }
  if (
    m.format &&
    active.formats.length < universe.formats.length &&
    !active.formats.includes(m.format)
  ) {
    return false;
  }
  return true;
}

/** Whether a series label passes a per-group hidden-series filter. */
export function seriesPassesGroupFilter(
  filter: { hiddenSeries: readonly string[] } | null | undefined,
  label: string,
): boolean {
  if (!filter || !filter.hiddenSeries) {
    return true;
  }
  return !filter.hiddenSeries.includes(label);
}
