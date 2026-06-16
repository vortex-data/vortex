// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

'use client';

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import type {
  ActiveElement,
  Chart as ChartJs,
  ChartDataset,
  ChartEvent,
  LegendItem,
  Plugin,
  TooltipModel,
} from 'chart.js';

import {
  CHART_FETCH_N,
  clampRangeWindow,
  collectAllValues,
  colorFor,
  DEFAULT_VISIBLE,
  escapeHtml,
  FETCH_N,
  FETCH_TIMEOUT_MS,
  firstLine,
  formatDisplayValue,
  HOVER_DWELL_MS,
  HOVER_PREFETCH_PRIORITY,
  IDENTITY_UNIT,
  INTERACTION_FULL_PRIORITY,
  labelForCommit,
  LAZY_HYDRATION_ROOT_MARGIN,
  lttbIndices,
  MAX_VISIBLE_POINTS,
  normalizeChartPayload,
  PAN_THROTTLE_MS,
  parsePrNumber,
  pickDisplayUnit,
  predecessorValue,
  rangeTouchesUnloadedHistory,
  seriesPassesFilter,
  seriesPassesGroupFilter,
  shortDate,
  shortSha,
  throttle,
  truncate,
  visibleRange,
  ZOOM_THROTTLE_MS,
  type DisplayUnit,
  type NormalizedChartPayload,
} from '@/lib/chart-format';
import { loadChartJs } from '@/lib/chart-js';
import {
  abortGroupBundle,
  emptyGroupSnapshot,
  ensureGroupBundle,
  fullHistoryQueue,
  getCachedPayload,
  getGlobalFilterSnapshot,
  getGroupSnapshot,
  hydrationQueue,
  noteGroupSeries,
  subscribeGlobalFilter,
  subscribeGroup,
  type QueueEntry,
} from '@/lib/chart-store';
import type { ChartResponse } from '@/lib/queries';

/**
 * The per-chart client island, the React port of the per-card layer of
 * `server/static/chart-init.js`: the card chrome (title, toolbar, tooltip host,
 * canvas, range strip, downsample badge) plus the full interactive behavior.
 *
 * Architecture notes (where the port deviates mechanically from v3; the few
 * DELIBERATE behavioral deviations are documented at their sites, e.g.
 * `applyY`'s pre-construction recording and the permalink page's title row):
 *
 * - v3 planted per-chart state on the canvas node (`canvas.__bench_*`); here
 *   the same fields live in a [`CardState`] owned by a [`ChartController`]
 *   created once per EFFECT MOUNT (torn down in that effect's cleanup, so
 *   React StrictMode's dev replay gets a fresh controller), and the v3 free
 *   functions became its methods. On creation the controller replays the
 *   group store's current state (group Y), since the store outlives mounts.
 * - v3 wired group hydration per group (shard fetches); v4 has no shard route,
 *   so each island lazily fetches its own `/api/chart/{slug}?n=100` through the
 *   shared bounded [`hydrationQueue`] on group open (or pointer intent). The
 *   one-shot `?n=all` upgrade through [`fullHistoryQueue`] is opt-in: it runs
 *   only on per-chart intent (window-chip click, hover dwell, or pan/zoom into
 *   the unloaded region), never as an automatic group-open warmup.
 * - High-frequency mutations (slider value, badge text, range-strip geometry,
 *   tooltip markup, `dataset.data` rebuilds) stay imperative on refs, exactly
 *   as v3 mutated the DOM; React state is reserved for low-frequency bits (the
 *   Y-button highlight, the loading/error indicators).
 * - Group open/close stays native `<details>` behavior; the island listens for
 *   the `toggle` event of its enclosing disclosure (fired for both user and
 *   scripted changes, which is how Expand All reaches the islands without a
 *   shared React tree).
 */

/** A Chart.js line dataset extended with the bench-specific carry-alongs. */
interface BenchDataset extends ChartDataset<'line', (number | null)[]> {
  /** The unmodified payload values; the tooltip reads these regardless of
   * which indices LTTB kept in `data`. */
  rawData: (number | null)[];
  /** Engine/format tag used by the global filter's bulk hide/show. */
  benchMeta: { engine?: string; format?: string };
}

/** The per-island mutable state, the port of the v3 `canvas.__bench_*` contract. */
interface CardState {
  chart: ChartJs | null;
  constructing: boolean;
  payload: NormalizedChartPayload | null;
  ui: { y: 'linear' | 'log'; scope: number | 'all' };
  /** Series labels the user has explicitly legend-toggled on this card. Once
   * set, the global/group filters no longer drive that label here. */
  overrides: Record<string, true>;
  displayUnit: DisplayUnit;
  /** v3 also tracked `__bench_inline_trimmed`; it was write-only there and is
   * dropped here. `payload.history.complete` carries the same information. */
  fullLoaded: boolean;
  initialFetchEntry: QueueEntry | null;
  fullFetchEntry: QueueEntry | null;
  fullFetchPending: Promise<void> | null;
  /** The in-flight `?n=100` fetch's per-fetch aborter; lets a group close or
   * destroy cancel it without aborting the controller-lifetime `aborter`. */
  initialFetchController: AbortController | null;
  /** The in-flight `?n=all` fetch's per-fetch aborter; same role as above. */
  fullFetchController: AbortController | null;
  /** True once this card has rendered a bounded window (so the chip is shown);
   * a chart born complete never sets it and shows no chip. */
  everWindowed: boolean;
  /** The most recent full-history fetch failed; the chip offers a retry. */
  chipError: boolean;
  /** The pointer is currently resting on this card (chip shows the action). */
  hovering: boolean;
  /** Pending hover-dwell prefetch timer; cleared on `pointerleave`/destroy. */
  hoverDwellTimer: ReturnType<typeof setTimeout> | null;
  /** A full-history fetch returned 404: there is nothing beyond the window to
   * load, so the chip stops offering the action and hovers stop re-fetching. */
  fullUnavailable: boolean;
  yUserSet: boolean;
  stripRender: (() => void) | null;
  rebuild: ((chart: ChartJs) => void) | null;
  wheelAttached: boolean;
  disposed: boolean;
}

/** The DOM handles the controller operates on, resolved from refs at call time. */
interface CardElements {
  card: HTMLElement | null;
  canvas: HTMLCanvasElement | null;
  tooltipHost: HTMLDivElement | null;
  slider: HTMLInputElement | null;
  badge: HTMLSpanElement | null;
  chip: HTMLButtonElement | null;
  strip: HTMLDivElement | null;
  stripWindow: HTMLDivElement | null;
}

/** React-state setters the controller drives (low-frequency UI bits only). */
interface CardCallbacks {
  setY: (y: 'linear' | 'log') => void;
  setLoading: (on: boolean) => void;
  setError: (msg: string | null) => void;
  /** Show/hide the initial-fetch retry control in the error region. */
  setRetryable: (on: boolean) => void;
  /** Flip once the Chart.js instance exists, so the pre-data placeholder hides. */
  setConstructed: (on: boolean) => void;
}

// ---------------------------------------------------------------------------
// Crosshair plugin: draws a vertical line at the chart's active hover index.
// An inline plugin is cheaper than chartjs-plugin-crosshair, which is overkill
// for this one feature.
// ---------------------------------------------------------------------------
const crosshairPlugin: Plugin<'line'> = {
  id: 'benchCrosshair',
  afterDatasetsDraw(chart) {
    const active = chart.tooltip?.getActiveElements ? chart.tooltip.getActiveElements() : [];
    if (!active || active.length === 0) {
      return;
    }
    const x = active[0].element.x;
    const ya = chart.scales?.y;
    if (!ya || !Number.isFinite(x)) {
      return;
    }
    const ctx = chart.ctx;
    ctx.save();
    // `--muted` from the page theme, read lazily so dark mode picks up the
    // right color.
    const muted =
      getComputedStyle(document.documentElement).getPropertyValue('--muted').trim() || '#9ca3af';
    ctx.strokeStyle = muted;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, ya.top);
    ctx.lineTo(x, ya.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

/** Read a chart's datasets with the bench carry-alongs visible to the types. */
function benchDatasets(chart: ChartJs): BenchDataset[] {
  return chart.data.datasets as BenchDataset[];
}

/**
 * Build the per-series dataset shells. `data` starts as a full-length
 * null-padded array; `rebuildVisibleAndUpdate` fills it in based on the current
 * visible range. `rawData` holds a reference to the original payload so the
 * tooltip can show raw values regardless of LTTB.
 */
function buildDatasets(payload: NormalizedChartPayload): BenchDataset[] {
  const raw = payload.series ?? {};
  const meta = payload.series_meta ?? {};
  const n = payload.commits.length;
  const global = getGlobalFilterSnapshot();
  return Object.keys(raw)
    .sort()
    .map((name, i) => {
      const seriesMeta = meta[name] ?? {};
      const rawValues = Array.isArray(raw[name]) ? raw[name] : [];
      // `data` starts null-padded; `rebuildVisibleAndUpdate` fills the current
      // visible window with raw or LTTB-kept values. With `spanGaps: true` the
      // line connects across nulls, so a series with partial coverage still
      // draws as a continuous trend; markers only appear at non-null indices.
      const data = new Array<number | null>(n).fill(null);
      return {
        label: name,
        data,
        rawData: rawValues,
        borderColor: colorFor(i),
        backgroundColor: `${colorFor(i)}20`,
        borderWidth: 1.5,
        spanGaps: true,
        tension: 0,
        pointRadius: 2,
        pointHoverRadius: 5,
        pointHitRadius: 8,
        pointStyle: 'cross',
        benchMeta: { engine: seriesMeta.engine, format: seriesMeta.format },
        hidden: !seriesPassesFilter(seriesMeta, global.active, global.universe),
      } satisfies BenchDataset;
    });
}

/**
 * The external tooltip handler factory, ported verbatim from v3 including the
 * flicker fix: the tooltip host is ALWAYS `pointer-events: none` (via CSS); the
 * previous v2 implementation flipped it to `auto` when visible and the cursor
 * would oscillate between canvas and tooltip at event-loop frequency. Clicks on
 * a data point are handled by the chart's `onClick`, so the tooltip itself
 * never needs to be interactive.
 */
function externalTooltipHandler(state: CardState, host: HTMLDivElement | null) {
  return (context: { chart: ChartJs; tooltip: TooltipModel<'line'> }) => {
    const tt = context.tooltip;
    if (!host) {
      return;
    }
    if (tt.opacity === 0) {
      host.style.opacity = '0';
      return;
    }

    const chart = context.chart;
    const firstDp = tt.dataPoints?.[0];
    if (!firstDp) {
      host.style.opacity = '0';
      return;
    }
    // Snap to a single commit: `mode: "nearest"` means `firstDp.dataIndex` is
    // the single closest data point to the cursor (skipping nulls). If the
    // cursor falls between two LTTB-kept points, exactly one wins.
    const idx = firstDp.dataIndex;
    const commit = state.payload?.commits?.[idx] ?? null;
    const displayUnit = state.displayUnit ?? IDENTITY_UNIT;

    // Build one row per dataset from each series' `rawData` so the tooltip
    // shows raw measurements even when LTTB nulled out `dataset.data[idx]`.
    // Iterating `chart.data.datasets` directly (instead of `tt.dataPoints`)
    // guarantees one row per series at this single commit.
    const rowItems = benchDatasets(chart)
      .map((ds, dsIndex) => {
        const meta = chart.getDatasetMeta(dsIndex);
        if (meta?.hidden || ds.hidden) {
          return null;
        }
        const raw = ds.rawData?.[idx];
        if (raw === null || raw === undefined || Number.isNaN(raw)) {
          return null;
        }
        // Per-row delta is `(current - previous) / previous`, where "previous"
        // is the chronologically preceding commit per the BAN-pinned `idx - 1`
        // oldest-first walk in [`predecessorValue`].
        const prevRaw = predecessorValue(ds.rawData ?? [], idx);
        let deltaHtml = '';
        if (prevRaw !== null && prevRaw !== 0) {
          const pct = ((raw - prevRaw) / prevRaw) * 100;
          const cls =
            pct > 0
              ? 'tt-delta tt-delta--worse'
              : pct < 0
                ? 'tt-delta tt-delta--better'
                : 'tt-delta';
          const sign = pct > 0 ? '+' : '';
          deltaHtml = `<span class="${cls}">${sign}${pct.toFixed(1)}%</span>`;
        }
        return { label: ds.label ?? '', color: String(ds.borderColor), raw, deltaHtml };
      })
      .filter((r): r is NonNullable<typeof r> => r !== null);

    // Top-to-bottom order matches the visual stack of lines at this x.
    rowItems.sort((a, b) => b.raw - a.raw);

    const rows = rowItems
      .map(
        (r) =>
          `<div class="tt-row">` +
          `<span class="tt-swatch" style="background:${r.color}"></span>` +
          `<span class="tt-label">${escapeHtml(r.label)}</span>` +
          `<span class="tt-value">${escapeHtml(formatDisplayValue(r.raw, displayUnit))}</span>` +
          r.deltaHtml +
          `</div>`,
      )
      .join('');

    // If every series was hidden or had no value at this commit, treat this as
    // a no-op hover instead of flashing an empty popup.
    if (!rows) {
      host.style.opacity = '0';
      return;
    }

    const titleHtml =
      `<div class="tt-title">` +
      `${escapeHtml(shortSha(commit?.sha))} · ${escapeHtml(shortDate(commit?.timestamp))}` +
      `</div>`;

    // Short SHA + first-line commit message, truncated. The full URL (or PR
    // URL) is wired up via the chart's onClick handler.
    const msg = truncate(firstLine(commit?.message ?? ''), 80);
    const footerLine = commit?.sha
      ? msg
        ? `${escapeHtml(shortSha(commit.sha))} · ${escapeHtml(msg)}`
        : escapeHtml(shortSha(commit.sha))
      : escapeHtml(msg);
    const footerHtml = footerLine
      ? `<div class="tt-footer"><div class="tt-msg">${footerLine}</div></div>`
      : '';

    host.innerHTML = `${titleHtml}<div class="tt-rows">${rows}</div>${footerHtml}`;

    // Position the tooltip relative to its container, offset 12px from the
    // cursor; flip horizontally if it would overflow.
    const canvasRect = context.chart.canvas.getBoundingClientRect();
    const parent = host.parentNode as HTMLElement;
    const hostRect = parent.getBoundingClientRect();
    const x = canvasRect.left - hostRect.left + tt.caretX;
    const y = canvasRect.top - hostRect.top + tt.caretY;
    host.style.opacity = '1';
    host.style.left = `${x}px`;
    host.style.top = `${y}px`;
    // Measure after the content swap so flipping is correct.
    const ttWidth = host.offsetWidth || 0;
    const containerWidth = parent.clientWidth || 0;
    const flip = x + ttWidth + 24 > containerWidth;
    host.style.transform = flip ? 'translate(calc(-100% - 12px), 12px)' : 'translate(12px, 12px)';
  };
}

/**
 * The imperative per-island engine. Created once per MOUNT (not per component
 * instance): React StrictMode runs every dev effect as mount, cleanup, remount,
 * and `destroy()` is one-way, so the mount effect constructs a fresh controller
 * each time it runs and tears the previous one down in its cleanup. Every
 * method is a direct port of the corresponding `chart-init.js` function with
 * the canvas field-stash replaced by [`CardState`]; the DOM listeners the
 * controller attaches (wheel pan, range-strip pointers) are registered with its
 * own abort signal so a teardown removes them from the still-mounted nodes
 * before the next controller re-binds.
 */
class ChartController {
  private readonly state: CardState;
  /** Aborted on [`destroy`]; every controller-attached DOM listener uses it. */
  private readonly aborter = new AbortController();
  /** Failed Chart.js dynamic-import attempts; bounds the error-dismiss retry. */
  private loadAttempts = 0;

  constructor(
    private readonly slug: string,
    private readonly groupSlug: string | undefined,
    private readonly els: () => CardElements,
    private readonly cb: CardCallbacks,
  ) {
    this.state = {
      chart: null,
      constructing: false,
      payload: null,
      ui: { y: 'linear', scope: DEFAULT_VISIBLE },
      overrides: {},
      displayUnit: IDENTITY_UNIT,
      fullLoaded: false,
      initialFetchEntry: null,
      fullFetchEntry: null,
      fullFetchPending: null,
      initialFetchController: null,
      fullFetchController: null,
      everWindowed: false,
      chipError: false,
      hovering: false,
      hoverDwellTimer: null,
      fullUnavailable: false,
      yUserSet: false,
      stripRender: null,
      rebuild: null,
      wheelAttached: false,
      disposed: false,
    };
  }

  /** Seed the permalink page's server-fetched payload before any fetch runs. */
  seedPayload(payload: ChartResponse): void {
    const normalized = normalizeChartPayload(payload);
    this.state.payload = normalized;
    this.state.fullLoaded = normalized.history.complete;
    if (!normalized.history.complete) {
      this.state.everWindowed = true;
    }
    this.syncWindowChip();
    if (this.groupSlug) {
      noteGroupSeries(this.groupSlug, normalized.series_meta);
    }
  }

  /** Whether the enclosing group disclosure (if any) is open. */
  private groupIsOpen(): boolean {
    const card = this.els().card;
    const group = card?.closest('.group-details');
    const details = group?.querySelector('details.group-disclosure');
    return !details || (details as HTMLDetailsElement).open;
  }

  /**
   * Ensure this chart's default `?n=100` payload is loaded. Consults the session
   * payload cache first (a sibling group-bundle fetch may have already cached
   * it), then on the landing page drives one bundle fetch per group, falling
   * back to the per-chart [`fetchInitialPayloadDirect`] only when the bundle does
   * not cover this slug. The permalink page (no group slug) goes straight to the
   * per-chart fetch. `showLoading` mirrors v3: the group-open path shows the
   * per-card loading indicator, the pointer-intent prefetch stays silent.
   */
  ensureInitialPayload(priority: number, showLoading: boolean): Promise<void> {
    const state = this.state;
    if (state.payload || state.disposed) {
      return Promise.resolve();
    }
    // Fast path: a sibling group-bundle fetch may have already cached this
    // chart's default payload. Seed from it synchronously (the same steps as the
    // fetch success path) so no per-chart request is issued.
    const cached = getCachedPayload(this.slug);
    if (cached) {
      this.seedFromCachedPayload(cached);
      return Promise.resolve();
    }
    // On the landing page (a group slug is present), drive one bundle fetch per
    // group and hydrate from it. Only fall through to the per-chart fetch when
    // the bundle is unavailable (404 / failed / this slug missing).
    if (this.groupSlug) {
      if (showLoading) {
        this.cb.setLoading(true);
        this.cb.setRetryable(false);
      }
      const groupSlug = this.groupSlug;
      return ensureGroupBundle(groupSlug, priority).then(() => {
        // The group can close while this card awaits the in-flight bundle. The
        // close runs `abortInFlightFetches` + `abortGroupBundle` already, so a
        // per-chart fallback issued now would never be aborted and would defeat
        // the "closing a group frees server capacity" contract. Bail when the
        // group is no longer open.
        if (state.disposed || state.payload || !this.groupIsOpen()) {
          return;
        }
        const fromBundle = getCachedPayload(this.slug);
        if (fromBundle) {
          this.seedFromCachedPayload(fromBundle);
          return;
        }
        // Bundle did not cover this chart: fall back to the per-chart fetch.
        return this.fetchInitialPayloadDirect(priority, showLoading);
      });
    }
    return this.fetchInitialPayloadDirect(priority, showLoading);
  }

  /** Seed state from a cached default payload (the bundle/cache hit path),
   * mirroring the per-chart fetch's success handler. Does NOT call
   * `maybeConstruct`; the caller does after the returned promise resolves. */
  private seedFromCachedPayload(raw: ChartResponse): void {
    const state = this.state;
    // A concurrent full-history upgrade may already have constructed from the
    // `?n=all` payload; the late cache seed must not clobber that back to the
    // bounded window (same invariant as the per-chart success handler).
    if (state.fullLoaded) {
      return;
    }
    const normalized = normalizeChartPayload(raw);
    state.payload = normalized;
    state.fullLoaded = normalized.history.complete;
    if (!normalized.history.complete) {
      state.everWindowed = true;
    }
    this.syncWindowChip();
    this.cb.setLoading(false);
    if (this.groupSlug) {
      noteGroupSeries(this.groupSlug, normalized.series_meta);
    }
  }

  /**
   * Queue the initial `?n=100` fetch through the bounded hydration queue (or
   * bump its priority if already queued). The per-chart fallback for the bundle
   * path and the only path on the permalink page. `showLoading` mirrors v3: the
   * group-open path shows the per-card loading indicator, the pointer-intent
   * prefetch stays silent.
   */
  private fetchInitialPayloadDirect(priority: number, showLoading: boolean): Promise<void> {
    const state = this.state;
    if (state.initialFetchEntry) {
      if (priority > state.initialFetchEntry.priority) {
        state.initialFetchEntry.priority = priority;
        hydrationQueue.drain();
      }
      if (showLoading) {
        this.cb.setLoading(true);
        this.cb.setRetryable(false);
      }
      return state.initialFetchEntry.promise.then(
        () => undefined,
        () => undefined,
      );
    }
    if (showLoading) {
      this.cb.setLoading(true);
      this.cb.setRetryable(false);
    }
    const url = `/api/chart/${encodeURIComponent(this.slug)}?n=${encodeURIComponent(CHART_FETCH_N)}`;
    const fc = new AbortController();
    state.initialFetchController = fc;
    // Bridge the controller-lifetime aborter to this per-fetch controller so
    // `destroy()` cancels the in-flight request. `{ once: true }` self-removes
    // the listener after a single abort; the `finally` removes it on the no-abort
    // path. Propagate the parent's reason so a destroy reads as `AbortError`.
    const onParentAbort = (): void => fc.abort(this.aborter.signal.reason);
    this.aborter.signal.addEventListener('abort', onParentAbort, { once: true });
    if (this.aborter.signal.aborted) {
      fc.abort(this.aborter.signal.reason);
    }
    const entry = hydrationQueue.schedule(async () => {
      // The timeout starts when the task actually runs (not while queued), so it
      // measures fetch duration, not queue wait. A `TimeoutError` reason lets the
      // catch tell a timeout apart from a close/destroy `AbortError`.
      const timer = setTimeout(
        () => fc.abort(new DOMException('Fetch timed out', 'TimeoutError')),
        FETCH_TIMEOUT_MS,
      );
      try {
        const r = await fetch(url, { headers: { accept: 'application/json' }, signal: fc.signal });
        if (!r.ok) {
          throw new Error(r.status === 404 ? 'not found' : `HTTP ${r.status}`);
        }
        return (await r.json()) as ChartResponse;
      } finally {
        clearTimeout(timer);
        this.aborter.signal.removeEventListener('abort', onParentAbort);
      }
    }, priority);
    state.initialFetchEntry = entry;
    return entry.promise.then(
      (raw) => {
        if (state.initialFetchEntry === entry) {
          state.initialFetchEntry = null;
        }
        if (state.initialFetchController === fc) {
          state.initialFetchController = null;
        }
        if (state.disposed) {
          return;
        }
        // A concurrent full-history upgrade (hover dwell or chip click) may have
        // already resolved and constructed from the `?n=all` payload while this
        // `?n=100` window was still in flight. The late window resolution must
        // not clobber that full payload back to the bounded window (which would
        // diverge `payload` from the rendered datasets, regress the chip, and
        // re-arm a redundant pan-triggered refetch).
        if (state.fullLoaded) {
          return;
        }
        const normalized = normalizeChartPayload(raw as ChartResponse);
        state.payload = normalized;
        state.fullLoaded = normalized.history.complete;
        if (!normalized.history.complete) {
          state.everWindowed = true;
        }
        this.syncWindowChip();
        this.cb.setLoading(false);
        if (this.groupSlug) {
          noteGroupSeries(this.groupSlug, normalized.series_meta);
        }
        void this.maybeConstruct();
      },
      (err: unknown) => {
        if (state.initialFetchEntry === entry) {
          state.initialFetchEntry = null;
        }
        if (state.initialFetchController === fc) {
          state.initialFetchController = null;
        }
        if (state.disposed) {
          return;
        }
        // A close/destroy cancellation aborts with `AbortError`: stay silent and
        // do NOT touch the loading state, which may now belong to a fresh fetch
        // scheduled after a reopen (clearing it here would extinguish that
        // newer fetch's spinner). A timeout (`TimeoutError`) or a genuine
        // network/HTTP failure clears loading and surfaces the error indicator.
        if (err instanceof DOMException && err.name === 'AbortError') {
          return;
        }
        this.cb.setLoading(false);
        const message =
          err instanceof DOMException && err.name === 'TimeoutError'
            ? 'timed out'
            : err instanceof Error
              ? err.message
              : 'unknown error';
        this.cb.setError(`failed to load: ${message}`);
        this.cb.setRetryable(true);
      },
    );
  }

  /** Re-issue the initial `?n=100` fetch after a failure/timeout. User-initiated
   * (the error region's retry control), so it is naturally bounded; clears the
   * error first and schedules at the top of the hydration queue.
   */
  retryInitialPayload(): void {
    if (this.state.disposed || this.state.payload) {
      return;
    }
    this.cb.setError(null);
    this.cb.setRetryable(false);
    void this.ensureInitialPayload(0, true).then(() => {
      if (this.state.disposed) {
        return;
      }
      void this.maybeConstruct();
    });
  }

  /**
   * Hydrate this chart's latest-100 window at `priority` and construct. Full
   * history is NOT warmed here; it loads only on explicit per-chart intent
   * (window-chip click, hover dwell, or pan/zoom into the unloaded region). The
   * priority is the negated visual `index`, so top cards drain ahead of lower
   * ones (the queue drains highest-priority-first).
   */
  onGroupOpen(priority: number): void {
    void this.ensureInitialPayload(priority, true).then(() => {
      if (this.state.disposed) {
        return;
      }
      void this.maybeConstruct();
    });
  }

  /**
   * Queue the one-shot `?n=all` full-history upgrade (or promote the queued
   * entry's priority). Triggered only by explicit intent — window-chip click
   * (`INTERACTION_FULL_PRIORITY`), hover dwell (`HOVER_PREFETCH_PRIORITY`), or
   * pan/zoom touching the unloaded region — never as an automatic warmup.
   */
  ensureFullHistory(priority: number): Promise<void> {
    const state = this.state;
    // `fullUnavailable` is checked here (not only at the chip/hover call sites)
    // so every intent path shares one terminal-404 guard, including the pan/zoom
    // `rangeTouchesUnloadedHistory` promotion, which must not re-issue a fetch
    // that already 404'd.
    if (state.fullLoaded || state.fullUnavailable || state.disposed) {
      return Promise.resolve();
    }
    if (state.fullFetchEntry) {
      if (priority > state.fullFetchEntry.priority) {
        state.fullFetchEntry.priority = priority;
        fullHistoryQueue.drain();
      }
      return state.fullFetchPending ?? Promise.resolve();
    }
    const url = `/api/chart/${encodeURIComponent(this.slug)}?n=${encodeURIComponent(FETCH_N)}`;
    const fc = new AbortController();
    state.fullFetchController = fc;
    const onParentAbort = (): void => fc.abort(this.aborter.signal.reason);
    this.aborter.signal.addEventListener('abort', onParentAbort, { once: true });
    if (this.aborter.signal.aborted) {
      fc.abort(this.aborter.signal.reason);
    }
    const entry = fullHistoryQueue.schedule(async () => {
      const timer = setTimeout(
        () => fc.abort(new DOMException('Fetch timed out', 'TimeoutError')),
        FETCH_TIMEOUT_MS,
      );
      try {
        const r = await fetch(url, { headers: { accept: 'application/json' }, signal: fc.signal });
        if (r.status === 404) {
          return null;
        }
        if (!r.ok) {
          throw new Error(`HTTP ${r.status}`);
        }
        return (await r.json()) as ChartResponse;
      } finally {
        clearTimeout(timer);
        this.aborter.signal.removeEventListener('abort', onParentAbort);
      }
    }, priority);
    state.fullFetchEntry = entry;
    state.fullFetchPending = entry.promise
      .then((full) => {
        if (state.disposed) {
          return;
        }
        if (full === null) {
          state.fullUnavailable = true;
          return;
        }
        this.replaceChartPayload(full as ChartResponse);
        state.fullLoaded = true;
        if (state.fullFetchController === fc) {
          state.fullFetchController = null;
        }
        state.chipError = false;
        this.cb.setLoading(false);
        if (!state.chart && this.groupIsOpen()) {
          void this.maybeConstruct();
        }
      })
      .catch((err: unknown) => {
        if (state.fullFetchController === fc) {
          state.fullFetchController = null;
        }
        // A close/destroy cancellation is silent; a timeout or genuine failure
        // leaves the chip's retry affordance (chipError) so the user can re-try.
        if (err instanceof DOMException && err.name === 'AbortError') {
          return;
        }
        // Quiet: the latest-100 payload is still usable. Surface to the console
        // for debugging; the chip exposes the retry affordance.
        console.warn('bench: full history fetch failed', err);
        state.chipError = true;
      })
      .then(() => {
        if (state.fullFetchEntry === entry) {
          state.fullFetchEntry = null;
          state.fullFetchController = null;
          state.fullFetchPending = null;
        }
        // Always re-sync the chip so the UI reflects the latest state even when
        // the guard above skips the stale entry clears.
        this.syncWindowChip();
      });
    this.syncWindowChip();
    return state.fullFetchPending;
  }

  /**
   * Construct the Chart.js instance when the payload is present and the
   * enclosing group (if any) is open. Loads Chart.js lazily on first need.
   * Idempotent across overlapping calls via the `constructing` latch.
   */
  async maybeConstruct(): Promise<void> {
    const state = this.state;
    if (state.chart || state.constructing || !state.payload || state.disposed) {
      return;
    }
    if (!this.groupIsOpen()) {
      return;
    }
    const { canvas, tooltipHost, card } = this.els();
    if (!canvas || !card) {
      return;
    }
    state.constructing = true;
    try {
      let Chart;
      try {
        Chart = await loadChartJs();
      } catch (err) {
        // A failed chunk load (deploy rotated the hashed assets, flaky
        // network) surfaces like a fetch failure; the loader has already reset
        // its cache, and the error indicator's auto-dismiss retries
        // construction (bounded by `shouldRetryConstruct`) so charts whose
        // only construction trigger already fired (the permalink page's
        // one-shot IntersectionObserver) are not left permanently blank.
        this.loadAttempts += 1;
        if (!state.disposed) {
          const message = err instanceof Error ? err.message : 'unknown error';
          this.cb.setError(`failed to load chart library: ${message}`);
        }
        return;
      }
      if (state.disposed || state.chart || !state.payload) {
        return;
      }
      // Re-check the disclosure AFTER the await: the group can close while the
      // Chart.js chunk loads (a window v3 never had, its library being
      // preloaded), and constructing into the display:none grid would render a
      // zero-size chart. The next toggle-open re-enters via onGroupOpen.
      if (!this.groupIsOpen()) {
        return;
      }
      const payload = state.payload;

      // Lock the display unit for the lifetime of this loaded payload; it is
      // recomputed only when `replaceChartPayload` swaps in the wider window.
      state.displayUnit = pickDisplayUnit(payload.unit_kind, collectAllValues(payload));

      const labels = payload.commits.map(labelForCommit);
      const datasets = buildDatasets(payload);
      const range = visibleRange(labels.length, state.ui.scope);
      const legendPosition = window.matchMedia?.('(max-width: 768px)').matches ? 'top' : 'bottom';

      // Throttled rebuild for pan/zoom: both axes mutate `scales.x.min/max`
      // continuously during interaction, so the rendered points re-derive at
      // most every PAN_THROTTLE_MS and the range strip refreshes in the same
      // call so LTTB and the strip never diverge.
      const throttledRebuild = throttle((chart: ChartJs) => {
        const sx = chart.scales?.x;
        if (!sx) {
          return;
        }
        this.rebuildVisibleAndUpdate(chart, sx.min, sx.max, true);
        state.stripRender?.();
      }, PAN_THROTTLE_MS);

      const state_ = state;
      const chart = new Chart(canvas, {
        type: 'line',
        data: { labels, datasets },
        plugins: [crosshairPlugin],
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          // Snap to the single nearest commit THAT HAS RENDERED DATA. After
          // LTTB most commit indices are null in `dataset.data`; `mode:
          // "index"` would pick null indices (empty tooltip) and `mode: "x"`
          // would pick multiple closely-packed LTTB columns at once (duplicate
          // rows). `intersect: false` keeps the tooltip active anywhere on the
          // chart and, combined with `pointer-events: none` on the host, is
          // also the flicker fix.
          interaction: { mode: 'nearest', intersect: false, axis: 'x' },
          onClick: (event: ChartEvent, _active: ActiveElement[], c: ChartJs) => {
            const native = event.native;
            if (!native) {
              return;
            }
            const points = c.getElementsAtEventForMode(
              native,
              'nearest',
              { intersect: false, axis: 'x' },
              true,
            );
            if (points.length === 0) {
              return;
            }
            const commit = state_.payload?.commits?.[points[0].index];
            if (!commit) {
              return;
            }
            const pr = parsePrNumber(commit.message);
            const url = pr ? `https://github.com/vortex-data/vortex/pull/${pr}` : commit.url;
            if (url) {
              window.open(url, '_blank', 'noopener');
            }
          },
          scales: {
            y: {
              type: state.ui.y === 'log' ? 'logarithmic' : 'linear',
              beginAtZero: state.ui.y !== 'log',
              // The axis title reflects the locked display unit; empty for
              // dimensionless kinds so a "1.2x speedup" chart does not get an
              // arbitrary label.
              title: {
                display: state.displayUnit.axisLabel !== '',
                text: state.displayUnit.axisLabel,
              },
            },
            x: {
              min: range.min,
              max: range.max,
              title: { display: false },
              // One tick per commit is unreadable on a 5000-commit history;
              // Chart.js picks a sensible subset.
              ticks: { maxTicksLimit: 12, autoSkip: true },
            },
          },
          plugins: {
            legend: {
              position: legendPosition,
              // Wrap the default toggle to record the per-card override and
              // keep `dataset.hidden` in sync with the legend's visibility
              // flag; the filter passes write to `dataset.hidden`, so they
              // need to track each other.
              onClick: (_e: ChartEvent, item: LegendItem, legend) => {
                const ci = legend.chart;
                const ds = ci.data.datasets[item.datasetIndex ?? 0];
                const label = ds?.label;
                if (label) {
                  state_.overrides[label] = true;
                }
                const visible = ci.isDatasetVisible(item.datasetIndex ?? 0);
                ci.setDatasetVisibility(item.datasetIndex ?? 0, !visible);
                if (ds) {
                  // Flipped: was visible means now hidden, and vice versa.
                  ds.hidden = visible;
                }
                ci.update();
              },
            },
            tooltip: {
              enabled: false,
              external: externalTooltipHandler(state, tooltipHost),
            },
            // A mouse drag draws a selection rectangle and zooms to it (the v3
            // behavior), never pans. Panning is reserved for the wheel (manual
            // listener below) and the range strip, so a plain drag highlights a
            // region without sliding the axes. Wheel-zoom is disabled because
            // wheel means PAN here.
            zoom: {
              zoom: {
                wheel: { enabled: false },
                drag: { enabled: true, backgroundColor: 'rgba(37, 99, 235, 0.10)' },
                mode: 'x',
                onZoom: (ctx) => throttledRebuild(ctx.chart),
              },
              // Drag-to-pan is disabled so a drag is a zoom selection, not a pan.
              // The plugin's `chart.pan()` API (used by the wheel listener and the
              // range strip) is NOT gated by this flag, so wheel/strip panning is
              // unaffected; only the built-in drag-pan gesture is removed.
              pan: {
                enabled: false,
                mode: 'x',
              },
              limits: {
                x: { min: 0, max: Math.max(0, labels.length - 1), minRange: 4 },
              },
            },
          },
        },
      });

      state.chart = chart;
      this.cb.setConstructed(true);
      state.rebuild = throttledRebuild;
      this.attachWheelPan(canvas, chart, throttledRebuild);
      this.syncSliderBounds(labels.length);
      // Initial render: populate the null data for the initial window, then
      // bind the strip so its first paint reflects the same range.
      this.rebuildVisibleAndUpdate(chart, range.min ?? 0, range.max ?? 0, false);
      this.bindRangeStrip(chart);
      state.stripRender?.();
      // `buildDatasets` seeded `hidden` from the global filter; reapply through
      // the layered helper so a per-group filter set before this card hydrated
      // also takes effect.
      this.applyFilters();
      if (this.groupSlug) {
        noteGroupSeries(this.groupSlug, payload.series_meta);
      }
    } finally {
      state.constructing = false;
    }
  }

  /**
   * The single source of truth for the rendered point count, ported verbatim:
   * build the per-commit max-across-series union over `[rangeMin, rangeMax]`,
   * keep at most [`MAX_VISIBLE_POINTS`] shared commit indices (LTTB above the
   * cap), and write the kept values into every `dataset.data` in place.
   */
  rebuildVisibleAndUpdate(
    chart: ChartJs,
    rangeMin: number,
    rangeMax: number,
    allowFullFetch: boolean,
  ): void {
    const state = this.state;
    // The throttled pan/zoom wrapper preserves its trailing call, which can
    // land after teardown; updating a destroyed Chart.js instance throws.
    if (state.disposed) {
      return;
    }
    const payload = state.payload;
    if (!payload) {
      return;
    }
    const datasets = benchDatasets(chart);
    const n = payload.commits.length;
    if (n === 0) {
      return;
    }

    const min = Math.max(0, Math.floor(rangeMin));
    let max = Math.min(n - 1, Math.ceil(rangeMax));
    if (max < min) {
      max = min;
    }

    // One "virtual series" for LTTB: for each visible commit index, the max
    // non-null value across all datasets. Series in a Vortex chart share unit
    // and scale, so max-across-series picks visually salient peaks. The kept
    // indices are then SHARED across every dataset, which is the cap's only
    // correct interpretation (per-series LTTB picked different peaks per
    // series and the union of x-positions blew past the cap).
    const unionIdxs: number[] = [];
    const unionVals: number[] = [];
    for (let i = min; i <= max; i++) {
      let bestY: number | null = null;
      for (const ds of datasets) {
        const v = ds.rawData?.[i];
        if (v !== null && v !== undefined && !Number.isNaN(v) && (bestY === null || v > bestY)) {
          bestY = v;
        }
      }
      if (bestY !== null) {
        unionIdxs.push(i);
        unionVals.push(bestY);
      }
    }

    const keptSet = new Set<number>();
    let anyDownsampled = false;
    if (unionIdxs.length <= MAX_VISIBLE_POINTS) {
      for (const idx of unionIdxs) {
        keptSet.add(idx);
      }
    } else {
      for (const local of lttbIndices(unionIdxs, unionVals, MAX_VISIBLE_POINTS)) {
        keptSet.add(unionIdxs[local]);
      }
      anyDownsampled = true;
    }

    // Write the kept set into every dataset, scaled by the locked display-unit
    // multiplier (applied here, not on ingest or in SQL, so the wire payload
    // stays in base units). Values outside `[min, max]` stay null: planting
    // off-screen neighbours would blow up the y-axis auto-scale (it reads
    // every non-null value regardless of `scales.x.min/max`).
    const multiplier = state.displayUnit.multiplier;
    for (const ds of datasets) {
      const dsRaw = ds.rawData;
      if (!Array.isArray(dsRaw)) {
        continue;
      }
      let data = ds.data;
      if (!Array.isArray(data) || data.length !== n) {
        data = new Array<number | null>(n);
        ds.data = data;
      }
      for (let z = 0; z < n; z++) {
        data[z] = null;
      }
      for (const idx of keptSet) {
        const val = dsRaw[idx];
        if (val !== null && val !== undefined && !Number.isNaN(val)) {
          data[idx] = val * multiplier;
        }
      }
    }

    const visibleCommits = max - min + 1;
    const keptCommits = keptSet.size;
    chart.update('none');
    this.syncSliderFromRange(visibleCommits);
    this.syncDownsampleBadge(keptCommits, visibleCommits, anyDownsampled);
    // If the user moves into the virtual, not-yet-loaded part of the x-axis,
    // promote this chart's queued full-history fetch ahead of background work.
    if (allowFullFetch && rangeTouchesUnloadedHistory(payload, min, max)) {
      void this.ensureFullHistory(INTERACTION_FULL_PRIORITY);
    }
  }

  /**
   * Swap the chart's labels + datasets to the freshly fetched unbounded payload
   * while preserving the current x-range. The virtual latest-100 payload and
   * the full payload share a full-history x-axis, so the chart does not jump
   * when the real older values arrive.
   */
  replaceChartPayload(rawPayload: ChartResponse): void {
    const state = this.state;
    const payload = normalizeChartPayload(rawPayload);
    state.payload = payload;
    const chart = state.chart;
    if (!chart) {
      return;
    }
    // Re-pick the display unit against the now-wider window: the refetch may
    // surface older commits with a different magnitude, and moving the y-axis
    // once at the refetch boundary beats leaving the chart on a stale unit.
    state.displayUnit = pickDisplayUnit(payload.unit_kind, collectAllValues(payload));
    const yAxis = chart.options.scales?.y;
    if (yAxis && 'title' in yAxis && yAxis.title) {
      yAxis.title.display = state.displayUnit.axisLabel !== '';
      yAxis.title.text = state.displayUnit.axisLabel;
    }
    const newLabels = payload.commits.map(labelForCommit);
    const newDatasets = buildDatasets(payload);
    // Honour any explicit legend toggles the user had made already.
    for (const ds of newDatasets) {
      if (ds.label && state.overrides[ds.label]) {
        const prev = chart.data.datasets.find((p) => p.label === ds.label);
        if (prev) {
          ds.hidden = Boolean(prev.hidden);
        }
      }
    }
    chart.data.labels = newLabels;
    chart.data.datasets = newDatasets;
    this.applyFilters();
    if (this.groupSlug) {
      noteGroupSeries(this.groupSlug, payload.series_meta);
    }
    const newMaxIdx = Math.max(0, newLabels.length - 1);
    const zoomLimits = chart.options.plugins?.zoom?.limits?.x;
    if (zoomLimits) {
      zoomLimits.max = newMaxIdx;
    }
    this.syncSliderBounds(newLabels.length);
    const sx = chart.options.scales?.x;
    if (!sx) {
      return;
    }
    // v3's `replaceChartPayload` carries an unreachable non-finite else branch
    // here; the values are coerced finite above, so only the clamp survives.
    const prevMin = typeof sx.min === 'number' && Number.isFinite(sx.min) ? sx.min : 0;
    const prevMax = typeof sx.max === 'number' && Number.isFinite(sx.max) ? sx.max : 0;
    sx.min = Math.max(0, Math.min(newMaxIdx, prevMin));
    sx.max = Math.max(sx.min, Math.min(newMaxIdx, prevMax));
    this.rebuildVisibleAndUpdate(chart, sx.min, sx.max, false);
    state.stripRender?.();
  }

  /** Mirror the chart's visible commit count onto the toolbar slider; called
   * from every path that changes the visible range. Programmatic value writes
   * do not fire the slider's `input` event, so this never re-enters
   * `applyScope`. */
  private syncSliderFromRange(visibleCommits: number): void {
    const slider = this.els().slider;
    if (!slider) {
      return;
    }
    const lo = parseInt(slider.min, 10) || 1;
    const hi = parseInt(slider.max, 10) || visibleCommits;
    slider.value = String(Math.max(lo, Math.min(hi, visibleCommits)));
  }

  /** Show the badge when the visible range was downsampled. The numbers are
   * commit counts, matching the slider's mental model. */
  private syncDownsampleBadge(
    keptCommits: number,
    visibleCommits: number,
    anyDownsampled: boolean,
  ): void {
    const badge = this.els().badge;
    if (!badge) {
      return;
    }
    if (!anyDownsampled || keptCommits >= visibleCommits) {
      badge.setAttribute('hidden', '');
      badge.textContent = '';
      return;
    }
    badge.removeAttribute('hidden');
    badge.textContent = `downsampled · ${keptCommits} / ${visibleCommits}`;
    badge.setAttribute(
      'title',
      `Showing ${keptCommits} of ${visibleCommits} commits in view. Each series renders at most ` +
        `${MAX_VISIBLE_POINTS} points at a time; when more are in view, we apply LTTB (Largest ` +
        `Triangle, Three Buckets), an algorithm that picks representative points by maximising ` +
        `the area of triangles formed with neighbouring buckets. Visual peaks and valleys are ` +
        `preserved while the chart stays responsive. Zoom in past ${MAX_VISIBLE_POINTS} visible ` +
        `commits to see every raw measurement.`,
    );
  }

  /** Render the per-card window chip from controller state. Imperative, like
   * `syncDownsampleBadge`: the chip is hidden for charts born complete, and
   * otherwise reflects windowed → loading → complete, with an error → retry
   * path and a hover-revealed "load all N" action. */
  private syncWindowChip(): void {
    const chip = this.els().chip;
    if (!chip) {
      return;
    }
    const state = this.state;
    const payload = state.payload;
    if (!payload || !state.everWindowed) {
      chip.setAttribute('hidden', '');
      chip.dataset.state = 'hidden';
      chip.textContent = '';
      chip.disabled = true;
      chip.removeAttribute('title');
      return;
    }
    // Pin the locale so the grouped digits ("3,572") are deterministic across
    // runtimes and CI ICU builds (and match the test expectations) rather than
    // following the host's default locale.
    const total = payload.history.total_commits.toLocaleString('en-US');
    const loaded = payload.history.loaded_commits.toLocaleString('en-US');
    chip.removeAttribute('hidden');
    if (state.fullLoaded) {
      chip.dataset.state = 'complete';
      chip.disabled = true;
      chip.textContent = `all ${total}`;
      chip.removeAttribute('title');
      return;
    }
    if (state.fullUnavailable) {
      chip.dataset.state = 'windowed';
      chip.disabled = true;
      chip.textContent = `latest ${loaded} of ${total}`;
      chip.removeAttribute('title');
      return;
    }
    if (state.fullFetchPending) {
      chip.dataset.state = 'loading';
      chip.disabled = true;
      chip.textContent = `loading all ${total}…`;
      chip.removeAttribute('title');
      return;
    }
    if (state.chipError) {
      chip.dataset.state = 'error';
      chip.disabled = false;
      chip.textContent = 'retry';
      chip.setAttribute('title', 'Loading the full history failed. Click to retry.');
      return;
    }
    chip.dataset.state = 'windowed';
    chip.disabled = false;
    chip.textContent = state.hovering ? `load all ${total}` : `latest ${loaded} of ${total}`;
    chip.setAttribute(
      'title',
      `Showing the latest ${loaded} of ${total} commits. Click to load the full history.`,
    );
  }

  /** Window-chip click: load the full history at top priority, or retry after a
   * failure. A no-op once full history is loaded or a fetch is already pending. */
  onWindowChipClick(): void {
    const state = this.state;
    if (state.disposed || state.fullLoaded || state.fullFetchPending || state.fullUnavailable) {
      return;
    }
    state.chipError = false;
    void this.ensureFullHistory(INTERACTION_FULL_PRIORITY);
  }

  /** Pointer resting on the card: reveal the chip's action immediately and arm
   * the dwell-prefetch timer. Only a deliberate dwell (not a sweep) fetches. */
  onCardHoverStart(): void {
    const state = this.state;
    if (state.disposed) {
      return;
    }
    state.hovering = true;
    this.syncWindowChip();
    if (
      state.fullLoaded ||
      state.fullFetchPending ||
      state.fullUnavailable ||
      state.hoverDwellTimer !== null
    ) {
      return;
    }
    state.hoverDwellTimer = setTimeout(() => {
      state.hoverDwellTimer = null;
      if (state.disposed) {
        return;
      }
      void this.ensureFullHistory(HOVER_PREFETCH_PRIORITY);
    }, HOVER_DWELL_MS);
  }

  /** Pointer left the card: restore the chip label and cancel a pending dwell. */
  onCardHoverEnd(): void {
    const state = this.state;
    if (state.disposed) {
      return;
    }
    state.hovering = false;
    if (state.hoverDwellTimer !== null) {
      clearTimeout(state.hoverDwellTimer);
      state.hoverDwellTimer = null;
    }
    this.syncWindowChip();
  }

  /** Cap the slider's `max` to the chart's full x-axis length; for a virtual
   * latest-100 payload this is intentionally larger than the loaded count so
   * "show all" can expose the unloaded older range while the full-history
   * fetch is warming. */
  private syncSliderBounds(commitCount: number): void {
    const slider = this.els().slider;
    if (!slider) {
      return;
    }
    const max = Math.max(5, commitCount);
    slider.max = String(max);
    // ~200 stops across the slider so dragging feels continuous regardless of
    // history size.
    slider.step = String(Math.max(1, Math.round(max / 200)));
    const current = parseInt(slider.value, 10);
    if (!Number.isFinite(current) || current > max) {
      slider.value = String(Math.min(DEFAULT_VISIBLE, max));
    }
  }

  /** Wheel means horizontal pan (the zoom plugin only offers wheel-zoom), so a
   * manual `wheel` listener translates the dominant delta into `chart.pan`. */
  private attachWheelPan(
    canvas: HTMLCanvasElement,
    chart: ChartJs,
    rebuild: (chart: ChartJs) => void,
  ): void {
    const state = this.state;
    if (state.wheelAttached) {
      return;
    }
    state.wheelAttached = true;
    canvas.addEventListener(
      'wheel',
      (e: WheelEvent) => {
        // Horizontal-wheel-or-shift+wheel pans horizontally; plain vertical
        // wheel also pans so trackpad scroll moves through commit history
        // without modifier keys.
        const dx = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        if (!dx) {
          return;
        }
        e.preventDefault();
        chart.pan({ x: dx * 0.5 }, undefined, 'none');
        rebuild(chart);
      },
      { passive: false, signal: this.aborter.signal },
    );
  }

  /**
   * Bind the range scrollbar strip below the canvas: the highlighted window
   * matches the chart's visible x-range and can be dragged or edge-resized to
   * pan/zoom. Ported verbatim from v3's pointer math.
   */
  private bindRangeStrip(chart: ChartJs): void {
    const { strip, stripWindow } = this.els();
    const state = this.state;
    if (!strip || !stripWindow) {
      return;
    }

    const commitCount = (): number => (chart.data.labels ?? []).length;

    const visibleBounds = (): { min: number; max: number } => {
      const n = commitCount();
      if (n <= 0) {
        return { min: 0, max: 0 };
      }
      const maxIdx = n - 1;
      const sx = chart.options.scales?.x ?? {};
      let min = typeof sx.min === 'number' && Number.isFinite(sx.min) ? sx.min : 0;
      let max = typeof sx.max === 'number' && Number.isFinite(sx.max) ? sx.max : maxIdx;
      min = Math.max(0, Math.min(maxIdx, min));
      max = Math.max(min, Math.min(maxIdx, max));
      return { min, max };
    };

    const render = (): void => {
      const n = commitCount();
      if (n <= 0) {
        stripWindow.style.left = '0%';
        stripWindow.style.width = '100%';
        return;
      }
      const b = visibleBounds();
      const span = Math.max(1, n - 1);
      let leftPct = (b.min / span) * 100;
      let widthPct = ((b.max - b.min) / span) * 100;
      // A minimum visible width keeps the handles grabbable when zoomed in
      // tight on a single commit.
      if (widthPct < 1.5) {
        widthPct = 1.5;
      }
      if (leftPct + widthPct > 100) {
        leftPct = 100 - widthPct;
      }
      stripWindow.style.left = `${leftPct}%`;
      stripWindow.style.width = `${widthPct}%`;
      strip.setAttribute('aria-valuenow', String(Math.round(leftPct)));
    };

    const setRange = (rawMin: number, rawMax: number): void => {
      const n = commitCount();
      if (n <= 0) {
        return;
      }
      const maxIdx = n - 1;
      // Looser than the plugin's `limits.x.minRange = 4`; the strip allows a
      // tighter window. `clampRangeWindow` collapses the minimum span to zero on
      // a single-commit chart so a bare-track click cannot push `x.max` one slot
      // past the only label.
      const { min: newMin, max: newMax } = clampRangeWindow(maxIdx, rawMin, rawMax);
      const sx = chart.options.scales?.x;
      if (!sx) {
        return;
      }
      sx.min = newMin;
      sx.max = newMax;
      // Track scope so the toolbar slider stays consistent on later drags.
      state.ui.scope = Math.round(newMax - newMin + 1);
      this.rebuildVisibleAndUpdate(chart, newMin, newMax, true);
      render();
    };

    const pxToIndex = (px: number, trackWidth: number): number => {
      const n = commitCount();
      if (n <= 1 || trackWidth <= 0) {
        return 0;
      }
      const pct = Math.max(0, Math.min(1, px / trackWidth));
      return pct * (n - 1);
    };

    let dragState: {
      mode: 'pan' | 'resize-left' | 'resize-right';
      rect: DOMRect;
      startX: number;
      startMin: number;
      startMax: number;
      pointerId: number;
    } | null = null;

    const onPointerDown = (e: PointerEvent): void => {
      if (e.button !== undefined && e.button !== 0) {
        return;
      }
      const target = e.target as HTMLElement;
      const role = target.getAttribute?.('data-role');
      const rect = strip.getBoundingClientRect();
      let b = visibleBounds();
      const idxAtCursor = pxToIndex(e.clientX - rect.left, rect.width);

      let mode: 'pan' | 'resize-left' | 'resize-right';
      if (role === 'range-handle-left') {
        mode = 'resize-left';
      } else if (role === 'range-handle-right') {
        mode = 'resize-right';
      } else if (role === 'range-window') {
        mode = 'pan';
      } else {
        // Click on bare track: jump the window so its center lands at the
        // cursor, then begin a pan drag.
        const width = b.max - b.min;
        const newMin = idxAtCursor - width / 2;
        setRange(newMin, newMin + width);
        b = visibleBounds();
        mode = 'pan';
      }
      dragState = {
        mode,
        rect,
        startX: e.clientX,
        startMin: b.min,
        startMax: b.max,
        pointerId: e.pointerId,
      };
      try {
        strip.setPointerCapture(e.pointerId);
      } catch {
        // Pointer capture is best-effort; drag still works without it.
      }
      e.preventDefault();
      strip.classList.add('chart-range-strip--dragging');
    };

    const onPointerMove = (e: PointerEvent): void => {
      if (!dragState) {
        return;
      }
      const n = commitCount();
      if (n <= 1) {
        return;
      }
      const dxIdx = ((e.clientX - dragState.startX) / Math.max(1, dragState.rect.width)) * (n - 1);
      if (dragState.mode === 'pan') {
        setRange(dragState.startMin + dxIdx, dragState.startMax + dxIdx);
      } else if (dragState.mode === 'resize-left') {
        setRange(dragState.startMin + dxIdx, dragState.startMax);
      } else {
        setRange(dragState.startMin, dragState.startMax + dxIdx);
      }
    };

    const onPointerUp = (): void => {
      if (!dragState) {
        return;
      }
      try {
        strip.releasePointerCapture(dragState.pointerId);
      } catch {
        // Capture may already be released; nothing to do.
      }
      dragState = null;
      strip.classList.remove('chart-range-strip--dragging');
    };

    const { signal } = this.aborter;
    strip.addEventListener('pointerdown', onPointerDown, { signal });
    strip.addEventListener('pointermove', onPointerMove, { signal });
    strip.addEventListener('pointerup', onPointerUp, { signal });
    strip.addEventListener('pointercancel', onPointerUp, { signal });

    // Expose the strip's render so the toolbar slider, wheel-pan, and the
    // throttled LTTB rebuild keep the strip in lockstep without knowing strip
    // internals.
    state.stripRender = render;
    render();
  }

  /**
   * Apply a toolbar/strip scope change. Preserves the visible center when the
   * user has panned away from the right edge (see [`visibleRange`]).
   *
   * Like [`applyY`], the target scope is recorded BEFORE the chart-null guard
   * (a deliberate deviation from v3, where the toolbar only bound after
   * construction so pre-construction input was unreachable): a slider drag on
   * a not-yet-hydrated card still selects the window construction renders.
   */
  applyScope(scopeValue: string): void {
    const state = this.state;
    const scope = scopeValue === 'all' ? ('all' as const) : parseInt(scopeValue, 10);
    state.ui.scope = scope;
    const chart = state.chart;
    if (!chart) {
      return;
    }
    const commits = (chart.data.labels ?? []).length;
    const sx = chart.options.scales?.x;
    const currentRange = sx
      ? {
          min: typeof sx.min === 'number' ? sx.min : undefined,
          max: typeof sx.max === 'number' ? sx.max : undefined,
        }
      : null;
    const range = visibleRange(commits, scope, currentRange);
    if (sx) {
      sx.min = range.min;
      sx.max = range.max;
    }
    this.rebuildVisibleAndUpdate(
      chart,
      range.min ?? 0,
      range.max ?? Math.max(0, commits - 1),
      true,
    );
    state.stripRender?.();
  }

  /**
   * Apply a Y-scale change. `userInitiated` toggles the sticky flag: once the
   * user clicks the per-chart Y toolbar, the per-group Y broadcast skips this
   * chart so the local click stays honored.
   *
   * Deliberate deviation from v3: `chart-init.js::applyY` no-ops entirely when
   * the chart has not constructed yet, dropping the click; here the sticky flag
   * and target scale are recorded BEFORE the chart-null guard so a click on a
   * not-yet-hydrated card still applies once the chart constructs.
   */
  applyY(yValue: 'linear' | 'log', userInitiated: boolean): void {
    const state = this.state;
    if (userInitiated) {
      state.yUserSet = true;
    }
    state.ui.y = yValue;
    this.cb.setY(yValue);
    const chart = state.chart;
    if (!chart) {
      return;
    }
    const yAxis = chart.options.scales?.y;
    if (yAxis) {
      yAxis.type = yValue === 'log' ? 'logarithmic' : 'linear';
      if ('beginAtZero' in yAxis) {
        yAxis.beginAtZero = yValue !== 'log';
      }
    }
    chart.update('none');
  }

  /** Whether the per-group Y broadcast should skip this chart. */
  yIsSticky(): boolean {
    return this.state.yUserSet;
  }

  /** Whether the error-dismiss timer should retry construction: a payload is
   * waiting, no chart exists, and the bounded Chart.js import budget (3
   * attempts) is not exhausted. After the budget, the card stays blank until
   * reload, which is v3's behavior for a failed static script. */
  shouldRetryConstruct(): boolean {
    const state = this.state;
    return (
      !state.disposed && state.chart === null && state.payload !== null && this.loadAttempts < 3
    );
  }

  /**
   * Re-evaluate every dataset under the layered filter resolution: per-card
   * legend overrides win, then the per-group hidden-series filter, then the
   * global engine/format filter.
   */
  applyFilters(): void {
    const state = this.state;
    const chart = state.chart;
    if (!chart) {
      return;
    }
    const global = getGlobalFilterSnapshot();
    const group = this.groupSlug ? getGroupSnapshot(this.groupSlug) : emptyGroupSnapshot();
    for (const ds of benchDatasets(chart)) {
      if (ds.label && state.overrides[ds.label]) {
        continue;
      }
      // `dataset.hidden` directly (not `setDatasetVisibility`) so the legend
      // stays in sync; the visibility map is a separate channel.
      ds.hidden =
        !seriesPassesGroupFilter(group, ds.label ?? '') ||
        !seriesPassesFilter(ds.benchMeta, global.active, global.universe);
    }
    chart.update('none');
  }

  /** Cancel any in-flight `?n=100` / `?n=all` request WITHOUT tearing down the
   * controller, so closing a group (or its IO disconnect) frees server capacity
   * and stops open/close from piling requests up. A reopen re-issues the fetch.
   * The aborts reject the in-flight promises with `AbortError`, which the fetch
   * catch paths treat as a silent cancellation. */
  abortInFlightFetches(): void {
    this.state.initialFetchController?.abort();
    this.state.fullFetchController?.abort();
    // Drop the entry references so a reopen schedules a FRESH fetch rather than
    // joining the now-aborting promise (which resolves to nothing and would leave
    // the card blank). The aborted tasks still settle; their handlers' identity-
    // guarded clears below no-op once a newer fetch owns these refs.
    this.state.initialFetchController = null;
    this.state.initialFetchEntry = null;
    this.state.fullFetchController = null;
    this.state.fullFetchEntry = null;
    this.state.fullFetchPending = null;
  }

  /** Tear down this controller: destroy the chart, remove every DOM listener
   * it attached (wheel, strip pointers), and drop late async results. One-way;
   * the mount effect constructs a fresh controller for the next mount. */
  destroy(): void {
    this.state.disposed = true;
    this.aborter.abort(new DOMException('chart controller destroyed', 'AbortError'));
    if (this.state.hoverDwellTimer !== null) {
      clearTimeout(this.state.hoverDwellTimer);
      this.state.hoverDwellTimer = null;
    }
    this.state.chart?.destroy();
    this.state.chart = null;
  }
}

/** Props for one chart island. */
export interface ChartIslandProps {
  /** The chart's payload slug (`/api/chart/{slug}` and the permalink). */
  slug: string;
  /** Display name rendered as the card title. */
  name: string;
  /** Page-unique chart index (v3's `data-chart-index` contract). */
  index: number;
  /** Enclosing group's slug on the landing page; omit on the permalink page. */
  groupSlug?: string;
  /** Server-fetched payload on the permalink page (skips the initial fetch). */
  initialPayload?: ChartResponse;
}

/**
 * One interactive chart card. Renders the full card chrome (the v3
 * `chart_card` markup) and drives Chart.js through a [`ChartController`].
 */
export function Chart({ slug, name, index, groupSlug, initialPayload }: ChartIslandProps) {
  const cardRef = useRef<HTMLElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tooltipHostRef = useRef<HTMLDivElement>(null);
  const sliderRef = useRef<HTMLInputElement>(null);
  const badgeRef = useRef<HTMLSpanElement>(null);
  const chipRef = useRef<HTMLButtonElement>(null);
  const stripRef = useRef<HTMLDivElement>(null);
  const stripWindowRef = useRef<HTMLDivElement>(null);

  const [y, setY] = useState<'linear' | 'log'>('linear');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryable, setRetryable] = useState(false);
  const [constructed, setConstructed] = useState(false);

  // The live controller for the CURRENT mount. Created inside the mount effect
  // (not once per component instance) because `destroy()` is one-way and React
  // StrictMode replays every dev effect as mount, cleanup, remount: a latched
  // controller surviving the replay would refuse all fetch/construct work and
  // every chart would stay blank under `next dev`. Handlers and the store
  // subscriptions reach the current controller through this ref and no-op
  // between teardown and the next mount.
  const controllerRef = useRef<ChartController | null>(null);
  // Cancels a pending deferred dismiss-retry (armed by the error effect below).
  const cleanupRetryRef = useRef<(() => void) | null>(null);

  // Re-apply the layered filter whenever the global filter or this group's
  // state changes (chip clicks anywhere on the page).
  const globalFilter = useSyncExternalStore(
    subscribeGlobalFilter,
    getGlobalFilterSnapshot,
    getGlobalFilterSnapshot,
  );
  // The subscribe callback is memoized on `groupSlug`: a fresh identity per
  // render would make React unsubscribe/re-subscribe the group store on every
  // re-render (loading/error/Y state changes).
  const subscribeToGroup = useCallback(
    (cb: () => void) => (groupSlug ? subscribeGroup(groupSlug, cb) : () => {}),
    [groupSlug],
  );
  const groupState = useSyncExternalStore(
    subscribeToGroup,
    () => (groupSlug ? getGroupSnapshot(groupSlug) : emptyGroupSnapshot()),
    () => emptyGroupSnapshot(),
  );
  useEffect(() => {
    controllerRef.current?.applyFilters();
  }, [globalFilter, groupState.hiddenSeries]);

  // Broadcast the per-group Y override to non-sticky charts; `null` (the
  // resting default and the post-Reset state) reverts to linear.
  useEffect(() => {
    const controller = controllerRef.current;
    if (!groupSlug || !controller || controller.yIsSticky()) {
      return;
    }
    controller.applyY(groupState.groupY ?? 'linear', false);
  }, [groupSlug, groupState.groupY]);

  // Mount wiring: controller construction, payload seeding, the throttled
  // slider listener, group toggle/intent listeners, and (on the permalink
  // page) intersection-based construction.
  useEffect(() => {
    const card = cardRef.current;
    if (!card) {
      return;
    }
    const controller = new ChartController(
      slug,
      groupSlug,
      () => ({
        card: cardRef.current,
        canvas: canvasRef.current,
        tooltipHost: tooltipHostRef.current,
        slider: sliderRef.current,
        badge: badgeRef.current,
        chip: chipRef.current,
        strip: stripRef.current,
        stripWindow: stripWindowRef.current,
      }),
      { setY, setLoading, setError, setRetryable, setConstructed },
    );
    controllerRef.current = controller;
    if (initialPayload) {
      controller.seedPayload(initialPayload);
    }
    // Replay the group store's current Y override: the store outlives mounts
    // (module scope), the group-Y broadcast effect above may have run while no
    // controller existed, and a remounted island would otherwise construct on
    // the default linear scale despite an active group-level `log` override.
    if (groupSlug) {
      const groupY = getGroupSnapshot(groupSlug).groupY;
      if (groupY !== null) {
        controller.applyY(groupY, false);
      }
    }
    const group = card.closest('.group-details');
    const details = group?.querySelector('details.group-disclosure') as HTMLDetailsElement | null;
    const cleanups: (() => void)[] = [];

    const onCardEnter = (): void => controller.onCardHoverStart();
    const onCardLeave = (): void => controller.onCardHoverEnd();
    card.addEventListener('pointerenter', onCardEnter);
    card.addEventListener('pointerleave', onCardLeave);
    cleanups.push(() => {
      card.removeEventListener('pointerenter', onCardEnter);
      card.removeEventListener('pointerleave', onCardLeave);
    });

    // The scope slider binds a THROTTLED native `input` listener (NOT
    // `change`, which only fires on release) so dragging re-renders
    // continuously.
    const slider = sliderRef.current;
    if (slider) {
      const onInput = throttle(() => {
        controller.applyScope(slider.value);
      }, ZOOM_THROTTLE_MS);
      slider.addEventListener('input', onInput);
      cleanups.push(() => slider.removeEventListener('input', onInput));
    }

    if (details) {
      // Landing page: hydrate each card lazily when it scrolls near the viewport
      // (reusing the permalink page's IntersectionObserver shape), so opening a
      // big group hydrates only the ~visible charts, top-first by visual index,
      // and the rest hydrate on scroll. The `toggle` event also fires for
      // scripted `details.open` writes, which is how Expand All reaches every
      // island. Closing the group disconnects the observer and aborts in-flight
      // fetches; reopening re-arms.
      // Top cards drain first: priority `0` for the first card, negative `index`
      // for the rest (the queue sorts highest-priority-first). The `index === 0`
      // branch yields `+0` not `-0`; runtime sorting treats them identically, but
      // the visual-order test asserts `Math.max(...).toBe(0)`, and `toBe`'s
      // `Object.is` check distinguishes `-0` from `0`.
      const priority = index === 0 ? 0 : -index;
      // The bundle kick + close abort only run on the landing page, where the
      // island has a group slug; bind a non-null local so the closures below can
      // pass it without re-narrowing.
      const bundleGroupSlug = groupSlug;
      let io: IntersectionObserver | null = null;
      const armHydration = (): void => {
        if (io) {
          // Already armed; do not double-observe.
          return;
        }
        // Start the group's bundle fetch immediately on open so every chart's
        // last-100 data loads eagerly (top-group-first by index priority), even
        // off-screen. Construction stays gated on intersection below. The
        // per-group in-flight dedupe means sibling islands issue only ONE fetch.
        if (bundleGroupSlug !== undefined) {
          void ensureGroupBundle(bundleGroupSlug, priority);
        }
        if (typeof IntersectionObserver === 'undefined') {
          // Graceful degradation for SSR and legacy browsers that lack
          // `IntersectionObserver`: hydrate immediately rather than never.
          controller.onGroupOpen(priority);
          return;
        }
        io = new IntersectionObserver(
          (entries) => {
            for (const entry of entries) {
              if (entry.isIntersecting) {
                io?.disconnect();
                io = null;
                controller.onGroupOpen(priority);
              }
            }
          },
          { rootMargin: LAZY_HYDRATION_ROOT_MARGIN },
        );
        io.observe(card);
      };
      const disarmHydration = (): void => {
        io?.disconnect();
        io = null;
        controller.abortInFlightFetches();
        // Abort the group bundle and drop its in-flight entry so a reopen
        // re-issues a fresh fetch (idempotent; the first island wins, the rest
        // no-op). Mirrors `abortInFlightFetches`'s entry-clearing rationale.
        if (bundleGroupSlug !== undefined) {
          abortGroupBundle(bundleGroupSlug);
        }
      };
      const onToggle = (): void => {
        if (details.open) {
          armHydration();
        } else {
          disarmHydration();
        }
      };
      details.addEventListener('toggle', onToggle);
      cleanups.push(() => details.removeEventListener('toggle', onToggle));
      cleanups.push(() => {
        io?.disconnect();
        io = null;
      });
      if (details.open) {
        armHydration();
      }
    } else {
      // Permalink page: the payload is inlined; construct lazily when the card
      // scrolls near the viewport (v3's IntersectionObserver behavior).
      if (typeof IntersectionObserver === 'undefined') {
        void controller.maybeConstruct();
      } else {
        const io = new IntersectionObserver(
          (entries) => {
            for (const entry of entries) {
              if (entry.isIntersecting) {
                io.disconnect();
                void controller.maybeConstruct();
              }
            }
          },
          { rootMargin: '150px 0px' },
        );
        io.observe(card);
        cleanups.push(() => io.disconnect());
      }
    }

    return () => {
      for (const cleanup of cleanups) {
        cleanup();
      }
      // Cancel any error-dismiss retry still pending; see the dismiss
      // effect's comment for why the cancel lives here and not there.
      cleanupRetryRef.current?.();
      cleanupRetryRef.current = null;
      controller.destroy();
      if (controllerRef.current === controller) {
        controllerRef.current = null;
      }
      // Re-show the placeholder on the next mount. `constructed` latches true
      // once a chart builds but is never otherwise reset, so a StrictMode dev
      // remount (or any re-mount) would briefly suppress the placeholder until
      // the fresh controller reconstructs.
      setConstructed(false);
    };
    // The island's identity props never change after mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-dismiss the transient fetch-error indicator, matching v3's 4s timer.
  // The dismiss also retries construction when a payload is waiting and the
  // bounded import budget allows (see `shouldRetryConstruct`), covering charts
  // whose only construction trigger already fired before a transient Chart.js
  // chunk-load failure.
  useEffect(() => {
    // A retryable initial-fetch error owns its own dismissal (the user clicks
    // retry), so the 4s construction-retry auto-dismiss does not apply to it.
    if (error === null || retryable) {
      return;
    }
    const timer = setTimeout(() => {
      setError(null);
      // The retry is deferred one macrotask so the null commit flushes first:
      // an INSTANTLY-rejecting import retry (the module map caches evaluation
      // errors) would otherwise re-set the identical message inside the same
      // React batch, the committed value would never transition, and the
      // dismiss effect would not arm a new timer (stuck toast, dead retry).
      const retry = setTimeout(() => {
        const controller = controllerRef.current;
        if (controller?.shouldRetryConstruct()) {
          void controller.maybeConstruct();
        }
      }, 0);
      cleanupRetryRef.current = () => clearTimeout(retry);
    }, 4000);
    // The cleanup deliberately does NOT cancel the pending retry: this
    // cleanup runs on the error -> null transition (the dismiss itself), and
    // in real browsers React flushes that commit's effects ahead of the 0ms
    // macrotask, so cancelling here would kill every retry the dismiss just
    // armed (act/jsdom test environments invert that ordering and mask it).
    // The retry is cancelled only on unmount, in the mount effect's cleanup;
    // the retry closure is unmount-safe regardless via the `controllerRef`
    // null check and `shouldRetryConstruct`'s disposed check.
    return () => {
      clearTimeout(timer);
    };
  }, [error, retryable]);

  return (
    <section className="chart-card" data-chart-index={index} data-chart-slug={slug} ref={cardRef}>
      <h3 className="chart-card-title">
        <a href={`/chart/${slug}`}>{name}</a>
        <span
          className="chart-badge chart-badge--downsampled"
          data-role="downsample-badge"
          hidden
          ref={badgeRef}
        />
        <button
          type="button"
          className="chart-window-chip"
          data-role="window-chip"
          hidden
          ref={chipRef}
          onClick={() => controllerRef.current?.onWindowChipClick()}
        />
      </h3>
      <div className="toolbar toolbar--card" aria-label="Chart controls">
        <div className="toolbar-group" role="group" aria-label="Visible commits">
          <span className="toolbar-label">Show</span>
          {/* `max` and `step` are placeholders; the controller resets them
              after construction so the slider tracks the loaded commit count. */}
          <input
            id={`scope-slider-${index}`}
            className="toolbar-slider"
            type="range"
            min={5}
            max={100}
            step={1}
            defaultValue={100}
            data-role="scope-slider"
            aria-label="Custom commit window"
            ref={sliderRef}
          />
        </div>
        <div className="toolbar-group" role="group" aria-label="Y-axis scale">
          <span className="toolbar-label">Y</span>
          <button
            className={`toolbar-btn${y === 'linear' ? ' toolbar-btn--active' : ''}`}
            type="button"
            data-y="linear"
            onClick={() => controllerRef.current?.applyY('linear', true)}
          >
            linear
          </button>
          <button
            className={`toolbar-btn${y === 'log' ? ' toolbar-btn--active' : ''}`}
            type="button"
            data-y="log"
            onClick={() => controllerRef.current?.applyY('log', true)}
          >
            log
          </button>
        </div>
      </div>
      <div className="chart-tooltip-host" ref={tooltipHostRef} />
      <div className="chart-wrap">
        {!constructed && !error && (
          <div className="chart-placeholder" role="status" aria-live="polite">
            <span className="chart-spinner" aria-hidden="true" />
            <span className="chart-placeholder-text">loading…</span>
          </div>
        )}
        <canvas data-chart-index={index} ref={canvasRef} />
      </div>
      {/* The aria value attributes track the window's left edge as a percent
          of the full history; the strip render keeps aria-valuenow current. */}
      <div
        className="chart-range-strip"
        data-chart-index={index}
        data-role="range-strip"
        aria-label="Visible commit range"
        role="slider"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={100}
        ref={stripRef}
      >
        <div className="chart-range-strip-track">
          <div className="chart-range-strip-window" data-role="range-window" ref={stripWindowRef}>
            <span
              className="chart-range-strip-handle chart-range-strip-handle--left"
              data-role="range-handle-left"
              aria-hidden="true"
            />
            <span
              className="chart-range-strip-handle chart-range-strip-handle--right"
              data-role="range-handle-right"
              aria-hidden="true"
            />
          </div>
        </div>
      </div>
      {loading && (
        <div className="chart-loading" role="status" aria-live="polite">
          <span className="chart-spinner" aria-hidden="true" />
          <span className="chart-loading-text">loading…</span>
        </div>
      )}
      {error && (
        <div className="chart-error">
          <span>{error}</span>
          {retryable && (
            <button
              type="button"
              className="chart-error-retry"
              data-role="fetch-retry"
              onClick={() => {
                const controller = controllerRef.current;
                if (!controller) {
                  return;
                }
                controller.retryInitialPayload();
              }}
            >
              retry
            </button>
          )}
        </div>
      )}
    </section>
  );
}
