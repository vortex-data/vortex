// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment jsdom

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Chart } from '@/components/Chart';
import { FETCH_TIMEOUT_MS } from '@/lib/chart-format';
import { fullHistoryQueue } from '@/lib/chart-store';
import { loadChartJs } from '@/lib/chart-js';

// Mock Chart.js construction to a NEVER-RESOLVING loader by default:
// maybeConstruct awaits it forever and never reaches `new Chart(...)`, so the
// fetch-orchestration path runs to completion without constructing a chart in
// jsdom. Tests that need construction to complete override with mockResolvedValueOnce.
vi.mock('@/lib/chart-js', () => ({
  loadChartJs: vi.fn(() => new Promise(() => {})),
}));

// The payloads flow through the fetch stub as `unknown`, so they need not be
// statically typed as `ChartResponse`; they DO need the correct runtime shape
// (`lib/queries.ts`: ChartResponse has `display_name`/`history`; CommitPoint is
// `{ sha, timestamp, message, url }`) so `normalizeChartPayload` and the chip
// read real values.

/** A latest-100 windowed payload for a chart with `total` commits. */
function windowedPayload(total: number) {
  return {
    display_name: 'tpch q1',
    unit_kind: 'time_ns',
    history: {
      total_commits: total,
      start_index: total - 100,
      loaded_commits: 100,
      complete: false,
    },
    commits: Array.from({ length: 100 }, (_, i) => ({
      sha: `sha${i}`,
      timestamp: `2026-01-01T00:00:${String(i).padStart(2, '0')}Z`,
      message: `commit ${i}`,
      url: `https://github.com/x/y/commit/sha${i}`,
    })),
    series: { 'vortex/nvme': Array.from({ length: 100 }, (_, i) => i + 1) },
    series_meta: { 'vortex/nvme': { engine: 'vortex', format: 'nvme' } },
  };
}

/** A complete payload (born with all its history, fewer than 100 commits). */
function completePayload(total: number) {
  return {
    display_name: 'polarsignals q0',
    unit_kind: 'time_ns',
    history: { total_commits: total, start_index: 0, loaded_commits: total, complete: true },
    commits: Array.from({ length: total }, (_, i) => ({
      sha: `sha${i}`,
      timestamp: `2026-01-01T00:00:${String(i).padStart(2, '0')}Z`,
      message: `commit ${i}`,
      url: `https://github.com/x/y/commit/sha${i}`,
    })),
    series: { 'vortex/nvme': Array.from({ length: total }, (_, i) => i + 1) },
    series_meta: { 'vortex/nvme': { engine: 'vortex', format: 'nvme' } },
  };
}

// The PR-5.0.97 group-bundle path fires a `/api/group/{slug}?n=100` fetch on
// open before the per-chart fetch. These per-CHART resilience/loading tests
// exercise the per-chart path (the bundle's fallback), so the bundle is forced
// to 404 here: every island then falls straight back to its own `/api/chart`
// fetch, leaving each test's per-chart assertions intact.
function isBundleUrl(url: string): boolean {
  return url.includes('/api/group/');
}

const BUNDLE_404 = { ok: false, status: 404 } as unknown as Response;

describe('Chart opt-in full-history loading', () => {
  let container: HTMLElement;
  let root: Root | null = null;
  let fetchCalls: string[];
  // Per-URL-substring responders; default resolves a windowed payload.
  let responders: {
    match: (url: string) => boolean;
    respond: (url: string) => Promise<Response>;
  }[];

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    fetchCalls = [];
    responders = [];
    vi.stubGlobal('fetch', (url: string | URL) => {
      const u = String(url);
      fetchCalls.push(u);
      if (isBundleUrl(u)) {
        return Promise.resolve(BUNDLE_404);
      }
      const r = responders.find((x) => x.match(u));
      if (r) {
        return r.respond(u);
      }
      return Promise.resolve(jsonResponse(windowedPayload(3572)));
    });
    container = document.createElement('div');
    document.body.appendChild(container);
  });

  afterEach(async () => {
    await act(async () => {
      root?.unmount();
    });
    container.remove();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  function jsonResponse(body: unknown): Response {
    return { ok: true, status: 200, json: () => Promise.resolve(body) } as unknown as Response;
  }

  /** Render one Chart island inside an OPEN group disclosure and flush the
   * initial `?n=100` fetch. Returns the chip button (or null). */
  async function renderOpenGroup(slug = 'qm.eyJrIjoidHBjaCJ9'): Promise<HTMLButtonElement | null> {
    container.innerHTML =
      '<section class="group-details">' +
      '<details class="group-disclosure" open><summary class="group-summary">g</summary></details>' +
      '<div class="chart-grid"><div id="mount"></div></div>' +
      '</section>';
    const mount = container.querySelector('#mount') as HTMLElement;
    root = createRoot(mount);
    await act(async () => {
      root?.render(<Chart slug={slug} name="tpch q1" index={0} groupSlug="tpch" />);
    });
    // Let the queued bundle fetch (forced to 404 here), its fall-through to the
    // per-chart fetch, and the normalization microtasks settle. The bundle 404
    // and the per-chart fetch each add a few microtask turns, so flush generously.
    await act(async () => {
      for (let i = 0; i < 6; i++) {
        await Promise.resolve();
      }
    });
    return container.querySelector<HTMLButtonElement>('[data-role="window-chip"]');
  }

  it('opening a group issues the windowed fetch but NO full-history warmup', async () => {
    const scheduleSpy = vi.spyOn(fullHistoryQueue, 'schedule');
    await renderOpenGroup();
    const windowFetches = fetchCalls.filter(
      (u) => u.includes('/api/chart/') && u.includes('n=100'),
    );
    const fullFetches = fetchCalls.filter((u) => u.includes('n=all'));
    expect(windowFetches.length).toBeGreaterThanOrEqual(1);
    expect(fullFetches).toHaveLength(0);
    expect(scheduleSpy).not.toHaveBeenCalled();
  });

  it('shows the window chip "latest 100 of 3,572" for a windowed chart', async () => {
    const chip = await renderOpenGroup();
    expect(chip).not.toBeNull();
    expect(chip?.hasAttribute('hidden')).toBe(false);
    expect(chip?.dataset.state).toBe('windowed');
    expect(chip?.textContent).toBe('latest 100 of 3,572');
  });

  it('hides the chip for a chart born with its complete history', async () => {
    responders.push({
      match: (u) => u.includes('n=100'),
      respond: () => Promise.resolve(jsonResponse(completePayload(40))),
    });
    const chip = await renderOpenGroup();
    expect(chip?.hasAttribute('hidden')).toBe(true);
  });

  it('chip click loads full history at top priority and reaches "all N"', async () => {
    let resolveFull: (r: Response) => void = () => {};
    responders.push({
      match: (u) => u.includes('n=all'),
      respond: () =>
        new Promise<Response>((res) => {
          resolveFull = res;
        }),
    });
    const chip = await renderOpenGroup();
    await act(async () => {
      chip?.click();
      await Promise.resolve();
    });
    expect(fetchCalls.some((u) => u.includes('n=all'))).toBe(true);
    expect(chip?.dataset.state).toBe('loading');
    await act(async () => {
      resolveFull(jsonResponse(completePayload(3572)));
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(chip?.dataset.state).toBe('complete');
    expect(chip?.textContent).toBe('all 3,572');
  });

  it('a failed full fetch surfaces a retry affordance', async () => {
    let rejectFull: (e: unknown) => void = () => {};
    responders.push({
      match: (u) => u.includes('n=all'),
      respond: () =>
        new Promise<Response>((_, rej) => {
          rejectFull = rej;
        }),
    });
    const chip = await renderOpenGroup();
    await act(async () => {
      chip?.click();
      await Promise.resolve();
    });
    await act(async () => {
      rejectFull(new Error('boom'));
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(chip?.dataset.state).toBe('error');
    expect(chip?.textContent).toBe('retry');
    expect(chip?.disabled).toBe(false);
  });

  it('a deliberate dwell prefetches full history; a brief hover does not', async () => {
    await renderOpenGroup();
    const card = container.querySelector('.chart-card') as HTMLElement;
    vi.useFakeTimers();
    card.dispatchEvent(new Event('pointerenter'));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(599);
    });
    expect(fetchCalls.some((u) => u.includes('n=all'))).toBe(false);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2);
    });
    expect(fetchCalls.some((u) => u.includes('n=all'))).toBe(true);
  });

  it('pointerleave before the dwell threshold cancels the prefetch', async () => {
    await renderOpenGroup();
    const card = container.querySelector('.chart-card') as HTMLElement;
    vi.useFakeTimers();
    card.dispatchEvent(new Event('pointerenter'));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    card.dispatchEvent(new Event('pointerleave'));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600);
    });
    expect(fetchCalls.some((u) => u.includes('n=all'))).toBe(false);
  });

  it('a 404 on full history is terminal: the chip stops offering the action', async () => {
    responders.push({
      match: (u) => u.includes('n=all'),
      respond: () =>
        Promise.resolve({
          ok: false,
          status: 404,
          json: () => Promise.resolve(null),
        } as unknown as Response),
    });
    const chip = await renderOpenGroup();
    await act(async () => {
      chip?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(chip?.dataset.state).toBe('windowed');
    expect(chip?.disabled).toBe(true);
    expect(chip?.textContent).toBe('latest 100 of 3,572');
    const before = fetchCalls.filter((u) => u.includes('n=all')).length;
    const card = container.querySelector('.chart-card') as HTMLElement;
    vi.useFakeTimers();
    card.dispatchEvent(new Event('pointerenter'));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });
    expect(fetchCalls.filter((u) => u.includes('n=all')).length).toBe(before);
  });

  it('hover reveals the "load all N" action without fetching', async () => {
    const chip = await renderOpenGroup();
    const card = container.querySelector('.chart-card') as HTMLElement;
    card.dispatchEvent(new Event('pointerenter'));
    expect(chip?.textContent).toBe('load all 3,572');
    expect(fetchCalls.some((u) => u.includes('n=all'))).toBe(false);
    card.dispatchEvent(new Event('pointerleave'));
    expect(chip?.textContent).toBe('latest 100 of 3,572');
  });

  it('a full-history upgrade that lands before the pending initial fetch is not clobbered', async () => {
    // Park the initial `?n=100` so it stays in flight while the dwell-triggered
    // `?n=all` upgrade resolves first. The full payload is born complete, so the
    // chip is hidden (no windowed state was ever observed). The late `?n=100`
    // resolution must NOT clobber that full payload: with the resolver guard it
    // stays hidden/complete; without it the late window flips the chart back to
    // the bounded window and re-reveals the chip as 'windowed' — the regression.
    let resolveWindow: (r: Response) => void = () => {};
    responders.push({
      match: (u) => u.includes('n=100'),
      respond: () =>
        new Promise<Response>((res) => {
          resolveWindow = res;
        }),
    });
    responders.push({
      match: (u) => u.includes('n=all'),
      respond: () => Promise.resolve(jsonResponse(completePayload(3572))),
    });
    // The parked `?n=100` leaves the chip hidden (no payload yet).
    const chip = await renderOpenGroup();
    const card = container.querySelector('.chart-card') as HTMLElement;
    vi.useFakeTimers();
    card.dispatchEvent(new Event('pointerenter'));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(601);
    });
    vi.useRealTimers();
    // The `?n=all` upgrade drains through the full-history queue across several
    // microtask hops (queue drain, fetch, json, replaceChartPayload); flush
    // generously so it lands before the still-parked `?n=100`.
    await act(async () => {
      for (let i = 0; i < 10; i += 1) {
        await Promise.resolve();
      }
    });
    // The full payload loaded; born complete, its chip is hidden.
    expect(chip?.hasAttribute('hidden')).toBe(true);
    // Now the late initial `?n=100` resolves; the resolver must early-return on
    // `fullLoaded` and leave the full payload (and the hidden chip) intact.
    await act(async () => {
      resolveWindow(jsonResponse(windowedPayload(3572)));
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(chip?.hasAttribute('hidden')).toBe(true);
    expect(chip?.dataset.state).not.toBe('windowed');
  });

  describe('PR-5.0.95 fetch resilience: timeout + abort', () => {
    // A fetch stub whose `?n=100` response never resolves on its own but DOES
    // reject when its AbortSignal fires, so timeout/destroy aborts are observable.
    // Returns the captured signals for assertions.
    function stubAbortableWindowFetch(): { signals: AbortSignal[] } {
      const signals: AbortSignal[] = [];
      vi.stubGlobal('fetch', (url: string | URL, init?: { signal?: AbortSignal }) => {
        const u = String(url);
        fetchCalls.push(u);
        if (isBundleUrl(u)) {
          return Promise.resolve(BUNDLE_404);
        }
        const signal = init?.signal;
        if (signal) {
          signals.push(signal);
        }
        return new Promise<Response>((_resolve, reject) => {
          signal?.addEventListener('abort', () => {
            reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
          });
        });
      });
      return { signals };
    }

    it('passes an AbortSignal into the window fetch', async () => {
      const { signals } = stubAbortableWindowFetch();
      await renderOpenGroup();
      expect(signals.length).toBeGreaterThanOrEqual(1);
      expect(signals[0].aborted).toBe(false);
    });

    it('aborts a stalled window fetch at FETCH_TIMEOUT_MS and shows an error', async () => {
      vi.useFakeTimers();
      const { signals } = stubAbortableWindowFetch();
      await renderOpenGroup();
      expect(signals[0].aborted).toBe(false);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(FETCH_TIMEOUT_MS);
      });
      expect(signals[0].aborted).toBe(true);
      const err = container.querySelector('.chart-error');
      expect(err).not.toBeNull();
      vi.useRealTimers();
    });

    it('unmount (destroy) aborts an in-flight window fetch', async () => {
      const { signals } = stubAbortableWindowFetch();
      await renderOpenGroup();
      expect(signals[0].aborted).toBe(false);
      await act(async () => {
        root?.unmount();
        root = null;
      });
      expect(signals[0].aborted).toBe(true);
    });

    it('a close/destroy abort is silent (no error indicator)', async () => {
      const { signals } = stubAbortableWindowFetch();
      await renderOpenGroup();
      await act(async () => {
        root?.unmount();
        root = null;
      });
      expect(signals[0].aborted).toBe(true);
      // destroy sets disposed, so the rejected fetch must not paint an error.
      expect(container.querySelector('.chart-error')).toBeNull();
    });

    // A fetch stub that resolves `?n=100` normally (so the chip appears) but
    // returns a never-resolving, signal-honoring promise for `?n=all`.
    // Returns the signals captured per request so the `?n=all` signal can be
    // asserted.
    function stubAbortableFullFetch(): { signals: AbortSignal[] } {
      const signals: AbortSignal[] = [];
      vi.stubGlobal('fetch', (url: string | URL, init?: { signal?: AbortSignal }) => {
        const u = String(url);
        fetchCalls.push(u);
        if (isBundleUrl(u)) {
          return Promise.resolve(BUNDLE_404);
        }
        if (u.includes('n=all')) {
          const signal = init?.signal;
          if (signal) {
            signals.push(signal);
          }
          return new Promise<Response>((_resolve, reject) => {
            signal?.addEventListener('abort', () => {
              reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
            });
          });
        }
        // Resolve the `?n=100` window fetch normally so the chip renders.
        return Promise.resolve(jsonResponse(windowedPayload(3572)));
      });
      return { signals };
    }

    it('destroy aborts an in-flight full-history fetch', async () => {
      const { signals } = stubAbortableFullFetch();
      const chip = await renderOpenGroup();
      // Trigger the `?n=all` fetch via chip click, then flush microtasks so
      // the fetch promise is created and the signal is captured.
      await act(async () => {
        chip?.click();
        await Promise.resolve();
      });
      expect(signals.length).toBeGreaterThanOrEqual(1);
      expect(signals[0].aborted).toBe(false);
      // Unmounting destroys the controller; the per-fetch aborter is bridged
      // from the controller-lifetime aborter so it must fire immediately.
      await act(async () => {
        root?.unmount();
        root = null;
      });
      expect(signals[0].aborted).toBe(true);
    });

    it('a stalled full-history fetch times out and the chip offers retry', async () => {
      // Use real timers for the initial render so `?n=100` resolves and the
      // chip appears, then switch to fake timers before the chip click so the
      // `?n=all` timeout `setTimeout` is fully controlled.
      const { signals } = stubAbortableFullFetch();
      const chip = await renderOpenGroup();
      expect(chip?.dataset.state).toBe('windowed');
      vi.useFakeTimers();
      // Click the chip to start the `?n=all` fetch; the 30s timeout is now
      // governed by fake timers.
      await act(async () => {
        chip?.click();
        await Promise.resolve();
      });
      expect(chip?.dataset.state).toBe('loading');
      // Advance past `FETCH_TIMEOUT_MS`; the timer fires, the per-fetch
      // controller aborts the fetch, and `chipError` flips to `true`.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(FETCH_TIMEOUT_MS);
      });
      expect(signals[0].aborted).toBe(true);
      expect(chip?.dataset.state).toBe('error');
      expect(chip?.textContent).toBe('retry');
      vi.useRealTimers();
    });
  });

  describe('PR-5.0.95 loading spinner', () => {
    it('renders an animated spinner element (not bare text) while loading', async () => {
      // A never-resolving window fetch keeps the card in the loading state.
      vi.stubGlobal('fetch', (url: string | URL) => {
        fetchCalls.push(String(url));
        return new Promise<Response>(() => {});
      });
      await renderOpenGroup();
      const loading = container.querySelector('.chart-loading');
      expect(loading).not.toBeNull();
      expect(loading?.querySelector('.chart-spinner')).not.toBeNull();
    });
  });

  describe('PR-5.0.97 pre-data placeholder', () => {
    it('shows a .chart-placeholder with role="status" before any fetch resolves', async () => {
      // A never-resolving fetch keeps the card permanently pre-constructed, so
      // the placeholder must be visible from the first paint through to cleanup.
      vi.stubGlobal('fetch', (url: string | URL) => {
        fetchCalls.push(String(url));
        return new Promise<Response>(() => {});
      });
      await renderOpenGroup();
      const placeholder = container.querySelector('.chart-placeholder');
      expect(placeholder).not.toBeNull();
      expect(placeholder?.getAttribute('role')).toBe('status');
      expect(placeholder?.querySelector('.chart-spinner')).not.toBeNull();
      expect(placeholder?.querySelector('.chart-placeholder-text')).not.toBeNull();
    });

    it('shows the .chart-placeholder while the initial fetch is pending', async () => {
      // The default fetch stub resolves, but the chart never constructs because
      // loadChartJs is mocked to a never-resolving loader. The placeholder must
      // therefore persist after the fetch resolves (the chart was never built).
      await renderOpenGroup();
      const placeholder = container.querySelector('.chart-placeholder');
      expect(placeholder).not.toBeNull();
    });

    it('does NOT show the .chart-placeholder alongside the error block', async () => {
      // When an error fires the .chart-error block is shown; the placeholder
      // must not appear at the same time (it is suppressed by the !error guard).
      vi.stubGlobal('fetch', (url: string | URL) => {
        const u = String(url);
        fetchCalls.push(u);
        if (isBundleUrl(u)) {
          return Promise.resolve(BUNDLE_404);
        }
        if (u.includes('n=100')) {
          return Promise.resolve({ ok: false, status: 500 } as unknown as Response);
        }
        return Promise.resolve(jsonResponse(windowedPayload(3572)));
      });
      await renderOpenGroup();
      await act(async () => {
        for (let i = 0; i < 6; i++) {
          await Promise.resolve();
        }
      });
      const errorEl = container.querySelector('.chart-error');
      expect(errorEl).not.toBeNull();
      expect(container.querySelector('.chart-placeholder')).toBeNull();
    });

    it('removes .chart-placeholder once the chart successfully constructs', async () => {
      // Override the default never-resolving loader with a stub that resolves to
      // a minimal Chart constructor, letting maybeConstruct run to completion and
      // call setConstructed(true). The stub stores the config's labels/datasets so
      // the post-construction helpers (rebuildVisibleAndUpdate, bindRangeStrip,
      // applyFilters) can read chart.data and chart.options without throwing.
      let stubConstructed = false;
      class StubChart {
        data: { labels: unknown[]; datasets: unknown[] };
        options: Record<string, unknown>;
        constructor(
          _canvas: HTMLCanvasElement,
          config: {
            data: { labels: unknown[]; datasets: unknown[] };
            options: Record<string, unknown>;
          },
        ) {
          stubConstructed = true;
          this.data = { labels: config.data.labels ?? [], datasets: config.data.datasets ?? [] };
          this.options = config.options ?? {};
        }
        update(): void {}
        destroy(): void {}
      }
      vi.mocked(loadChartJs).mockResolvedValueOnce(StubChart as never);

      await renderOpenGroup();
      // renderOpenGroup already flushed 6 microtasks for the bundle-404 and the
      // per-chart fetch. Construction (loadChartJs await + new StubChart + React
      // state commit) adds several more microtask hops; flush generously so
      // setConstructed(true) and the React re-render complete before asserting.
      await act(async () => {
        for (let i = 0; i < 10; i++) {
          await Promise.resolve();
        }
      });
      expect(container.querySelector('.chart-placeholder')).toBeNull();
      // Distinguish the construction path from the error path: both remove the
      // placeholder, but only construction leaves .chart-error absent and sets
      // stubConstructed. These two assertions make the test non-tautological.
      expect(container.querySelector('.chart-error')).toBeNull();
      expect(stubConstructed).toBe(true);
    });
  });

  describe('PR-5.0.95 initial-fetch retry', () => {
    // Overrides the `beforeEach` default `fetch` stub so the `?n=100` fetch can
    // be rejected on demand; `afterEach`'s `vi.unstubAllGlobals()` still cleans
    // both up.
    function stubControllableWindowFetch(): {
      rejectNext: (e: unknown) => void;
      calls: () => number;
    } {
      let rejecter: (e: unknown) => void = () => {};
      vi.stubGlobal('fetch', (url: string | URL) => {
        const u = String(url);
        fetchCalls.push(u);
        if (isBundleUrl(u)) {
          return Promise.resolve(BUNDLE_404);
        }
        if (u.includes('n=100')) {
          return new Promise<Response>((_res, rej) => {
            rejecter = rej;
          });
        }
        return Promise.resolve(jsonResponse(windowedPayload(3572)));
      });
      return {
        rejectNext: (e) => rejecter(e),
        calls: () =>
          fetchCalls.filter((u) => u.includes('/api/chart/') && u.includes('n=100')).length,
      };
    }

    it('a failed initial fetch surfaces a clickable retry that re-issues the fetch', async () => {
      const ctl = stubControllableWindowFetch();
      await renderOpenGroup();
      expect(ctl.calls()).toBe(1);
      await act(async () => {
        ctl.rejectNext(new Error('boom'));
        await Promise.resolve();
        await Promise.resolve();
      });
      const retry = container.querySelector<HTMLButtonElement>(
        '.chart-error [data-role="fetch-retry"]',
      );
      expect(retry).not.toBeNull();
      await act(async () => {
        retry?.click();
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(ctl.calls()).toBe(2);
    });
  });
});
