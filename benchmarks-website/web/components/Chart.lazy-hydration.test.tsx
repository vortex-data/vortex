// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment jsdom

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Chart } from '@/components/Chart';
import { bundleQueue, hydrationQueue, resetPayloadCache } from '@/lib/chart-store';

vi.mock('@/lib/chart-js', () => ({
  loadChartJs: () => new Promise(() => {}),
}));

// jsdom has no IntersectionObserver. This mock records each instance and its
// observed elements, and exposes `fire()` to simulate a card scrolling into
// view. One instance is created per chart card (the mount effect arms one IO
// per card), in card-registration (DOM/visual) order.
class MockIO {
  static instances: MockIO[] = [];
  callback: IntersectionObserverCallback;
  elements: Element[] = [];
  disconnected = false;
  constructor(cb: IntersectionObserverCallback) {
    this.callback = cb;
    MockIO.instances.push(this);
  }
  observe(el: Element): void {
    this.elements.push(el);
  }
  unobserve(el: Element): void {
    this.elements = this.elements.filter((e) => e !== el);
  }
  disconnect(): void {
    this.disconnected = true;
    this.elements = [];
  }
  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }
  /** Simulate this card scrolling into (or out of) view. */
  fire(isIntersecting = true): void {
    this.callback(
      this.elements.map(
        (el) => ({ target: el, isIntersecting }) as unknown as IntersectionObserverEntry,
      ),
      this as unknown as IntersectionObserver,
    );
  }
}

function windowedPayload(total: number) {
  return {
    display_name: 'q',
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
      message: `c${i}`,
      url: `https://github.com/x/y/commit/sha${i}`,
    })),
    series: { 'vortex/nvme': Array.from({ length: 100 }, (_, i) => i + 1) },
    series_meta: { 'vortex/nvme': { engine: 'vortex', format: 'nvme' } },
  };
}

function jsonResponse(body: unknown): Response {
  return { ok: true, status: 200, json: () => Promise.resolve(body) } as unknown as Response;
}

describe('PR-5.0.95 landing-page lazy hydration', () => {
  let container: HTMLElement;
  let root: Root | null = null;
  let fetchCalls: string[];

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    fetchCalls = [];
    MockIO.instances = [];
    vi.stubGlobal('IntersectionObserver', MockIO);
    vi.stubGlobal('fetch', (url: string | URL) => {
      fetchCalls.push(String(url));
      return Promise.resolve(jsonResponse(windowedPayload(3572)));
    });
    container = document.createElement('div');
    document.body.appendChild(container);
  });

  afterEach(async () => {
    await act(async () => {
      root?.unmount();
    });
    root = null;
    container.remove();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  /** Render `n` charts (index 0..n-1) inside one OPEN group disclosure. */
  async function renderGroup(n: number): Promise<void> {
    const mounts = Array.from({ length: n }, (_, i) => `<div id="m${i}"></div>`).join('');
    container.innerHTML =
      '<section class="group-details">' +
      '<details class="group-disclosure" open><summary class="group-summary">g</summary></details>' +
      `<div class="chart-grid">${mounts}</div>` +
      '</section>';
    // Render each island into its OWN root so each gets its own mount effect.
    const roots: Root[] = [];
    await act(async () => {
      for (let i = 0; i < n; i++) {
        const r = createRoot(container.querySelector(`#m${i}`) as HTMLElement);
        roots.push(r);
        r.render(<Chart slug={`s${i}`} name={`q${i}`} index={i} groupSlug="g" />);
      }
    });
    // A teardown handle the shared afterEach can unmount (every island at once).
    root = {
      unmount: () => roots.forEach((r) => r.unmount()),
      render: () => {},
    } as unknown as Root;
    await act(async () => {
      await Promise.resolve();
    });
  }

  function windowFetchCount(): number {
    return fetchCalls.filter((u) => u.includes('/api/chart/') && u.includes('n=100')).length;
  }

  it('opening a group schedules NO fetch until a card intersects', async () => {
    await renderGroup(5);
    expect(MockIO.instances.length).toBe(5);
    expect(windowFetchCount()).toBe(0);
  });

  it('only intersecting (in-viewport) cards hydrate on group open', async () => {
    await renderGroup(5);
    await act(async () => {
      MockIO.instances[0].fire(true);
      MockIO.instances[1].fire(true);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(windowFetchCount()).toBe(2);
  });

  it('an off-viewport card hydrates only once its observer fires', async () => {
    await renderGroup(5);
    expect(windowFetchCount()).toBe(0);
    await act(async () => {
      MockIO.instances[4].fire(true);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(windowFetchCount()).toBe(1);
  });

  it('schedules top cards at a higher priority than lower cards (visual order)', async () => {
    const scheduleSpy = vi.spyOn(hydrationQueue, 'schedule');
    await renderGroup(5);
    await act(async () => {
      MockIO.instances[0].fire(true);
      MockIO.instances[3].fire(true);
      await Promise.resolve();
    });
    const priorities = scheduleSpy.mock.calls.map((c) => c[1]);
    // index 0 => priority 0; index 3 => priority -3; higher drains first.
    expect(priorities).toContain(0);
    expect(priorities).toContain(-3);
    expect(Math.max(...(priorities as number[]))).toBe(0);
  });

  it('does NOT bulk-prefetch every card when the group summary is hovered', async () => {
    await renderGroup(5);
    const summary = container.querySelector('.group-summary') as HTMLElement;
    await act(async () => {
      summary.dispatchEvent(new Event('pointerenter'));
      await Promise.resolve();
    });
    // No summary-hover bulk prefetch: hovering the summary schedules nothing.
    expect(windowFetchCount()).toBe(0);
  });

  it('reopening the group re-arms fresh observers and hydrates previously-unseen cards', async () => {
    await renderGroup(2);
    // Fire card 0's initial observer so it hydrates before close.
    await act(async () => {
      MockIO.instances[0].fire(true);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(windowFetchCount()).toBe(1);
    const instanceCountAfterOpen = MockIO.instances.length;

    const details = container.querySelector('details.group-disclosure') as HTMLDetailsElement;
    // Close the group; this disconnects observers and aborts in-flight fetches.
    await act(async () => {
      details.open = false;
      details.dispatchEvent(new Event('toggle'));
      await Promise.resolve();
    });

    // Reopen: each card's mount effect runs `armHydration` again, creating a
    // fresh `MockIO` instance per card.
    await act(async () => {
      details.open = true;
      details.dispatchEvent(new Event('toggle'));
      await Promise.resolve();
    });
    // Re-arming must have created at least one new `MockIO` instance.
    expect(MockIO.instances.length).toBeGreaterThan(instanceCountAfterOpen);

    // Fire the re-armed observer for card 1 (which was NOT hydrated before close).
    // The newest `MockIO` instances correspond to the re-armed cards; fire the
    // last one to trigger card 1's fetch.
    const fetchCountBeforeRefire = windowFetchCount();
    await act(async () => {
      MockIO.instances[MockIO.instances.length - 1].fire(true);
      await Promise.resolve();
      await Promise.resolve();
    });
    // Card 1 should now have triggered a fetch, proving re-arming created a
    // working observer.
    expect(windowFetchCount()).toBeGreaterThan(fetchCountBeforeRefire);
  });

  it('reopen re-schedules a fetch even when the aborted task was still QUEUED (UF-1 regression)', async () => {
    // Saturate the hydration queue (concurrency 4) with rejectable blockers so
    // the target card's `?n=100` task stays QUEUED and never runs. Only the
    // synchronous entry-clear in `abortInFlightFetches` can then make reopen
    // schedule a fresh fetch; without it, reopen joins the stale entry and the
    // card stays blank.
    const rejecters: Array<(reason: unknown) => void> = [];
    for (let i = 0; i < 4; i++) {
      // Suppress the unhandled-rejection warning: the entry promise is rejected
      // intentionally in the `finally` to drain the module-singleton queue.
      hydrationQueue
        .schedule(() => new Promise((_res, rej) => rejecters.push(rej)), 10_000)
        .promise.catch(() => {});
    }
    // Flush one microtask so the 4 blocker tasks actually start (occupy the
    // concurrency slots) before the target card schedules its task.
    await act(async () => {
      await Promise.resolve();
    });
    // All 4 slots must be occupied before the target schedules.
    expect(rejecters.length).toBe(4);

    // Spy on `hydrationQueue.schedule` AFTER the blockers so only the target
    // card's calls are counted.
    const scheduleSpy = vi.spyOn(hydrationQueue, 'schedule');
    try {
      await renderGroup(1);
      // Fire card 0's observer. Its `?n=100` task is queued behind the blockers
      // (all concurrency slots are taken); no fetch runs yet.
      await act(async () => {
        MockIO.instances[0].fire(true);
        await Promise.resolve();
      });
      const scheduledAfterOpen = scheduleSpy.mock.calls.length;

      const details = container.querySelector('details.group-disclosure') as HTMLDetailsElement;
      // Close the group. With the fix, `abortInFlightFetches` synchronously
      // clears `state.initialFetchEntry`; the queued task never runs because
      // all concurrency slots are still held by the blockers.
      await act(async () => {
        details.open = false;
        details.dispatchEvent(new Event('toggle'));
        await Promise.resolve();
      });

      // Reopen and fire the freshly re-armed observer for card 0.
      await act(async () => {
        details.open = true;
        details.dispatchEvent(new Event('toggle'));
        MockIO.instances[MockIO.instances.length - 1].fire(true);
        await Promise.resolve();
      });

      // The reopen MUST have scheduled a new task. Without UF-1a the stale
      // entry survives the close and `ensureInitialPayload` joins it instead of
      // scheduling a fresh one, so the count stays at `scheduledAfterOpen` and
      // this assertion fails against the pre-fix code.
      expect(scheduleSpy.mock.calls.length).toBeGreaterThan(scheduledAfterOpen);
    } finally {
      // Drain the module-singleton queue so later tests are not affected.
      rejecters.forEach((reject) => reject(new Error('test cleanup')));
      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });
    }
  });

  it('closing the group disconnects observers and aborts in-flight fetches', async () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal('fetch', (url: string | URL, init?: { signal?: AbortSignal }) => {
      fetchCalls.push(String(url));
      if (init?.signal) {
        signals.push(init.signal);
      }
      return new Promise<Response>((_res, reject) => {
        init?.signal?.addEventListener('abort', () =>
          reject(init.signal?.reason ?? new DOMException('Aborted', 'AbortError')),
        );
      });
    });
    await renderGroup(2);
    await act(async () => {
      MockIO.instances[0].fire(true);
      await Promise.resolve();
    });
    expect(signals.length).toBe(1);
    expect(signals[0].aborted).toBe(false);
    const details = container.querySelector('details.group-disclosure') as HTMLDetailsElement;
    await act(async () => {
      details.open = false;
      details.dispatchEvent(new Event('toggle'));
      await Promise.resolve();
    });
    expect(signals[0].aborted).toBe(true);
    expect(MockIO.instances.every((io) => io.disconnected)).toBe(true);
  });
});

describe('PR-5.0.97 group-bundle hydration', () => {
  let container: HTMLElement;
  let root: Root | null = null;
  let fetchCalls: string[];

  /** Build a `/api/group/{slug}?n=100` bundle covering `slugs`. */
  function bundleResponse(slugs: readonly string[]): Response {
    const charts = slugs.map((slug, i) => ({
      name: `q${i}`,
      slug,
      ...windowedPayload(3572),
    }));
    return jsonResponse({ name: 'g', charts });
  }

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    fetchCalls = [];
    MockIO.instances = [];
    resetPayloadCache();
    vi.stubGlobal('IntersectionObserver', MockIO);
    container = document.createElement('div');
    document.body.appendChild(container);
  });

  afterEach(async () => {
    await act(async () => {
      root?.unmount();
    });
    root = null;
    container.remove();
    vi.unstubAllGlobals();
    vi.useRealTimers();
    resetPayloadCache();
  });

  /**
   * Render `n` charts (slugs `s0..s(n-1)`) inside one OPEN group disclosure with
   * `groupSlug="g"`, so the islands route through the group-bundle path.
   */
  async function renderGroup(n: number): Promise<void> {
    const mounts = Array.from({ length: n }, (_, i) => `<div id="m${i}"></div>`).join('');
    container.innerHTML =
      '<section class="group-details">' +
      '<details class="group-disclosure" open><summary class="group-summary">g</summary></details>' +
      `<div class="chart-grid">${mounts}</div>` +
      '</section>';
    const roots: Root[] = [];
    await act(async () => {
      for (let i = 0; i < n; i++) {
        const r = createRoot(container.querySelector(`#m${i}`) as HTMLElement);
        roots.push(r);
        r.render(<Chart slug={`s${i}`} name={`q${i}`} index={i} groupSlug="g" />);
      }
    });
    root = {
      unmount: () => roots.forEach((r) => r.unmount()),
      render: () => {},
    } as unknown as Root;
    await act(async () => {
      await Promise.resolve();
    });
  }

  function bundleFetchCount(): number {
    return fetchCalls.filter((u) => u.includes('/api/group/')).length;
  }

  function chartFetchCount(): number {
    return fetchCalls.filter((u) => u.includes('/api/chart/')).length;
  }

  it('group open issues exactly ONE bundle fetch and NO per-chart fetch for N islands', async () => {
    vi.stubGlobal('fetch', (url: string | URL) => {
      fetchCalls.push(String(url));
      return Promise.resolve(bundleResponse(['s0', 's1', 's2', 's3', 's4']));
    });
    await renderGroup(5);
    // The bundle is kicked eagerly from `armHydration` for every island, but the
    // per-group in-flight dedupe collapses the five calls into ONE fetch.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(bundleFetchCount()).toBe(1);
    expect(chartFetchCount()).toBe(0);
    expect(fetchCalls[0]).toContain('/api/group/g?n=100');
  });

  it('after the bundle resolves, firing an island IO constructs it (IO disconnects)', async () => {
    vi.stubGlobal('fetch', (url: string | URL) => {
      fetchCalls.push(String(url));
      return Promise.resolve(bundleResponse(['s0', 's1']));
    });
    await renderGroup(2);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(bundleFetchCount()).toBe(1);
    await act(async () => {
      MockIO.instances[0].fire(true);
      await Promise.resolve();
    });
    // Firing the IO disconnects it and drives construction (Chart.js load is a
    // parked stub here, so the observable construction signal is the disconnect).
    expect(MockIO.instances[0].disconnected).toBe(true);
    expect(chartFetchCount()).toBe(0);
  });

  it('closing the group aborts the in-flight bundle (its fetch signal aborts)', async () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal('fetch', (url: string | URL, init?: { signal?: AbortSignal }) => {
      fetchCalls.push(String(url));
      if (init?.signal) {
        signals.push(init.signal);
      }
      return new Promise<Response>((_res, reject) => {
        init?.signal?.addEventListener('abort', () =>
          reject(init.signal?.reason ?? new DOMException('Aborted', 'AbortError')),
        );
      });
    });
    await renderGroup(2);
    await act(async () => {
      await Promise.resolve();
    });
    expect(signals.length).toBe(1);
    expect(signals[0].aborted).toBe(false);
    const details = container.querySelector('details.group-disclosure') as HTMLDetailsElement;
    await act(async () => {
      details.open = false;
      details.dispatchEvent(new Event('toggle'));
      await Promise.resolve();
    });
    expect(signals[0].aborted).toBe(true);
  });

  it('reopen AFTER the bundle succeeded issues ZERO new fetches (cache hit)', async () => {
    vi.stubGlobal('fetch', (url: string | URL) => {
      fetchCalls.push(String(url));
      return Promise.resolve(bundleResponse(['s0', 's1']));
    });
    await renderGroup(2);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(bundleFetchCount()).toBe(1);
    const callsAfterOpen = fetchCalls.length;

    const details = container.querySelector('details.group-disclosure') as HTMLDetailsElement;
    await act(async () => {
      details.open = false;
      details.dispatchEvent(new Event('toggle'));
      await Promise.resolve();
    });
    await act(async () => {
      details.open = true;
      details.dispatchEvent(new Event('toggle'));
      await Promise.resolve();
      await Promise.resolve();
    });
    // The payload cache is warm, so reopen seeds every island synchronously and
    // issues no new fetch of any kind.
    expect(fetchCalls.length).toBe(callsAfterOpen);
  });

  it('a slug absent from the bundle falls back to one /api/chart fetch', async () => {
    vi.stubGlobal('fetch', (url: string | URL) => {
      fetchCalls.push(String(url));
      // The bundle covers only `s0`; `s1` is missing and must fall back.
      if (String(url).includes('/api/group/')) {
        return Promise.resolve(bundleResponse(['s0']));
      }
      return Promise.resolve(jsonResponse(windowedPayload(3572)));
    });
    await renderGroup(2);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(bundleFetchCount()).toBe(1);
    // The per-chart fallback only fires once a card is intersected (construction
    // is IO-gated). Fire both observers; only `s1` (uncovered) must refetch.
    await act(async () => {
      MockIO.instances[0].fire(true);
      MockIO.instances[1].fire(true);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });
    const chartCalls = fetchCalls.filter((u) => u.includes('/api/chart/'));
    expect(chartCalls.length).toBe(1);
    expect(chartCalls[0]).toContain('/api/chart/s1?n=100');
  });

  it('a bundle 404 makes every island fall back to its own /api/chart fetch', async () => {
    vi.stubGlobal('fetch', (url: string | URL) => {
      fetchCalls.push(String(url));
      if (String(url).includes('/api/group/')) {
        return Promise.resolve({ ok: false, status: 404 } as unknown as Response);
      }
      return Promise.resolve(jsonResponse(windowedPayload(3572)));
    });
    await renderGroup(3);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(bundleFetchCount()).toBe(1);
    // Every card intersecting after the 404 falls back to its own per-chart fetch.
    await act(async () => {
      MockIO.instances[0].fire(true);
      MockIO.instances[1].fire(true);
      MockIO.instances[2].fire(true);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });
    const chartCalls = fetchCalls.filter((u) => u.includes('/api/chart/'));
    expect(chartCalls.length).toBe(3);
    expect(chartCalls.map((u) => u).sort()).toEqual([
      '/api/chart/s0?n=100',
      '/api/chart/s1?n=100',
      '/api/chart/s2?n=100',
    ]);
  });

  it('closing a group while a card awaits the bundle issues NO per-chart fetch', async () => {
    // The bundle resolves only AFTER the group has closed, and it does NOT cover
    // this slug. Without the `!this.groupIsOpen()` guard in `ensureInitialPayload`,
    // the bundle `.then` would fire post-close and fall back to a per-chart
    // `/api/chart/` fetch that the already-run close abort can no longer cancel.
    let resolveBundle: (() => void) | null = null;
    vi.stubGlobal('fetch', (url: string | URL) => {
      fetchCalls.push(String(url));
      if (String(url).includes('/api/group/')) {
        return new Promise<Response>((resolve) => {
          // The bundle is empty (covers no slug), so a fall-through would refetch.
          resolveBundle = () => resolve(bundleResponse([]));
        });
      }
      return Promise.resolve(jsonResponse(windowedPayload(3572)));
    });
    await renderGroup(1);
    // Fire the card's observer so it calls `ensureInitialPayload` and begins
    // awaiting the still-pending bundle.
    await act(async () => {
      MockIO.instances[0].fire(true);
      await Promise.resolve();
    });
    expect(bundleFetchCount()).toBe(1);
    expect(chartFetchCount()).toBe(0);

    // Close the group: this aborts the in-flight bundle and per-chart fetches.
    const details = container.querySelector('details.group-disclosure') as HTMLDetailsElement;
    await act(async () => {
      details.open = false;
      details.dispatchEvent(new Event('toggle'));
      await Promise.resolve();
    });

    // Settle the bundle (an aborted fetch would normally reject, but resolving it
    // here exercises the worst case: the `.then` fires after the close abort ran).
    // The close-path continuation is a deep promise chain (`bundleQueue` task
    // `.then` -> entry resolve -> `ensureGroupBundle` settle -> the joined card
    // `.then` that evaluates the `groupIsOpen` guard -> a per-chart fetch on the
    // pre-fix path). A fixed tick count sits at the edge of how many microtask
    // hops that needs and flakes on a cold start, so drain microtasks until the
    // fetch count has stopped changing rather than counting hops by hand. The
    // "stable across consecutive drains" requirement is what makes this a real
    // pin: on the pre-fix path the suppressed-by-the-guard `/api/chart/` fetch
    // surfaces several hops after the bundle settles (it routes through the
    // `hydrationQueue`), and this loop keeps draining until that late fetch would
    // have appeared, so a regression cannot slip through as a premature pass.
    await act(async () => {
      resolveBundle?.();
      // A generous bounded drain: settle when the fetch count holds steady across
      // `stableThreshold` consecutive empty microtask flushes, capped so a hang
      // fails the test rather than spinning forever.
      const maxFlushes = 200;
      const stableThreshold = 10;
      let stableFlushes = 0;
      let lastCount = fetchCalls.length;
      for (let i = 0; i < maxFlushes && stableFlushes < stableThreshold; i++) {
        await Promise.resolve();
        if (fetchCalls.length === lastCount) {
          stableFlushes += 1;
        } else {
          lastCount = fetchCalls.length;
          stableFlushes = 0;
        }
      }
    });

    // The guard must have suppressed the per-chart fallback for the closed group.
    const chartCalls = fetchCalls.filter((u) => u.includes('/api/chart/'));
    expect(chartCalls.length).toBe(0);
  });

  it('reopen after a bundle 404 re-issues the group bundle fetch', async () => {
    // A 404 leaves `completedBundles` unset; `abortGroupBundle` (on close) clears
    // `attemptedBundles`, so a reopen must re-attempt the bundle rather than
    // short-circuit. This pins that re-attempt behavior.
    vi.stubGlobal('fetch', (url: string | URL) => {
      fetchCalls.push(String(url));
      if (String(url).includes('/api/group/')) {
        return Promise.resolve({ ok: false, status: 404 } as unknown as Response);
      }
      return Promise.resolve(jsonResponse(windowedPayload(3572)));
    });
    await renderGroup(2);
    // Let the eager bundle fetch settle as a 404, then let each island fall back.
    await act(async () => {
      MockIO.instances[0].fire(true);
      MockIO.instances[1].fire(true);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(bundleFetchCount()).toBe(1);

    const details = container.querySelector('details.group-disclosure') as HTMLDetailsElement;
    await act(async () => {
      details.open = false;
      details.dispatchEvent(new Event('toggle'));
      await Promise.resolve();
    });
    await act(async () => {
      details.open = true;
      details.dispatchEvent(new Event('toggle'));
      await Promise.resolve();
      await Promise.resolve();
    });
    // The reopen re-attempts the bundle because the 404 never marked it complete.
    expect(bundleFetchCount()).toBe(2);
  });

  it('two groups opened together respect BUNDLE_CONCURRENCY and top-group priority', async () => {
    const scheduleSpy = vi.spyOn(bundleQueue, 'schedule');
    vi.stubGlobal('fetch', (url: string | URL) => {
      fetchCalls.push(String(url));
      return Promise.resolve(bundleResponse(['ga0', 'gb0']));
    });
    // Two groups, each with one island. The page-wide `index` drives priority,
    // so the top group (index 0) outranks the lower one (index 1).
    container.innerHTML =
      '<section class="group-details" id="ga">' +
      '<details class="group-disclosure" open><summary class="group-summary">ga</summary></details>' +
      '<div class="chart-grid"><div id="ma"></div></div>' +
      '</section>' +
      '<section class="group-details" id="gb">' +
      '<details class="group-disclosure" open><summary class="group-summary">gb</summary></details>' +
      '<div class="chart-grid"><div id="mb"></div></div>' +
      '</section>';
    const ra = createRoot(container.querySelector('#ma') as HTMLElement);
    const rb = createRoot(container.querySelector('#mb') as HTMLElement);
    await act(async () => {
      ra.render(<Chart slug="ga0" name="qa" index={0} groupSlug="ga" />);
      rb.render(<Chart slug="gb0" name="qb" index={1} groupSlug="gb" />);
    });
    root = {
      unmount: () => {
        ra.unmount();
        rb.unmount();
      },
      render: () => {},
    } as unknown as Root;
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    // One bundle fetch per group (two groups, two distinct slugs).
    const groupCalls = fetchCalls.filter((u) => u.includes('/api/group/'));
    expect(groupCalls.length).toBe(2);
    const priorities = scheduleSpy.mock.calls.map((c) => c[1]);
    // index 0 => priority 0 (top group); index 1 => priority -1.
    expect(priorities).toContain(0);
    expect(priorities).toContain(-1);
    expect(Math.max(...(priorities as number[]))).toBe(0);
  });
});
