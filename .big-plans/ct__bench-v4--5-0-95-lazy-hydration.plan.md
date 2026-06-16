# PR-5.0.95: Lazy-hydration + resilient loading for large chart groups — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make opening a large chart group (e.g. clickbench, ~43 charts) hydrate only the ~visible charts top-first (the rest on scroll), give every fetch a timeout/abort/retry so a stalled request can no longer spin "loading…" forever, and replace the static loading text with an accessible spinner.

**Architecture:** All work is in the v4 Next.js read service under `benchmarks-website/web/`. The per-chart client island `components/Chart.tsx` owns fetch orchestration through two bounded priority queues (`hydrationQueue` for `?n=100`, `fullHistoryQueue` for `?n=all`, in `lib/chart-store.ts`). Today the landing `details` branch hydrates every chart in an opened group at once, out of visual order, with no viewport gating and no fetch timeout/abort. We (A) gate each landing group-chart's initial fetch behind an `IntersectionObserver` (reusing the permalink page's existing IO shape) and schedule top-first by visual index; (B) wire a per-fetch `AbortController` + a `FETCH_TIMEOUT_MS` timeout into both `fetch()` calls plus a clickable initial-fetch retry; (C) animate the loading state with a `prefers-reduced-motion` guard. No chart-construction or server-query changes.

**Tech Stack:** TypeScript, React 19 (client island, imperative-on-refs), Chart.js (lazy), Vitest (jsdom + node env), CSS.

---

## Context the engineer needs before starting

- **`benchmarks-website/web/` is the working dir** for every `npm`/`npx`/`vitest` command. `npm test` runs `vitest run`. Docker is NOT needed for any test in this plan (pure jsdom/node unit tests).
- **The authoritative design** is `.big-plans/ct__bench-v4-uiux-r2-design.md`. Its "## Open decisions RESOLVED (2026-06-12, pre-implementation)" section already pins every decision this plan encodes; do not re-open them. The pre-implementation investigation in that doc confirmed the hangs are a client-side burst + cold-start + no-recovery, NOT slow server queries, so there is **no server-side work** in this PR.
- **Chart.js never constructs in jsdom unit tests.** The tests mock `@/lib/chart-js` so `loadChartJs` returns a **never-resolving** promise; `maybeConstruct` parks at its `await` and never reaches `new Chart(...)`, letting the fetch path run to completion without constructing a chart. Do not deviate from this harness shape (it is exactly what `components/Chart.loading.test.tsx` already does).
- **The two fetch sites** in `components/Chart.tsx`:
  - `ensureInitialPayload(priority, showLoading)` (~L440-508): the `?n=100` window fetch, scheduled on `hydrationQueue`. Its `.then(onFulfilled, onRejected)` handlers update `state.payload`/chip/loading or set an error.
  - `ensureFullHistory(priority)` (~L532-590): the one-shot `?n=all` upgrade, scheduled on `fullHistoryQueue`; its catch sets `state.chipError = true` (the chip's retry).
  - Both currently call `fetch(url, { headers: { accept: 'application/json' } })` with **no signal and no timeout**.
- **The controller's `aborter`** (`this.aborter`, `new AbortController()`, ~L378) is controller-lifetime: `destroy()` calls `this.aborter.abort()` (~L1467) and every controller-attached DOM listener uses its signal. It must NOT be used as a per-fetch signal (aborting it permanently disables the controller). Per-fetch aborts use a fresh `AbortController` bridged to it.
- **Priority queue** (`lib/chart-store.ts`): `makeQueue(concurrency)` returns `{ schedule(task, priority), drain() }`; `drain()` sorts `queue.sort((a, b) => b.priority - a.priority)` so **higher priority drains first**. `hydrationQueue` concurrency is `HYDRATION_CONCURRENCY = 4`. `ensureInitialPayload` is the ONLY scheduler on `hydrationQueue`.
- **`nextGroupOpenPriority()`** (`lib/chart-store.ts` L115) + **`GROUP_OPEN_PRIORITY_STEP`** (`lib/chart-format.ts` L38) are used in exactly one place: `Chart.tsx:517` inside `onGroupOpen()`. After Task 3 removes that use, both become dead code and are removed.
- **The permalink page already lazy-constructs via `IntersectionObserver`** in the mount-effect `else` branch (~L1646-1661, `rootMargin: '150px 0px'`, one-shot disconnect-on-intersect). Task 3 reuses that exact shape for the landing `details` branch, except the landing branch fetches-then-constructs (the permalink has its payload inlined).
- **The error region** is React state: `{error && <div className="chart-error">{error}</div>}` (~L1809). `.chart-error` has `pointer-events: none` in `app/globals.css` (~L1171), so a clickable retry control must opt pointer events back in. A separate 4s auto-dismiss effect (`useEffect` on `[error]`, ~L1687) clears the error and retries CONSTRUCTION (not the fetch); Task 4 makes that effect skip retryable fetch errors so the retry control persists.
- **The chip is driven imperatively** by `syncWindowChip()` reading `state` and mutating the ref'd `<button data-role="window-chip">` (~L1011, rendered ~L1729). Its `data-state` is one of `hidden|windowed|loading|complete|error`. Do NOT route chip text through React state. The chip's existing retry (`onWindowChipClick`) is the pattern the initial-fetch retry mirrors.
- **`data-chart-index={index}`** (~L1720): `index` is the page-unique chart index, contiguous and ascending in DOM (top-to-bottom) order, so within a group a lower `index` is visually higher. Task 3 schedules the initial fetch at `priority = -index` so top charts drain first.
- **Self-test before each commit:** `npm test` (full vitest), and for the final task also `npx tsc --noEmit`, `npm run build`, `npm run lint`, `npm run format:check`. Targeted runs use `npx vitest run <file>`.
- **Commits** follow the repo convention `<area>: <scope>` and MUST be signed off:
  `Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>`
  plus the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Use `git commit -F <file>` (or `-m` with NO backticks in the message) to avoid shell command-substitution.

---

### Task 1: New constants — fetch timeout + lazy-hydration root margin

**Files:**
- Modify: `benchmarks-website/web/lib/chart-format.ts` (priority/timing constants block, ~lines 36-47)
- Test: `benchmarks-website/web/lib/chart-format.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `lib/chart-format.test.ts` (add the two new names to its imports from `./chart-format`):

```ts
import { FETCH_TIMEOUT_MS, LAZY_HYDRATION_ROOT_MARGIN } from './chart-format';

describe('lazy-hydration + fetch-resilience constants', () => {
  it('the per-fetch timeout is a generous bound over a cold function first-hit', () => {
    // Cold Vercel first-hit measured ~7.8s; 30s leaves headroom so a slow-but-live
    // request is not falsely aborted, while still bounding a true hang.
    expect(FETCH_TIMEOUT_MS).toBe(30000);
  });

  it('the IO root margin pre-hydrates charts slightly before they scroll in', () => {
    expect(LAZY_HYDRATION_ROOT_MARGIN).toMatch(/px/);
  });
});
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `cd benchmarks-website/web && npx vitest run lib/chart-format.test.ts`
Expected: FAIL — `FETCH_TIMEOUT_MS` / `LAZY_HYDRATION_ROOT_MARGIN` are not exported.

- [ ] **Step 3: Add the constants**

In `lib/chart-format.ts`, immediately after the `HOVER_DWELL_MS` declaration (currently ends the priority/timing block around L47):

```ts
/** Per-fetch timeout (ms) for the chart `?n=100` / `?n=all` requests. A stalled
 * request aborts at this bound instead of spinning the loading indicator
 * forever. 30s is generous headroom over a cold Vercel function first-hit
 * (~7.8s measured) so a slow-but-live request is not falsely aborted. */
export const FETCH_TIMEOUT_MS = 30000;
/** `IntersectionObserver` root margin for landing-page lazy hydration: a chart
 * begins hydrating slightly before it scrolls into view so it is rarely blank
 * by the time the user reaches it. */
export const LAZY_HYDRATION_ROOT_MARGIN = '300px 0px';
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `cd benchmarks-website/web && npx vitest run lib/chart-format.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks-website/web/lib/chart-format.ts benchmarks-website/web/lib/chart-format.test.ts
git commit -F - <<'EOF'
benchmarks-website: add FETCH_TIMEOUT_MS + LAZY_HYDRATION_ROOT_MARGIN constants (PR-5.0.95)

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

### Task 2: Per-fetch abort + timeout in both fetches (resilience core)

Wire a fresh per-fetch `AbortController` into BOTH `fetch()` calls, bridged to the controller-lifetime `this.aborter` (so `destroy()` cancels in-flight) plus a `FETCH_TIMEOUT_MS` timeout (so a stall aborts). Store the per-fetch controllers on `state` so a group close can abort them without destroying the controller. Distinguish in the catch: a close/destroy abort (`AbortError`) is silent; a timeout (`TimeoutError`) or genuine failure surfaces the error.

**Files:**
- Modify: `benchmarks-website/web/components/Chart.tsx`
  - `CardState` interface (~L107-139): add two fields
  - constructor `this.state = {...}` (~L388-409): init the two fields
  - import block (~L50-58): add `FETCH_TIMEOUT_MS`
  - `ensureInitialPayload` (~L440-508): per-fetch controller + timeout + catch
  - `ensureFullHistory` (~L532-590): per-fetch controller + timeout + catch
  - new method `abortInFlightFetches()` near `destroy()` (~L1462)
- Test: `benchmarks-website/web/components/Chart.loading.test.tsx` (append a describe block)

- [ ] **Step 1: Write the failing tests (append to `Chart.loading.test.tsx`)**

Append this describe block at the end of the file (it reuses the file's existing `jsonResponse`, `windowedPayload`, `renderOpenGroup`, `container`, `root`, `fetchCalls` — all module/closure scope in that file). It adds a signal-honoring fetch stub via a local helper so the never-resolving fetch actually rejects on abort:

```ts
describe('PR-5.0.95 fetch resilience: timeout + abort', () => {
  // A fetch stub whose `?n=100` response never resolves on its own but DOES
  // reject when its AbortSignal fires, so timeout/destroy aborts are observable.
  // Returns the captured signals for assertions.
  function stubAbortableWindowFetch(): { signals: AbortSignal[] } {
    const signals: AbortSignal[] = [];
    vi.stubGlobal('fetch', (url: string | URL, init?: { signal?: AbortSignal }) => {
      const u = String(url);
      fetchCalls.push(u);
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
      await vi.advanceTimersByTimeAsync(30000);
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
});
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx`
Expected: FAIL — no signal is passed (first test), and the stalled fetch is never aborted (timeout test).

- [ ] **Step 3: Add the two `CardState` fields**

In `interface CardState` (after `fullFetchPending: Promise<void> | null;`, ~L121):

```ts
  /** The in-flight `?n=100` fetch's per-fetch aborter; lets a group close or
   * destroy cancel it without aborting the controller-lifetime `aborter`. */
  initialFetchController: AbortController | null;
  /** The in-flight `?n=all` fetch's per-fetch aborter; same role as above. */
  fullFetchController: AbortController | null;
```

In the constructor's `this.state = { ... }` initializer (alongside `initialFetchEntry: null,` ~L396):

```ts
      initialFetchController: null,
      fullFetchController: null,
```

- [ ] **Step 4: Import `FETCH_TIMEOUT_MS`**

Add `FETCH_TIMEOUT_MS,` to the existing import from `@/lib/chart-format` (the block around L50-58 that already imports `CHART_FETCH_N`, `FETCH_N`, `nextGroupOpenPriority`, etc.).

- [ ] **Step 5: Add the `abortInFlightFetches()` method**

Immediately before `destroy()` (~L1462), add:

```ts
  /** Cancel any in-flight `?n=100` / `?n=all` request WITHOUT tearing down the
   * controller, so closing a group (or its IO disconnect) frees server capacity
   * and stops open/close from piling requests up. A reopen re-issues the fetch.
   * The aborts reject the in-flight promises with `AbortError`, which the fetch
   * catch paths treat as a silent cancellation. */
  abortInFlightFetches(): void {
    this.state.initialFetchController?.abort();
    this.state.fullFetchController?.abort();
  }
```

- [ ] **Step 6: Wire the per-fetch aborter + timeout into `ensureInitialPayload`**

Replace the fetch-scheduling section of `ensureInitialPayload` (from `const url = ...` through `state.initialFetchEntry = entry;`, ~L461-469) with:

```ts
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
```

Then update the `.then(onFulfilled, onRejected)` handlers (~L470-507): in the **fulfilled** handler add `state.initialFetchController = null;` next to the existing `state.initialFetchEntry = null;`. Replace the **rejected** handler body (currently sets `initialFetchEntry = null`, returns if disposed, clears loading, sets a generic error) with:

```ts
      (err: unknown) => {
        state.initialFetchEntry = null;
        state.initialFetchController = null;
        if (state.disposed) {
          return;
        }
        this.cb.setLoading(false);
        // A close/destroy cancellation aborts with `AbortError`: stay silent, the
        // card re-hydrates on reopen. A timeout (`TimeoutError`) or a genuine
        // network/HTTP failure surfaces the error indicator.
        if (err instanceof DOMException && err.name === 'AbortError') {
          return;
        }
        const message =
          err instanceof DOMException && err.name === 'TimeoutError'
            ? 'timed out'
            : err instanceof Error
              ? err.message
              : 'unknown error';
        this.cb.setError(`failed to load: ${message}`);
      },
```

(The retryable-flag wiring is added in Task 4; this task only restores the error/silent split.)

- [ ] **Step 7: Wire the per-fetch aborter + timeout into `ensureFullHistory`**

Replace the scheduling section of `ensureFullHistory` (from `const url = ...` through `state.fullFetchEntry = entry;`, ~L548-559) with:

```ts
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
```

In `ensureFullHistory`'s `.catch((err) => {...})` (~L577-582), clear the controller and keep the close/destroy abort silent (do not set `chipError` for a cancellation):

```ts
      .catch((err: unknown) => {
        state.fullFetchController = null;
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
```

And in the final `.then(() => {...})` (~L583-587) add `state.fullFetchController = null;` next to `state.fullFetchEntry = null;`.

- [ ] **Step 8: Run the tests, confirm they pass**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx`
Expected: PASS (all existing tests in the file plus the four new resilience tests).

- [ ] **Step 9: Run the full web suite (catch any pinned ordering)**

Run: `cd benchmarks-website/web && npm test`
Expected: PASS (the full suite, ~214+ tests, all green).

- [ ] **Step 10: Commit**

```bash
git add benchmarks-website/web/components/Chart.tsx benchmarks-website/web/components/Chart.loading.test.tsx
git commit -F - <<'EOF'
benchmarks-website: per-fetch AbortController + FETCH_TIMEOUT_MS timeout on both chart fetches (PR-5.0.95)

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

### Task 3: Viewport-gated lazy hydration on the landing page (A — the highest-leverage change)

Gate each landing group-chart's initial `?n=100` fetch behind an `IntersectionObserver` (reusing the permalink `else`-branch shape), so opening a group hydrates only the ~visible charts, top-first by visual `index`, and the rest hydrate on scroll. Drop the all-charts summary-hover bulk prefetch and the now-moot `+20`. Closing the group disconnects the observer and aborts in-flight fetches; reopening re-arms. Degrade gracefully to immediate hydration where `IntersectionObserver` is undefined.

**Files:**
- Modify: `benchmarks-website/web/components/Chart.tsx`
  - import block: add `LAZY_HYDRATION_ROOT_MARGIN`; remove `nextGroupOpenPriority`
  - `onGroupOpen()` (~L516-524): take a `priority` argument; drop `nextGroupOpenPriority`/`+20`
  - mount-effect `details` branch (~L1615-1642): replace eager hydration with an IO; drop the summary bulk prefetch
- Modify: `benchmarks-website/web/lib/chart-store.ts`: remove now-dead `nextGroupOpenPriority` (+ its `GROUP_OPEN_PRIORITY_STEP` import)
- Modify: `benchmarks-website/web/lib/chart-format.ts`: remove now-dead `GROUP_OPEN_PRIORITY_STEP`
- Test: `benchmarks-website/web/components/Chart.lazy-hydration.test.tsx` (NEW)

- [ ] **Step 1: Write the failing tests (new file `components/Chart.lazy-hydration.test.tsx`)**

```tsx
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment jsdom

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Chart } from '@/components/Chart';
import { hydrationQueue } from '@/lib/chart-store';

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
    history: { total_commits: total, start_index: total - 100, loaded_commits: 100, complete: false },
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
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.lazy-hydration.test.tsx`
Expected: FAIL — today the `details` branch hydrates immediately (no IO gating) and the summary hover bulk-prefetches.

- [ ] **Step 3: Change `onGroupOpen()` to take a priority**

Replace `onGroupOpen()` (~L516-524) with:

```ts
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
```

- [ ] **Step 4: Replace the mount-effect `details` branch with an IntersectionObserver**

Replace the entire `if (details) { ... }` block (~L1615-1642, ending just before the `} else {` permalink branch) with:

```ts
    if (details) {
      // Landing page: hydrate each card lazily when it scrolls near the viewport
      // (reusing the permalink page's IntersectionObserver shape), so opening a
      // big group hydrates only the ~visible charts, top-first by visual index,
      // and the rest hydrate on scroll. The `toggle` event also fires for
      // scripted `details.open` writes, which is how Expand All reaches every
      // island. Closing the group disconnects the observer and aborts in-flight
      // fetches; reopening re-arms.
      const priority = -index;
      let io: IntersectionObserver | null = null;
      const armHydration = (): void => {
        if (io || typeof IntersectionObserver === 'undefined') {
          // No IO support: hydrate immediately (graceful degradation; also the
          // path unit tests without an IO mock exercise).
          if (typeof IntersectionObserver === 'undefined') {
            controller.onGroupOpen(priority);
          }
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
```

Note: the summary `pointerenter`/`focusin` bulk-prefetch block is intentionally GONE (its responsibility is replaced by IO gating; keeping it would re-introduce the all-cards burst on a mere hover). The permalink `else` branch below is unchanged.

- [ ] **Step 5: Drop the dead `nextGroupOpenPriority` import + definition**

- In `Chart.tsx`, remove `nextGroupOpenPriority,` from the `@/lib/chart-format`... wait, it is imported from `@/lib/chart-store`. Remove `nextGroupOpenPriority` from the import that brings it in (confirm with `grep -n nextGroupOpenPriority components/Chart.tsx`); add `LAZY_HYDRATION_ROOT_MARGIN` to the `@/lib/chart-format` import.
- In `lib/chart-store.ts`, delete the `nextGroupOpenPriority` function + the `let groupOpenPriority = 0;` line (~L108-118) and drop `GROUP_OPEN_PRIORITY_STEP` from its import from `@/lib/chart-format`.
- In `lib/chart-format.ts`, delete the `GROUP_OPEN_PRIORITY_STEP` constant (~L36-38).
- Confirm nothing else references either name:

Run: `cd benchmarks-website/web && grep -rn "nextGroupOpenPriority\|GROUP_OPEN_PRIORITY_STEP" lib components app`
Expected: no matches.

- [ ] **Step 6: Run the new tests, confirm they pass**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.lazy-hydration.test.tsx`
Expected: PASS (all eight lazy-hydration tests).

- [ ] **Step 7: Run the FULL web suite (the existing PR-5.0.9 tests must stay green)**

Run: `cd benchmarks-website/web && npm test`
Expected: PASS. The existing `Chart.loading.test.tsx` tests do not mock `IntersectionObserver`; the graceful-degradation branch (`typeof IntersectionObserver === 'undefined'` → immediate `onGroupOpen`) preserves their behavior. If any existing test now mocks/needs IO, fix that test minimally to fire its observer — do NOT weaken an assertion.

- [ ] **Step 8: Commit**

```bash
git add benchmarks-website/web/components/Chart.tsx benchmarks-website/web/components/Chart.lazy-hydration.test.tsx benchmarks-website/web/lib/chart-store.ts benchmarks-website/web/lib/chart-format.ts
git commit -F - <<'EOF'
benchmarks-website: IntersectionObserver-gated top-first lazy hydration on the landing page (PR-5.0.95)

Drop the all-cards summary-hover bulk prefetch and the dead group-open priority
stepping; schedule each group chart's initial fetch at priority = -index so top
charts hydrate first, and only as they scroll into view.

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

### Task 4: Clickable initial-fetch retry affordance (B — recovery)

On an initial-fetch timeout/failure, surface a CLICKABLE retry in the card's error region that re-issues the `?n=100` fetch (mirroring the chip's retry). It is user-initiated, so naturally bounded; there is no automatic fetch-retry loop. Make the 4s auto-dismiss skip retryable fetch errors so the control persists until clicked.

**Files:**
- Modify: `benchmarks-website/web/components/Chart.tsx`
  - `CardCallbacks` interface (~L154-158): add `setRetryable`
  - `ensureInitialPayload` rejected handler (Task 2's version): set retryable on a real failure; clear it on a fresh attempt
  - new method `retryInitialPayload()`
  - controller construction callbacks (~L1574): pass `setRetryable`
  - component state + render + auto-dismiss effect (~L1505-1717): `retryable` state, retry button, skip auto-dismiss when retryable
- Modify: `benchmarks-website/web/app/globals.css`: re-enable pointer events on the retry control
- Test: `benchmarks-website/web/components/Chart.loading.test.tsx` (append)

- [ ] **Step 1: Write the failing tests (append to `Chart.loading.test.tsx`)**

Append to the `PR-5.0.95 fetch resilience` describe (reusing `stubAbortableWindowFetch`), or add a sibling describe. These assert the retry button appears on failure and re-issues the window fetch:

```ts
describe('PR-5.0.95 initial-fetch retry', () => {
  function stubControllableWindowFetch(): {
    rejectNext: (e: unknown) => void;
    calls: () => number;
  } {
    let rejecter: (e: unknown) => void = () => {};
    vi.stubGlobal('fetch', (url: string | URL) => {
      const u = String(url);
      fetchCalls.push(u);
      if (u.includes('n=100')) {
        return new Promise<Response>((_res, rej) => {
          rejecter = rej;
        });
      }
      return Promise.resolve(jsonResponse(windowedPayload(3572)));
    });
    return {
      rejectNext: (e) => rejecter(e),
      calls: () => fetchCalls.filter((u) => u.includes('n=100')).length,
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
    const retry = container.querySelector<HTMLButtonElement>('.chart-error [data-role="fetch-retry"]');
    expect(retry).not.toBeNull();
    await act(async () => {
      retry?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(ctl.calls()).toBe(2);
  });
});
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx`
Expected: FAIL — there is no `[data-role="fetch-retry"]` control yet.

- [ ] **Step 3: Add `setRetryable` to `CardCallbacks`**

In `interface CardCallbacks` (~L154-158), add:

```ts
  /** Show/hide the initial-fetch retry control in the error region. */
  setRetryable: (on: boolean) => void;
```

- [ ] **Step 4: Set retryable in the rejected handler + on a fresh attempt**

In `ensureInitialPayload`, at the TOP of the method's loading branch (right where `showLoading` first sets loading true, ~L458-459) clear any prior retryable state so a re-attempt starts clean:

```ts
    if (showLoading) {
      this.cb.setLoading(true);
      this.cb.setRetryable(false);
    }
```

In the rejected handler (Task 2's version), add `this.cb.setRetryable(true);` immediately after the `this.cb.setError(...)` call (the timeout/failure branch only — NOT the silent `AbortError` early return).

- [ ] **Step 5: Add `retryInitialPayload()`**

Add near `ensureInitialPayload` (after it, ~L508):

```ts
  /** Re-issue the initial `?n=100` fetch after a failure/timeout. User-initiated
   * (the error region's retry control), so it is naturally bounded; clears the
   * error first and schedules at the top of the hydration queue. */
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
```

- [ ] **Step 6: Wire the new callback + state into the component**

In `Chart(...)` add state next to `error` (~L1507):

```ts
  const [retryable, setRetryable] = useState(false);
```

Pass it in the controller construction callbacks (~L1574): `{ setY, setLoading, setError, setRetryable }`.

Make the 4s auto-dismiss effect (~L1687) skip retryable fetch errors so the retry persists. Change its guard:

```ts
  useEffect(() => {
    // A retryable initial-fetch error owns its own dismissal (the user clicks
    // retry), so the 4s construction-retry auto-dismiss does not apply to it.
    if (error === null || retryable) {
      return;
    }
```

and add `retryable` to its dependency array: `}, [error, retryable]);`.

Update the error render (~L1809). Replace:

```tsx
      {error && <div className="chart-error">{error}</div>}
```

with:

```tsx
      {error && (
        <div className="chart-error">
          <span>{error}</span>
          {retryable && (
            <button
              type="button"
              className="chart-error-retry"
              data-role="fetch-retry"
              onClick={() => {
                setError(null);
                setRetryable(false);
                controllerRef.current?.retryInitialPayload();
              }}
            >
              retry
            </button>
          )}
        </div>
      )}
```

- [ ] **Step 7: Re-enable pointer events on the retry control**

In `app/globals.css`, after the `.chart-error` rule (~L1179-1184), add:

```css
.chart-error-retry {
  margin-left: 0.4rem;
  pointer-events: auto;
  cursor: pointer;
  font: inherit;
  font-size: 0.7rem;
  color: var(--accent-fg);
  background: transparent;
  border: 1px solid currentColor;
  border-radius: 4px;
  padding: 0 0.35rem;
  text-decoration: underline;
}
```

- [ ] **Step 8: Run the tests, confirm they pass**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx`
Expected: PASS.

- [ ] **Step 9: Run the full web suite**

Run: `cd benchmarks-website/web && npm test`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add benchmarks-website/web/components/Chart.tsx benchmarks-website/web/components/Chart.loading.test.tsx benchmarks-website/web/app/globals.css
git commit -F - <<'EOF'
benchmarks-website: clickable initial-fetch retry that re-issues the window fetch (PR-5.0.95)

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

### Task 5: Spinner animation for the loading state (C)

Replace the static "loading…" text with an accessible CSS spinner (visually-hidden text retained for screen readers), tie the chip's `data-state="loading"` to a small inline spinner, and disable the animation under `prefers-reduced-motion: reduce`.

**Files:**
- Modify: `benchmarks-website/web/components/Chart.tsx`: the loading render (~L1808)
- Modify: `benchmarks-website/web/app/globals.css`: `@keyframes` + `.chart-spinner` + reduced-motion + chip loading spinner (near ~L1033 and ~L1174)
- Test: `benchmarks-website/web/components/Chart.loading.test.tsx` (append — spinner element renders) and `benchmarks-website/web/app/globals.spinner.test.ts` (NEW, node env — the reduced-motion guard + keyframes exist)

- [ ] **Step 1: Write the failing tests**

Append to `Chart.loading.test.tsx` (the spinner element renders while loading):

```ts
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
```

Create `app/globals.spinner.test.ts` (node env — jsdom does not evaluate CSS `@media`, so assert the rule text exists in the stylesheet):

```ts
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment node

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const css = readFileSync(join(__dirname, 'globals.css'), 'utf8');

describe('PR-5.0.95 spinner CSS', () => {
  it('defines a spin keyframes animation and a spinner rule', () => {
    expect(css).toMatch(/@keyframes\s+chart-spin/);
    expect(css).toMatch(/\.chart-spinner\b/);
  });

  it('disables the spinner animation under prefers-reduced-motion: reduce', () => {
    const reduced = css.match(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{[\s\S]*?\}/g,
    );
    expect(reduced).not.toBeNull();
    expect(reduced!.join('\n')).toMatch(/\.chart-spinner[\s\S]*animation:\s*none/);
  });
});
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx app/globals.spinner.test.ts`
Expected: FAIL — no `.chart-spinner` element/rule yet.

- [ ] **Step 3: Replace the loading markup with a spinner**

In `Chart.tsx`, replace (~L1808):

```tsx
      {loading && <div className="chart-loading">loading…</div>}
```

with:

```tsx
      {loading && (
        <div className="chart-loading" role="status" aria-live="polite">
          <span className="chart-spinner" aria-hidden="true" />
          <span className="chart-loading-text">loading…</span>
        </div>
      )}
```

- [ ] **Step 4: Add the spinner CSS**

In `app/globals.css`, in the `.chart-loading` block area (~L1174), make the container lay out the spinner + text inline, and add the spinner + keyframes + reduced-motion guard:

```css
.chart-loading {
  background: var(--code-bg);
  color: var(--muted);
  border: 1px solid var(--border);
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}
.chart-spinner {
  display: inline-block;
  width: 0.7rem;
  height: 0.7rem;
  border: 2px solid color-mix(in srgb, var(--muted) 35%, transparent);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: chart-spin 0.7s linear infinite;
}
@keyframes chart-spin {
  to {
    transform: rotate(360deg);
  }
}
/* The chip's loading state gets a small inline spinner for consistency. */
.chart-window-chip[data-state='loading']::before {
  content: '';
  display: inline-block;
  width: 0.6rem;
  height: 0.6rem;
  margin-right: 0.3rem;
  vertical-align: -0.05rem;
  border: 2px solid color-mix(in srgb, var(--muted) 35%, transparent);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: chart-spin 0.7s linear infinite;
}
@media (prefers-reduced-motion: reduce) {
  .chart-spinner,
  .chart-window-chip[data-state='loading']::before {
    animation: none;
  }
}
```

(The existing `.chart-window-chip[data-state='loading'] { cursor: progress; }` rule at ~L1033 stays; the `::before` spinner is additive.)

- [ ] **Step 5: Run the tests, confirm they pass**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx app/globals.spinner.test.ts`
Expected: PASS.

- [ ] **Step 6: Run the full web suite**

Run: `cd benchmarks-website/web && npm test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add benchmarks-website/web/components/Chart.tsx benchmarks-website/web/app/globals.css benchmarks-website/web/components/Chart.loading.test.tsx benchmarks-website/web/app/globals.spinner.test.ts
git commit -F - <<'EOF'
benchmarks-website: animated loading spinner with a prefers-reduced-motion guard (PR-5.0.95)

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

### Task 6: Full verification gates (type / build / lint / format / full suite)

**Files:** none (verification only; commit any formatter-only changes)

- [ ] **Step 1: Type-check**

Run: `cd benchmarks-website/web && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 2: Full unit suite**

Run: `cd benchmarks-website/web && npm test`
Expected: PASS (all suites, including the new lazy-hydration, resilience, retry, and spinner tests, plus every pre-existing PR-5.0.9 / lifecycle test).

- [ ] **Step 3: Production build**

Run: `cd benchmarks-website/web && npm run build`
Expected: build succeeds.

- [ ] **Step 4: Lint**

Run: `cd benchmarks-website/web && npm run lint`
Expected: no errors.

- [ ] **Step 5: Format check (and fix if needed)**

Run: `cd benchmarks-website/web && npm run format:check`
If it reports unformatted files, run `npm run format` and re-check.

- [ ] **Step 6: Commit any formatter-only changes**

```bash
git add -A benchmarks-website/web
git commit -F - <<'EOF'
benchmarks-website: prettier formatting for PR-5.0.95

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

(If `git status` shows nothing to commit, skip this step.)

---

## Self-review notes (author check against the design doc)

- **A (lazy hydration):** Task 3 gates landing hydration behind an `IntersectionObserver` (reusing the permalink shape), schedules `priority = -index` (top-first), drops the summary bulk prefetch + the moot `+20`, removes the now-dead `nextGroupOpenPriority`/`GROUP_OPEN_PRIORITY_STEP`, degrades to immediate hydration where IO is absent, and aborts in-flight + disconnects on group close. Tests cover: no fetch until intersect, only-visible hydrate, off-viewport-on-scroll, top-first priority, no summary bulk prefetch, close aborts + disconnects. Matches design §A + Open Decisions 1 & 2.
- **B (resilience):** Task 2 wires a per-fetch `AbortController` (bridged to `this.aborter`) + a `FETCH_TIMEOUT_MS = 30000` `setTimeout` abort into BOTH fetches, with a catch that keeps a close/destroy `AbortError` silent and surfaces a timeout/failure; Task 4 adds the clickable initial-fetch retry that re-issues the `?n=100` fetch and persists past the 4s auto-dismiss. No `AbortSignal.timeout`/`any` (fake-timer testability). Matches design §B + Open Decision 3.
- **C (spinner):** Task 5 replaces the static text with an accessible spinner (visually-hidden text kept), ties the chip loading state to an inline spinner, and disables the animation under `prefers-reduced-motion: reduce`. Matches design §C.
- **Regression:** every task runs `npm test`; Task 3 specifically re-confirms the pre-existing PR-5.0.9 + lifecycle tests stay green via the no-IO graceful-degradation branch. Task 6 runs tsc + build + lint + format. Matches the design's test plan + acceptance criteria.
- **Out of scope (honored):** no server-query changes, no `?n=all` downsampling, no chip/dwell/interaction-promotion mechanic changes (only the chip's loading visual), no layout redesign, no keep-warm infra.
