# PR-5.0.9: Opt-in full-history chart loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the benchmarks site from auto-downloading every chart's full history on group open; make full history a per-chart opt-in (window chip + hover-dwell prefetch + the existing interaction promotion), and lengthen the CDN stale-while-revalidate window.

**Architecture:** All work is in the v4 Next.js read service under `benchmarks-website/web/`. The per-chart client island `components/Chart.tsx` owns fetch orchestration through two bounded priority queues (`hydrationQueue` for `?n=100`, `fullHistoryQueue` for `?n=all`). Today `onGroupOpen` auto-queues a `?n=all` warmup for every chart; we delete that and route full-history fetches through three explicit user-intent triggers instead: a window chip (click), a ~600ms same-card hover dwell, and the already-present `rangeTouchesUnloadedHistory` pan/zoom promotion. The virtual full-length x-axis (`normalizeChartPayload`) already makes late fill-in jank-free, so no chart-construction changes are needed.

**Tech Stack:** TypeScript, React 19 (client island, imperative-on-refs), Chart.js (lazy), Vitest (jsdom + node env), Vercel CDN cache headers.

---

## Context the engineer needs before starting

- **`benchmarks-website/web/` is the working dir** for every `npm`/`vitest` command. Node is via the repo toolchain; `npm test` runs `vitest run`.
- **Docker must be running** for some suites, but the tests in THIS plan are pure jsdom/node unit tests (no testcontainers) — they do not need Docker.
- **Chart.js never constructs in jsdom unit tests.** The existing `components/Chart.lifecycle.test.tsx` mocks `@/lib/chart-js` so construction throws, and avoids construction by parking fetches. The new test file in this plan instead mocks `loadChartJs` to return a **never-resolving** promise, so `maybeConstruct` hangs at its `await` and never reaches `new Chart(...)`. That lets the fetch path run to completion (initial `?n=100` resolves) without constructing a chart. Do not deviate from this harness shape.
- **`history` wire/normalized shape** (`lib/queries.ts` `ChartHistory`, produced by `canonicalHistory`/`normalizeChartPayload` in `lib/chart-format.ts`):
  `{ total_commits: number; start_index: number; loaded_commits: number; complete: boolean }`.
  A windowed latest-100 payload has `loaded_commits === 100`, `total_commits` large, `complete === false`. A chart born with its complete history (fewer than 100 commits) has `loaded_commits === total_commits`, `complete === true`.
- **Priority queue** (`lib/chart-store.ts`): `hydrationQueue` (concurrency `HYDRATION_CONCURRENCY=4`), `fullHistoryQueue` (concurrency `FULL_HISTORY_CONCURRENCY=2`); `schedule(task, priority)` returns a `QueueEntry` whose `.priority` may be bumped and whose `.promise` resolves with the task result. Higher priority drains first.
- **Existing priority constants** (`lib/chart-format.ts`): `GROUP_OPEN_PRIORITY_STEP=100`, `INTERACTION_FULL_PRIORITY=1_000_000`.
- **The chip is driven imperatively**, exactly like the existing `syncDownsampleBadge` (a controller method that reads `state` and mutates a ref'd DOM node). Do NOT route chip text through React state.
- **Self-test before each commit:** `npm test` (vitest), and for the final task also `npx tsc --noEmit`, `npm run build`, `npm run lint`, `npm run format:check`.

---

### Task 1: New constants for the hover-dwell prefetch

**Files:**
- Modify: `benchmarks-website/web/lib/chart-format.ts` (priority-constants block, ~lines 36–40)
- Test: `benchmarks-website/web/lib/chart-format.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `lib/chart-format.test.ts` (import the two new names alongside the existing imports from `./chart-format`):

```ts
import { HOVER_DWELL_MS, HOVER_PREFETCH_PRIORITY, INTERACTION_FULL_PRIORITY } from './chart-format';

describe('hover-dwell prefetch constants', () => {
  it('dwell is a deliberate ~600ms pause, not an accidental sweep', () => {
    expect(HOVER_DWELL_MS).toBe(600);
  });

  it('hover-prefetch priority sits above background (0) and below direct interaction', () => {
    expect(HOVER_PREFETCH_PRIORITY).toBeGreaterThan(0);
    expect(HOVER_PREFETCH_PRIORITY).toBeLessThan(INTERACTION_FULL_PRIORITY);
  });
});
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `cd benchmarks-website/web && npx vitest run lib/chart-format.test.ts`
Expected: FAIL — `HOVER_DWELL_MS`/`HOVER_PREFETCH_PRIORITY` are not exported.

- [ ] **Step 3: Add the constants**

In `lib/chart-format.ts`, immediately after the `INTERACTION_FULL_PRIORITY` declaration (currently `export const INTERACTION_FULL_PRIORITY = 1_000_000;`):

```ts
/** A silent hover-dwell prefetch outranks idle background work but yields to a
 * direct user interaction (chip click, pan/zoom into the unloaded region). */
export const HOVER_PREFETCH_PRIORITY = 500_000;
/** How long the pointer must rest on one chart card before the silent
 * full-history prefetch starts, so a mouse sweep across the page fetches
 * nothing while a deliberate hover has data ready by the time the user acts. */
export const HOVER_DWELL_MS = 600;
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `cd benchmarks-website/web && npx vitest run lib/chart-format.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks-website/web/lib/chart-format.ts benchmarks-website/web/lib/chart-format.test.ts
git commit -F - <<'EOF'
benchmarks-website: add hover-dwell prefetch constants (PR-5.0.9)

Signed-off-by: "Connor Tsui" <connor@spiraldb.com>
EOF
```

---

### Task 2: Remove the automatic `?n=all` warmup from group open

This is the core behavioral change and its discriminating test. After this task, opening a group fetches only the `?n=100` windows; no `?n=all` is issued without explicit intent.

**Files:**
- Modify: `benchmarks-website/web/components/Chart.tsx` (`onGroupOpen`, ~lines 477–486; the header architecture docstring ~lines 80–83; the `ensureFullHistory` docstring ~lines 488–492)
- Create: `benchmarks-website/web/components/Chart.loading.test.tsx`

- [ ] **Step 1: Write the failing test (new file)**

Create `components/Chart.loading.test.tsx`. This file's harness (never-resolving `loadChartJs`, a windowed-payload fetch stub, a `renderOpenGroup` helper) is reused by Tasks 3 and 4.

```tsx
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment jsdom

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Chart } from '@/components/Chart';
import { fullHistoryQueue } from '@/lib/chart-store';

// Mock Chart.js construction to a NEVER-RESOLVING loader: maybeConstruct awaits
// it forever and never reaches `new Chart(...)`, so the fetch-orchestration path
// runs to completion without constructing a chart in jsdom.
vi.mock('@/lib/chart-js', () => ({
  loadChartJs: () => new Promise(() => {}),
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
    history: { total_commits: total, start_index: total - 100, loaded_commits: 100, complete: false },
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

describe('Chart opt-in full-history loading', () => {
  let container: HTMLElement;
  let root: Root | null = null;
  let fetchCalls: string[];
  // Per-URL-substring responders; default resolves a windowed payload.
  let responders: { match: (url: string) => boolean; respond: (url: string) => Promise<Response> }[];

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    fetchCalls = [];
    responders = [];
    vi.stubGlobal('fetch', (url: string | URL) => {
      const u = String(url);
      fetchCalls.push(u);
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
    // Let the queued initial fetch and its normalization microtasks settle.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    return container.querySelector('[data-role="window-chip"]');
  }

  it('opening a group issues the windowed fetch but NO full-history warmup', async () => {
    const scheduleSpy = vi.spyOn(fullHistoryQueue, 'schedule');
    await renderOpenGroup();
    const windowFetches = fetchCalls.filter((u) => u.includes('/api/chart/') && u.includes('n=100'));
    const fullFetches = fetchCalls.filter((u) => u.includes('n=all'));
    expect(windowFetches.length).toBeGreaterThanOrEqual(1);
    expect(fullFetches).toHaveLength(0);
    expect(scheduleSpy).not.toHaveBeenCalled();
  });
});
```

Note: the field names in the payload helpers must match the real `ChartResponse`/`CommitPoint` shape in `lib/queries.ts`. If `tsc` later flags a field, fix the helper to match the wire shape (do not change production types).

- [ ] **Step 2: Run the test, confirm it fails**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx`
Expected: FAIL — `fullHistoryQueue.schedule` IS called (the current warmup), so `fullFetches` is non-empty / the spy assertion fails.

- [ ] **Step 3: Remove the warmup from `onGroupOpen`**

In `components/Chart.tsx`, replace the `onGroupOpen` method (and its docstring) with:

```ts
  /**
   * Group-open hydration: fetch this chart's latest-100 window at the group's
   * base priority and construct. Full history is NOT warmed here; it loads only
   * on explicit per-chart intent (window-chip click, hover dwell, or pan/zoom
   * into the unloaded region) so opening a group costs only the cheap windows.
   */
  onGroupOpen(): void {
    const priority = nextGroupOpenPriority();
    void this.ensureInitialPayload(priority + 20, true).then(() => {
      if (this.state.disposed) {
        return;
      }
      void this.maybeConstruct();
    });
  }
```

(The only removed line is `void this.ensureFullHistory(priority);`. `nextGroupOpenPriority()` is still needed for the initial fetch priority.)

- [ ] **Step 4: Update the two stale docstrings**

In the header architecture note (~lines 80–83), change the bullet that reads
`... then queues the one-shot ?n=all upgrade through [fullHistoryQueue]. Fetch counts and concurrency caps match v3's shard pipeline shape.`
to:

```
 *   so each island lazily fetches its own `/api/chart/{slug}?n=100` through the
 *   shared bounded [`hydrationQueue`] on group open (or pointer intent). The
 *   one-shot `?n=all` upgrade through [`fullHistoryQueue`] is opt-in: it runs
 *   only on per-chart intent (window-chip click, hover dwell, or pan/zoom into
 *   the unloaded region), never as an automatic group-open warmup.
```

In the `ensureFullHistory` docstring (~lines 488–492), change
`... This is the ONLY chart refetch after the initial load; pan/zoom/slider interaction never refetches beyond promoting this hop.`
to:

```ts
  /**
   * Queue the one-shot `?n=all` full-history upgrade (or promote the queued
   * entry's priority). Triggered only by explicit intent — window-chip click
   * (`INTERACTION_FULL_PRIORITY`), hover dwell (`HOVER_PREFETCH_PRIORITY`), or
   * pan/zoom touching the unloaded region — never as an automatic warmup.
   */
```

- [ ] **Step 5: Run the test, confirm it passes**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx`
Expected: PASS — windowed fetch present, zero `n=all`, `fullHistoryQueue.schedule` not called.

- [ ] **Step 6: Run the whole web suite to catch any test that pinned the old warmup ordering**

Run: `cd benchmarks-website/web && npm test`
Expected: PASS. If a test in `lib/queries.test.ts` / `lib/server-smoke.test.ts` / elsewhere asserted that group open warms `?n=all`, update it to pin the NEW opt-in behavior (this is an intended behavioral change per the design doc's test plan). If none do, no other test changes are needed.

- [ ] **Step 7: Commit**

```bash
git add benchmarks-website/web/components/Chart.tsx benchmarks-website/web/components/Chart.loading.test.tsx
git commit -F - <<'EOF'
benchmarks-website: drop automatic full-history warmup on group open (PR-5.0.9)

Signed-off-by: "Connor Tsui" <connor@spiraldb.com>
EOF
```

---

### Task 3: Window chip — always-visible window state + click-to-load

Adds the per-card status chip ("latest 100 of 3,572" → "load all N" on hover → spinner → "all N" / "retry") driven imperatively from controller state, mirroring `syncDownsampleBadge`.

**Files:**
- Modify: `benchmarks-website/web/components/Chart.tsx` — `CardState` (add fields), `CardElements` (add `chip`), the state factory (~lines 375–388), `els()` registry (~lines 1396–1403), the JSX (`<h3 className="chart-card-title">`, ~lines 1542–1550), new `chipRef`, `syncWindowChip()`, `onWindowChipClick()`, and chip wiring in `ensureInitialPayload`/`seedPayload`/`ensureFullHistory`.
- Test: `benchmarks-website/web/components/Chart.loading.test.tsx` (append).

- [ ] **Step 1: Write the failing tests (append to `Chart.loading.test.tsx`)**

Add these tests inside the existing `describe('Chart opt-in full-history loading', ...)` block:

```tsx
  it('shows the window chip "latest 100 of 3,572" for a windowed chart', async () => {
    const chip = await renderOpenGroup();
    expect(chip).not.toBeNull();
    expect(chip?.hasAttribute('hidden')).toBe(false);
    expect(chip?.dataset.state).toBe('windowed');
    expect(chip?.textContent).toBe('latest 100 of 3,572');
  });

  it('hides the chip for a chart born with its complete history', async () => {
    responders.push({ match: (u) => u.includes('n=100'), respond: () => Promise.resolve(jsonResponse(completePayload(40))) });
    const chip = await renderOpenGroup();
    expect(chip?.hasAttribute('hidden')).toBe(true);
  });

  it('chip click loads full history at top priority and reaches "all N"', async () => {
    let resolveFull: (r: Response) => void = () => {};
    responders.push({
      match: (u) => u.includes('n=all'),
      respond: () => new Promise<Response>((res) => { resolveFull = res; }),
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
      respond: () => new Promise<Response>((_, rej) => { rejectFull = rej; }),
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
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx`
Expected: FAIL — there is no `[data-role="window-chip"]` element yet.

- [ ] **Step 3: Add the `CardState` fields**

In the `CardState` interface (after `fullFetchPending: Promise<void> | null;`), add:

```ts
  /** True once this card has rendered a bounded window (so the chip is shown);
   * a chart born complete never sets it and shows no chip. */
  everWindowed: boolean;
  /** The most recent full-history fetch failed; the chip offers a retry. */
  chipError: boolean;
  /** The pointer is currently resting on this card (chip shows the action). */
  hovering: boolean;
  /** Pending hover-dwell prefetch timer; cleared on `pointerleave`/destroy. */
  hoverDwellTimer: ReturnType<typeof setTimeout> | null;
```

In the state factory (the object literal returned around lines 375–388), add after `fullFetchPending: null,`:

```ts
      everWindowed: false,
      chipError: false,
      hovering: false,
      hoverDwellTimer: null,
```

- [ ] **Step 4: Add `chip` to `CardElements`**

In the `CardElements` interface (after `badge: HTMLSpanElement | null;`):

```ts
  chip: HTMLButtonElement | null;
```

- [ ] **Step 5: Add the import for the constants**

In the `@/lib/chart-format` import block in `Chart.tsx`, add `HOVER_DWELL_MS,` and `HOVER_PREFETCH_PRIORITY,` (keep the list alphabetical-ish; they sit near `INTERACTION_FULL_PRIORITY`).

- [ ] **Step 6: Add `syncWindowChip()` and `onWindowChipClick()` methods**

Add these methods to `ChartController`, right after `syncDownsampleBadge` (after its closing brace, ~line 954):

```ts
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
    const total = payload.history.total_commits.toLocaleString();
    const loaded = payload.history.loaded_commits.toLocaleString();
    chip.removeAttribute('hidden');
    if (state.fullLoaded) {
      chip.dataset.state = 'complete';
      chip.disabled = true;
      chip.textContent = `all ${total}`;
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
    chip.setAttribute('title', `Showing the latest ${loaded} of ${total} commits. Click to load the full history.`);
  }

  /** Window-chip click: load the full history at top priority, or retry after a
   * failure. A no-op once full history is loaded or a fetch is already pending. */
  onWindowChipClick(): void {
    const state = this.state;
    if (state.disposed || state.fullLoaded || state.fullFetchPending) {
      return;
    }
    state.chipError = false;
    void this.ensureFullHistory(INTERACTION_FULL_PRIORITY);
  }
```

- [ ] **Step 7: Mark `everWindowed` and sync the chip when the window arrives**

In `ensureInitialPayload`'s success callback, after `state.fullLoaded = normalized.history.complete;` (the existing line ~453), add:

```ts
        if (!normalized.history.complete) {
          state.everWindowed = true;
        }
        this.syncWindowChip();
```

In `seedPayload` (after `this.state.fullLoaded = normalized.history.complete;`, ~line 395), add the same two-statement block:

```ts
    if (!normalized.history.complete) {
      this.state.everWindowed = true;
    }
    this.syncWindowChip();
```

- [ ] **Step 8: Drive the chip through `ensureFullHistory`'s state transitions**

Replace the body of `ensureFullHistory` from `state.fullFetchPending = entry.promise` through its `return state.fullFetchPending;` with this version (adds `chipError` clearing, ordered chip syncs, and a synchronous "loading" sync):

```ts
    state.fullFetchPending = entry.promise
      .then((full) => {
        if (state.disposed || full === null) {
          return;
        }
        this.replaceChartPayload(full as ChartResponse);
        state.fullLoaded = true;
        state.chipError = false;
        this.cb.setLoading(false);
        if (!state.chart && this.groupIsOpen()) {
          void this.maybeConstruct();
        }
      })
      .catch((err: unknown) => {
        // Quiet: the latest-100 payload is still usable. Surface to the console
        // for debugging; the chip exposes the retry affordance.
        console.warn('bench: full history fetch failed', err);
        state.chipError = true;
      })
      .then(() => {
        state.fullFetchEntry = null;
        state.fullFetchPending = null;
        this.syncWindowChip();
      });
    this.syncWindowChip();
    return state.fullFetchPending;
```

(The trailing `.then` nulls the entry/pending FIRST, then syncs — so the final chip state reads `complete`/`error`/`windowed`, never a stale `loading`. The synchronous `this.syncWindowChip()` before the `return` shows the spinner while `fullFetchPending` is non-null.)

- [ ] **Step 9: Add the chip element to the JSX and the `els()` registry**

Add a ref near the other refs (~line 1332, after `const badgeRef = useRef<HTMLSpanElement>(null);`):

```ts
  const chipRef = useRef<HTMLButtonElement>(null);
```

In the `els()` construction object (~lines 1396–1403), add after `badge: badgeRef.current,`:

```ts
        chip: chipRef.current,
```

In the JSX title row, add the chip button right after the downsample-badge `<span ... ref={badgeRef} />` and before the closing `</h3>`:

```tsx
        <button
          type="button"
          className="chart-window-chip"
          data-role="window-chip"
          hidden
          ref={chipRef}
          onClick={() => controllerRef.current?.onWindowChipClick()}
        />
```

- [ ] **Step 10: Run the tests, confirm they pass**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx`
Expected: PASS — chip shows "latest 100 of 3,572"; hidden when born complete; click → loading → "all 3,572"; failure → "retry".

- [ ] **Step 11: Run the full web suite**

Run: `cd benchmarks-website/web && npm test`
Expected: PASS (the `Chart.test.tsx` static-markup tests still pass; the chip is `hidden` by default so it does not disturb the title-row contract).

- [ ] **Step 12: Commit**

```bash
git add benchmarks-website/web/components/Chart.tsx benchmarks-website/web/components/Chart.loading.test.tsx
git commit -F - <<'EOF'
benchmarks-website: add opt-in window chip with load/retry states (PR-5.0.9)

Signed-off-by: "Connor Tsui" <connor@spiraldb.com>
EOF
```

---

### Task 4: Hover-dwell silent prefetch

A continuous ~600ms dwell on one card silently prefetches the full history at the mid-tier priority; `pointerleave` cancels a pending dwell; the chip reveals "load all N" immediately on hover (no fetch until the dwell).

**Files:**
- Modify: `benchmarks-website/web/components/Chart.tsx` — `onCardHoverStart()`/`onCardHoverEnd()` methods, card `pointerenter`/`pointerleave` listeners in the mount effect, and `destroy()` cleanup.
- Test: `benchmarks-website/web/components/Chart.loading.test.tsx` (append).

- [ ] **Step 1: Write the failing tests (append)**

Add inside the same `describe` block. These use fake timers for the dwell window:

```tsx
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

  it('hover reveals the "load all N" action without fetching', async () => {
    const chip = await renderOpenGroup();
    const card = container.querySelector('.chart-card') as HTMLElement;
    card.dispatchEvent(new Event('pointerenter'));
    expect(chip?.textContent).toBe('load all 3,572');
    expect(fetchCalls.some((u) => u.includes('n=all'))).toBe(false);
    card.dispatchEvent(new Event('pointerleave'));
    expect(chip?.textContent).toBe('latest 100 of 3,572');
  });
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx`
Expected: FAIL — no hover handlers, so the chip text never changes and no dwell fetch fires.

- [ ] **Step 3: Add the hover-handler methods**

Add to `ChartController`, after `onWindowChipClick()`:

```ts
  /** Pointer resting on the card: reveal the chip's action immediately and arm
   * the dwell-prefetch timer. Only a deliberate dwell (not a sweep) fetches. */
  onCardHoverStart(): void {
    const state = this.state;
    if (state.disposed) {
      return;
    }
    state.hovering = true;
    this.syncWindowChip();
    if (state.fullLoaded || state.fullFetchPending || state.hoverDwellTimer !== null) {
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
    state.hovering = false;
    if (state.hoverDwellTimer !== null) {
      clearTimeout(state.hoverDwellTimer);
      state.hoverDwellTimer = null;
    }
    this.syncWindowChip();
  }
```

- [ ] **Step 4: Wire the card pointer listeners in the mount effect**

In the mount effect, just after `const cleanups: (() => void)[] = [];` (~line 1422), add the card-level hover listeners (these attach on both the landing and permalink pages; on a complete permalink chart the dwell fetch is a harmless no-op):

```ts
    const onCardEnter = (): void => controller.onCardHoverStart();
    const onCardLeave = (): void => controller.onCardHoverEnd();
    card.addEventListener('pointerenter', onCardEnter);
    card.addEventListener('pointerleave', onCardLeave);
    cleanups.push(() => {
      card.removeEventListener('pointerenter', onCardEnter);
      card.removeEventListener('pointerleave', onCardLeave);
    });
```

(`card` is already in scope as `const card = cardRef.current;` earlier in the effect; the effect already early-returns when it is null.)

- [ ] **Step 5: Cancel the dwell timer on destroy**

In `destroy()`, before `this.state.chart?.destroy();`, add:

```ts
    if (this.state.hoverDwellTimer !== null) {
      clearTimeout(this.state.hoverDwellTimer);
      this.state.hoverDwellTimer = null;
    }
```

- [ ] **Step 6: Run the tests, confirm they pass**

Run: `cd benchmarks-website/web && npx vitest run components/Chart.loading.test.tsx`
Expected: PASS — dwell fires only after the threshold, `pointerleave` cancels, hover reveals "load all N" with no fetch.

- [ ] **Step 7: Run the full web suite**

Run: `cd benchmarks-website/web && npm test`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add benchmarks-website/web/components/Chart.tsx benchmarks-website/web/components/Chart.loading.test.tsx
git commit -F - <<'EOF'
benchmarks-website: hover-dwell silent full-history prefetch (PR-5.0.9)

Signed-off-by: "Connor Tsui" <connor@spiraldb.com>
EOF
```

---

### Task 5: Lengthen the CDN stale-while-revalidate window

The site is low-traffic, so the 5-minute SWR window usually lapses between visits; lengthen SWR to a day so repeat visits serve from the CDN instantly while data (which lands a few times a day) revalidates in the background. `s-maxage` stays at 300.

**Files:**
- Modify: `benchmarks-website/web/lib/cache.ts` (the `READ_API_CACHE_CONTROL` value + its docstring)
- Modify: `benchmarks-website/web/vercel.json` (two `Vercel-CDN-Cache-Control` values)
- Create: `benchmarks-website/web/lib/cache.test.ts`

- [ ] **Step 1: Write the failing test (new file)**

Create `lib/cache.test.ts`:

```ts
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it } from 'vitest';

import { READ_API_CACHE_CONTROL } from './cache';

describe('READ_API_CACHE_CONTROL', () => {
  it('keeps a 5-minute fresh window but allows day-scale stale-while-revalidate', () => {
    expect(READ_API_CACHE_CONTROL).toContain('s-maxage=300');
    expect(READ_API_CACHE_CONTROL).toContain('stale-while-revalidate=86400');
  });
});
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `cd benchmarks-website/web && npx vitest run lib/cache.test.ts`
Expected: FAIL — current value has `stale-while-revalidate=300`.

- [ ] **Step 3: Update `cache.ts`**

Change the constant:

```ts
export const READ_API_CACHE_CONTROL = 'public, s-maxage=300, stale-while-revalidate=86400';
```

And update the docstring sentence `... and may serve it stale for up to another five minutes while refreshing in the background.` to:

```
 * the FULL request URL (so each `?n=` window is its own cache entry) for five
 * minutes, matching v2's S3 refresh cadence, and may serve it stale for up to a
 * day while it revalidates in the background — the site is low-traffic, so the
 * longer stale window keeps repeat visits on the CDN instead of paying a cold
 * function start. Error responses (400/404/500) deliberately omit this header so
 * they are never CDN-cached.
```

(Adjust surrounding lines so the comment reads cleanly and stays within the 100-column limit; keep the existing sentence about `revalidate = 300` below it unchanged.)

- [ ] **Step 4: Update `vercel.json`**

Change BOTH `Vercel-CDN-Cache-Control` values (the `/` rule and the `/chart/:slug` rule) from:

```
"value": "max-age=300, stale-while-revalidate=300"
```
to:
```
"value": "max-age=300, stale-while-revalidate=86400"
```

- [ ] **Step 5: Run the test + the existing header tests, confirm pass**

Run: `cd benchmarks-website/web && npx vitest run lib/cache.test.ts lib/queries.test.ts lib/groups.test.ts`
Expected: PASS — the `queries`/`groups` tests compare against the `READ_API_CACHE_CONTROL` constant (not a literal), so they track the new value automatically.

- [ ] **Step 6: Commit**

```bash
git add benchmarks-website/web/lib/cache.ts benchmarks-website/web/vercel.json benchmarks-website/web/lib/cache.test.ts
git commit -F - <<'EOF'
benchmarks-website: extend CDN stale-while-revalidate to one day (PR-5.0.9)

Signed-off-by: "Connor Tsui" <connor@spiraldb.com>
EOF
```

---

### Task 6: Full verification + type/build/lint/format gates

**Files:** none (verification only; one commit if formatter rewrites anything).

- [ ] **Step 1: Type-check**

Run: `cd benchmarks-website/web && npx tsc --noEmit`
Expected: no errors. (If a test payload helper field mismatches `ChartResponse`/`CommitPoint`, fix the helper to match the wire shape.)

- [ ] **Step 2: Full unit suite**

Run: `cd benchmarks-website/web && npm test`
Expected: all suites green (the prior 214 plus the new `Chart.loading` + `cache` + `chart-format` constant tests).

- [ ] **Step 3: Production build**

Run: `cd benchmarks-website/web && npm run build`
Expected: `next build` succeeds (no DB needed; the cache header changes do not affect build-time rendering).

- [ ] **Step 4: Lint**

Run: `cd benchmarks-website/web && npm run lint`
Expected: clean. Resolve any eslint findings on the touched files.

- [ ] **Step 5: Format check (and fix if needed)**

Run: `cd benchmarks-website/web && npm run format:check`
If it reports diffs: `npm run format`, then re-run `format:check` to confirm clean.

- [ ] **Step 6: Commit any formatter changes**

```bash
git add -A benchmarks-website/web
git commit -F - <<'EOF'
benchmarks-website: prettier/format pass for PR-5.0.9

Signed-off-by: "Connor Tsui" <connor@spiraldb.com>
EOF
```

(If steps 1–5 produced no changes, skip this commit.)

---

## Self-review notes (author check against the design doc)

- **Design item 1 (remove warmup):** Task 2, with the discriminating "zero `fullHistoryQueue.schedule`" test.
- **Design item 2 (window chip + state machine windowed/loading/complete/error):** Task 3; born-complete charts show no chip.
- **Design item 3 (hover dwell, staged; pointerleave cancels; immediate action reveal):** Task 4, mid-tier `HOVER_PREFETCH_PRIORITY`, `HOVER_DWELL_MS=600`.
- **Design item 4 (interaction promotion unchanged):** untouched; the `rangeTouchesUnloadedHistory` → `ensureFullHistory(INTERACTION_FULL_PRIORITY)` call at ~line 848 now also drives the chip for free.
- **Design item 5 (stale-while-revalidate):** Task 5 — note the design predates the existing `stale-while-revalidate=300`, so this is a `300 → 86400` bump in `cache.ts` + `vercel.json`, not a new directive.
- **Acceptance — no `?n=all` without intent; ~1MB group open; chip signals state with retry; jank-free fill-in (virtual axis untouched); CDN SWR present:** covered by Tasks 2–5 + manual post-deploy verification (network profile of a tpch group open) noted in the design.
- **Out of scope (NOT in this plan):** viewport hydration, server-side `?n=all` downsampling, any visual/layout redesign, cold-start keep-warm. Matches the design doc.
