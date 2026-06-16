// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment jsdom

import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Chart } from '@/components/Chart';
import { resetGroup, resetPayloadCache, setGroupY } from '@/lib/chart-store';

// Chart.js never actually constructs in this suite; the loader resolves to a
// throwing stub so any unexpected construction fails loudly. The lifecycle
// under test ends at the fetch layer.
vi.mock('@/lib/chart-js', () => ({
  loadChartJs: () =>
    Promise.resolve(
      class StubChart {
        constructor() {
          throw new Error('unexpected Chart.js construction in lifecycle test');
        }
      },
    ),
}));

/**
 * Regression pin for the StrictMode disposal bug found by the PR-4.4.b cycle-1
 * review: the chart controller was created once per component instance and
 * `destroy()` latched it permanently, so React StrictMode's dev-mode effect
 * replay (mount, cleanup, remount) left every island inert; no fetch ever
 * fired under `next dev`. The fix creates a fresh controller per effect mount,
 * so the replayed mount must still issue the initial `?n=100` fetch when its
 * group is open.
 */
describe('Chart island lifecycle (StrictMode effect replay)', () => {
  let container: HTMLElement;
  let root: Root | null = null;
  let fetchCalls: string[];
  let signals: AbortSignal[];

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    fetchCalls = [];
    signals = [];
    resetPayloadCache();
    vi.stubGlobal('fetch', (url: string | URL, init?: { signal?: AbortSignal }) => {
      fetchCalls.push(String(url));
      if (init?.signal) {
        signals.push(init.signal);
      }
      // Park the promise (rejecting only on abort): the assertion is that the
      // fetch was ISSUED; the construction path is exercised elsewhere.
      return new Promise((_res, reject) => {
        init?.signal?.addEventListener('abort', () =>
          reject(init.signal?.reason ?? new DOMException('Aborted', 'AbortError')),
        );
      });
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
    resetPayloadCache();
  });

  it('still fetches after the StrictMode mount/cleanup/remount replay', async () => {
    // The island finds its group context through the DOM, so render inside the
    // open-disclosure structure GroupSection produces.
    container.innerHTML =
      '<section class="group-details">' +
      '<details class="group-disclosure" open><summary class="group-summary">g</summary></details>' +
      '<div class="chart-grid"><div id="mount"></div></div>' +
      '</section>';
    const mount = container.querySelector('#mount') as HTMLElement;
    root = createRoot(mount);
    await act(async () => {
      root?.render(
        <StrictMode>
          <Chart slug="ra.eyJhIjoxfQ" name="gnomad" index={0} groupSlug="random_access" />
        </StrictMode>,
      );
    });
    // StrictMode ran the mount effect twice (mount, cleanup, remount). On the
    // landing page (PR-5.0.97) the initial fetch a group-open island issues is
    // the eager per-group bundle (`/api/group/{slug}?n=100`); the per-chart
    // `/api/chart` fetch is the fallback the IntersectionObserver gates. The
    // bundle is GROUP-scoped, not island-scoped, so a per-island unmount
    // (`destroy()`) does NOT abort it; only a group close does. The replayed
    // mount therefore finds the still-in-flight bundle and DEDUPES onto it,
    // collapsing the StrictMode replay to exactly ONE bundle fetch. That single
    // fetch staying alive (not aborted) is what hydrates the chart: the pre-fix
    // latched controller issued ZERO fetches on remount and left the island
    // permanently inert (the blank-charts-in-`next dev` regression this pins),
    // whereas here the survivor keeps the bundle in flight.
    const bundleFetches = fetchCalls.filter(
      (u) => u.includes('/api/group/') && u.includes('n=100'),
    );
    expect(bundleFetches).toHaveLength(1);
    expect(signals).toHaveLength(1);
    expect(signals[0].aborted).toBe(false);
  });

  it('replays a pre-existing group-Y override on mount (store outlives mounts)', async () => {
    // Regression pin for the cycle-2 review finding: the group store is
    // module-scoped and outlives mounts, the group-Y broadcast effect can run
    // while no controller exists yet, and a freshly mounted island must
    // therefore replay the store's current Y override itself; pre-fix it
    // constructed on the default linear scale despite an active `log`
    // override. The Y-button highlight is the observable: it tracks the
    // controller's applied scale through React state.
    const groupSlug = 'lifecycle-replay-group';
    setGroupY(groupSlug, 'log');
    try {
      container.innerHTML =
        '<section class="group-details">' +
        '<details class="group-disclosure"><summary class="group-summary">g</summary></details>' +
        '<div class="chart-grid"><div id="mount"></div></div>' +
        '</section>';
      const mount = container.querySelector('#mount') as HTMLElement;
      root = createRoot(mount);
      await act(async () => {
        root?.render(
          <StrictMode>
            <Chart slug="ra.eyJhIjoxfQ" name="gnomad" index={0} groupSlug={groupSlug} />
          </StrictMode>,
        );
      });
      const logBtn = container.querySelector('button[data-y="log"]');
      expect(logBtn?.className).toContain('toolbar-btn--active');
    } finally {
      resetGroup(groupSlug);
    }
  });
});
