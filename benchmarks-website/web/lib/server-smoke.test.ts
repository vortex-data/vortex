// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { StartedPostgreSqlContainer } from '@testcontainers/postgresql';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { getPool, resetPool } from './db';
import { chartKeyToSlug, groupKeyToSlug, type ChartKey, type GroupKey } from './slug';
import { dockerAvailable, seedChartFixture, startBenchContainer } from './test-harness';

// End-to-end smoke: boot the REAL production server (`next start` over the
// existing `.next` build) against a seeded testcontainers Postgres and assert
// the server-rendered landing page, chart permalink page, and chart API all
// serve the fixture data. This is the closest automated stand-in for "open the
// site and look at it"; the per-function behavior is covered by the unit and
// markup tests.
//
// Skips when Docker is unavailable OR when no `.next` production build exists
// (`pnpm build` first); both are environment, not code, conditions.

const WEB_ROOT = fileURLToPath(new URL('..', import.meta.url));
const BUILD_PRESENT = existsSync(`${WEB_ROOT}/.next/BUILD_ID`);
const PORT = 4319;
const BASE = `http://127.0.0.1:${PORT}`;

const QUERY_Q1: ChartKey = {
  k: 'QueryMeasurement',
  dataset: 'tpch',
  dataset_variant: null,
  scale_factor: '1',
  storage: 'nvme',
  query_idx: 1,
};

// The fixture seeds tpch/nvme/sf=1 query_measurements, which the read layer
// groups into the TPC-H (NVMe) (SF=1) QueryGroup. The cached default-window
// path for both /api/group/{slug} and /api/chart/{slug} routes through
// unstable_cache, so these slugs exercise the Data Cache path under the real
// Next runtime.
const TPCH_GROUP: GroupKey = {
  k: 'QueryGroup',
  dataset: 'tpch',
  dataset_variant: null,
  scale_factor: '1',
  storage: 'nvme',
};

async function waitForServer(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown = null;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url);
      if (r.ok) {
        return;
      }
      lastErr = new Error(`HTTP ${r.status}`);
    } catch (err) {
      lastErr = err;
    }
    await new Promise((res) => setTimeout(res, 250));
  }
  throw new Error(`server did not become ready at ${url}: ${String(lastErr)}`);
}

describe.skipIf(!dockerAvailable() || !BUILD_PRESENT)(
  'production-server smoke (next start + seeded Postgres)',
  () => {
    let container: StartedPostgreSqlContainer;
    let server: ChildProcess | null = null;

    beforeAll(async () => {
      container = await startBenchContainer();
      await seedChartFixture(getPool());
      server = spawn('node_modules/.bin/next', ['start', '-p', String(PORT)], {
        cwd: WEB_ROOT,
        // `startBenchContainer` exported the BENCH_DB_* connection env on
        // process.env; the server child inherits it.
        env: { ...process.env, NODE_ENV: 'production' },
        stdio: 'ignore',
      });
      await waitForServer(`${BASE}/api/health`, 60_000);
    });

    afterAll(async () => {
      server?.kill('SIGTERM');
      await resetPool();
      await container.stop();
    });

    it('serves the landing page with header chrome, group sections, and chart islands', async () => {
      const html = await (await fetch(`${BASE}/`)).text();
      // Header chrome (PR-4.4.b islands, server-rendered).
      expect(html).toContain('data-role="nav-mobile-toggle"');
      expect(html).toContain('data-role="theme-toggle"');
      expect(html).toContain('data-action="expand-all"');
      expect(html).toContain('data-role="global-filter-bar"');
      // The fixture's engines surface as filter chips via collectFilterUniverse.
      expect(html).toContain('data-value="datafusion"');
      expect(html).toContain('data-value="duckdb"');
      // Group section + per-group toolbar + chart-card islands.
      expect(html).toContain('class="group-details"');
      expect(html).toContain('data-role="group-toolbar"');
      expect(html).toContain('data-role="scope-slider"');
      expect(html).toContain('data-role="range-strip"');
      // The pre-paint theme bootstrap is inlined in <head>.
      expect(html).toContain('localStorage.getItem("bench-theme")');
    });

    it('serves the chart permalink page with the v3 title and meta line', async () => {
      const slug = chartKeyToSlug(QUERY_Q1);
      const raw = await (await fetch(`${BASE}/chart/${slug}`)).text();
      // React's streaming SSR separates adjacent text expressions with
      // `<!-- -->` comment nodes; strip them so the assertions read the
      // user-visible text.
      const html = raw.replace(/<!--.*?-->/g, '');
      expect(html).toContain('tpch sf=1 Q1 [nvme] - Vortex Benchmarks');
      expect(html).toContain('class="chart-meta"');
      expect(html).toContain('unit: ');
      expect(html).toContain('2 series');
      expect(html).toContain('3 commits');
      expect(html).toContain(`data-chart-slug="${slug}"`);
    });

    it('returns 404 HTML for an unknown chart slug', async () => {
      const r = await fetch(`${BASE}/chart/ra.unknown-slug`);
      expect(r.status).toBe(404);
    });

    it('serves the chart API the islands fetch from', async () => {
      const slug = chartKeyToSlug(QUERY_Q1);
      const r = await fetch(`${BASE}/api/chart/${slug}?n=all`);
      expect(r.status).toBe(200);
      const payload = (await r.json()) as {
        display_name: string;
        commits: unknown[];
        history: { complete: boolean };
      };
      expect(payload.display_name).toBe('tpch sf=1 Q1 [nvme]');
      expect(payload.commits).toHaveLength(3);
      expect(payload.history.complete).toBe(true);
    });

    // These two tests exercise the real unstable_cache path under the Next.js
    // production runtime (force-dynamic + real incrementalCache). The ?n=all
    // tests above use the direct query path; omitting ?n (or sending ?n=100)
    // routes through cachedDefaultGroupCharts / cachedDefaultChartPayload,
    // which call unstable_cache. If unstable_cache runs outside a valid request
    // context it throws "Invariant: incrementalCache missing"; a 200 here proves
    // the cached path is wired correctly under next start.
    it('serves the default-window group bundle via the Data Cache path', async () => {
      const slug = groupKeyToSlug(TPCH_GROUP);
      const r = await fetch(`${BASE}/api/group/${slug}`);
      expect(r.status).toBe(200);
      const payload = (await r.json()) as {
        name: string;
        charts: unknown[];
      };
      expect(typeof payload.name).toBe('string');
      expect(payload.name).toContain('TPC-H');
      expect(Array.isArray(payload.charts)).toBe(true);
      expect(payload.charts.length).toBeGreaterThan(0);
    });

    it('serves the default-window chart payload via the Data Cache path', async () => {
      const slug = chartKeyToSlug(QUERY_Q1);
      // No ?n= parameter: the route uses cachedDefaultChartPayload, which calls
      // unstable_cache and exercises the real Data Cache under next start.
      const r = await fetch(`${BASE}/api/chart/${slug}`);
      expect(r.status).toBe(200);
      const payload = (await r.json()) as {
        display_name: string;
        commits: unknown[];
      };
      expect(payload.display_name).toBe('tpch sf=1 Q1 [nvme]');
      expect(Array.isArray(payload.commits)).toBe(true);
      expect(payload.commits.length).toBeGreaterThan(0);
    });
  },
);
