// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';

// The route handlers route the default window through `unstable_cache`
// (`lib/data-cache.ts`), which needs Next's request/build `incrementalCache`
// context that plain vitest does not provide. These tests verify the route plus
// query behavior against the real testcontainer, not the cache layer (covered by
// `lib/data-cache.test.ts`), so make the cache wrapper a transparent pass-through.
vi.mock('next/cache', () => ({
  unstable_cache: (fn: (...args: unknown[]) => unknown) => fn,
  revalidateTag: () => {},
}));

import type { StartedPostgreSqlContainer } from '@testcontainers/postgresql';
import { READ_API_CACHE_CONTROL } from './cache';
import { chartPayload, collectFilterUniverse, compareGroupSortKey } from './queries';
import { chartKeyToSlug, type ChartKey } from './slug';
import {
  commitUrl,
  dockerAvailable,
  seedChartFixture,
  startBenchContainer,
  TREE_SHA,
} from './test-harness';
import { parseCommitWindow } from './window';
import { getPool, resetPool } from './db';
import { GET } from '@/app/api/chart/[slug]/route';

const QUERY_Q1: ChartKey = {
  k: 'QueryMeasurement',
  dataset: 'tpch',
  dataset_variant: null,
  scale_factor: '1',
  storage: 'nvme',
  query_idx: 1,
};

describe.skipIf(!dockerAvailable())('chartPayload (testcontainers Postgres)', () => {
  let container: StartedPostgreSqlContainer;

  beforeAll(async () => {
    container = await startBenchContainer();
    await seedChartFixture(getPool());
  });

  afterAll(async () => {
    await resetPool();
    await container.stop();
  });

  it('assembles the TPC-H Q1 chart semantically equivalent to the v3 Axum payload', async () => {
    // The expected object is the v3 server's golden snapshot
    // (`server/tests/snapshots/chart_page_query.snap`), with numeric values as
    // JS numbers (serde renders `1000000.0`; JS renders `1000000`; the same
    // number to every consumer, per the recorded semantic-equivalence
    // decision). Timestamps reproduce the DuckDB `CAST(... AS VARCHAR)` text.
    const payload = await chartPayload(QUERY_Q1, parseCommitWindow(null));
    expect(payload).toEqual({
      display_name: 'tpch sf=1 Q1 [nvme]',
      unit_kind: 'time_ns',
      history: { total_commits: 3, start_index: 0, loaded_commits: 3, complete: true },
      commits: [
        {
          sha: '1'.repeat(40),
          timestamp: '2026-04-23 12:00:00+00',
          message: 'first commit',
          url: commitUrl('1'.repeat(40)),
        },
        {
          sha: '2'.repeat(40),
          timestamp: '2026-04-24 12:00:00+00',
          message: 'second commit',
          url: commitUrl('2'.repeat(40)),
        },
        {
          sha: '3'.repeat(40),
          timestamp: '2026-04-25 12:00:00+00',
          message: 'third commit',
          url: commitUrl('3'.repeat(40)),
        },
      ],
      series: {
        'datafusion:vortex-file-compressed': [1_000_000, 1_050_000, 1_100_000],
        'duckdb:parquet': [800_000, 850_000, 900_000],
      },
      series_meta: {
        'datafusion:vortex-file-compressed': {
          engine: 'datafusion',
          format: 'vortex-file-compressed',
        },
        'duckdb:parquet': { engine: 'duckdb', format: 'parquet' },
      },
    });
  });

  it('never puts measurement_id on the wire', async () => {
    const payload = await chartPayload(QUERY_Q1, parseCommitWindow(null));
    expect(JSON.stringify(payload)).not.toContain('measurement_id');
  });

  it('collects the filter universe sorted, excluding vector-search flavors', async () => {
    const universe = await collectFilterUniverse();
    // Engines come from `query_measurements` only; formats union the four
    // format-bearing fact tables. The fixture's `vortex-turboquant` flavor
    // (vector_search_runs) must NOT leak in as a format.
    expect(universe).toEqual({
      engines: ['datafusion', 'duckdb'],
      formats: ['parquet', 'vortex-file-compressed'],
    });
  });

  it('declares the base unit_kind per fact-table family', async () => {
    const cases: ReadonlyArray<readonly [ChartKey, string]> = [
      [QUERY_Q1, 'time_ns'],
      [{ k: 'CompressionTime', dataset: 'tpch-lineitem', dataset_variant: null }, 'time_ns'],
      [{ k: 'CompressionSize', dataset: 'tpch-lineitem', dataset_variant: null }, 'bytes'],
      [{ k: 'RandomAccess', dataset: 'taxi' }, 'time_ns'],
      [
        { k: 'VectorSearch', dataset: 'cohere-large-10m', layout: 'partitioned', threshold: 0.75 },
        'time_ns',
      ],
    ];
    for (const [key, expected] of cases) {
      const payload = await chartPayload(key, parseCommitWindow(null));
      expect(payload?.unit_kind).toBe(expected);
    }
  });

  it('tags query series with engine + format and compression-time series with format only', async () => {
    const query = await chartPayload(QUERY_Q1, parseCommitWindow(null));
    expect(query?.series_meta?.['datafusion:vortex-file-compressed']).toEqual({
      engine: 'datafusion',
      format: 'vortex-file-compressed',
    });

    const compTime = await chartPayload(
      { k: 'CompressionTime', dataset: 'tpch-lineitem', dataset_variant: null },
      parseCommitWindow(null),
    );
    // The series key is `format:op`; the tag carries only the format (no engine).
    expect(compTime?.series_meta?.['vortex-file-compressed:encode']).toEqual({
      format: 'vortex-file-compressed',
    });
    expect(compTime?.series_meta?.['vortex-file-compressed:encode'].engine).toBeUndefined();
  });

  it('omits series_meta entirely for vector-search charts (no engine/format)', async () => {
    const payload = await chartPayload(
      { k: 'VectorSearch', dataset: 'cohere-large-10m', layout: 'partitioned', threshold: 0.75 },
      parseCommitWindow(null),
    );
    expect(payload?.series_meta).toBeUndefined();
    expect(payload?.display_name).toBe('cohere-large-10m / partitioned (threshold=0.75)');
    expect(payload?.series['vortex-turboquant']).toEqual([7_000, 57_000, 107_000]);
  });

  it('compression-size series are keyed by format and carry byte values', async () => {
    const payload = await chartPayload(
      { k: 'CompressionSize', dataset: 'tpch-lineitem', dataset_variant: null },
      parseCommitWindow(null),
    );
    expect(payload?.series).toEqual({
      parquet: [8_000, 108_000, 208_000],
      'vortex-file-compressed': [4_000, 54_000, 104_000],
    });
  });

  it('caps commits with ?n and stays unbounded with ?n=all', async () => {
    const one = await chartPayload(QUERY_Q1, parseCommitWindow('1'));
    expect(one?.commits).toHaveLength(1);
    // ?n keeps the most recent commit (oldest-first array, so the last one).
    expect(one?.commits[0].sha).toBe('3'.repeat(40));

    const all = await chartPayload(QUERY_Q1, parseCommitWindow('all'));
    expect(all?.commits).toHaveLength(3);

    const dflt = await chartPayload(QUERY_Q1, parseCommitWindow(null));
    expect(dflt?.commits).toHaveLength(3);
  });

  it('?n=2 selects exactly the two newest commits with correct values (commit_timestamp window)', async () => {
    // Pins the bounded-window result for a window smaller than the commit count,
    // the boundary that the `commit_timestamp` cutoff and `commit_sha` tie-trim
    // together govern. Must hold identically before and after the recency-filter
    // refactor.
    const payload = await chartPayload(QUERY_Q1, parseCommitWindow('2'));
    expect(payload).toEqual({
      display_name: 'tpch sf=1 Q1 [nvme]',
      unit_kind: 'time_ns',
      history: { total_commits: 3, start_index: 1, loaded_commits: 2, complete: false },
      commits: [
        {
          sha: '2'.repeat(40),
          timestamp: '2026-04-24 12:00:00+00',
          message: 'second commit',
          url: commitUrl('2'.repeat(40)),
        },
        {
          sha: '3'.repeat(40),
          timestamp: '2026-04-25 12:00:00+00',
          message: 'third commit',
          url: commitUrl('3'.repeat(40)),
        },
      ],
      series: {
        'datafusion:vortex-file-compressed': [1_050_000, 1_100_000],
        'duckdb:parquet': [850_000, 900_000],
      },
      series_meta: {
        'datafusion:vortex-file-compressed': {
          engine: 'datafusion',
          format: 'vortex-file-compressed',
        },
        'duckdb:parquet': { engine: 'duckdb', format: 'parquet' },
      },
    });
  });

  it('returns null for a well-formed slug whose chart has no rows', async () => {
    const missing = await chartPayload(
      {
        k: 'QueryMeasurement',
        dataset: 'missing-dataset',
        dataset_variant: null,
        scale_factor: null,
        storage: 'nvme',
        query_idx: 99,
      },
      parseCommitWindow(null),
    );
    expect(missing).toBeNull();
  });

  it('GET /api/chart/{slug} maps malformed/missing/populated slugs to 400/404/200', async () => {
    // Error envelopes match the Axum server (`error.rs`): a machine-readable
    // `error` code plus a prose `message`. Only the 200 payload carries the CDN
    // `Cache-Control` header; error responses must never be CDN-cached.
    const bad = await GET(new Request('http://localhost/api/chart/not-a-slug'), {
      params: Promise.resolve({ slug: 'not-a-slug' }),
    });
    expect(bad.status).toBe(400);
    expect(await bad.json()).toEqual({ error: 'bad_request', message: 'invalid chart slug' });
    expect(bad.headers.get('cache-control')).toBeNull();

    const missingSlug = chartKeyToSlug({
      k: 'QueryMeasurement',
      dataset: 'missing-dataset',
      dataset_variant: null,
      scale_factor: null,
      storage: 'nvme',
      query_idx: 99,
    });
    const missing = await GET(new Request(`http://localhost/api/chart/${missingSlug}`), {
      params: Promise.resolve({ slug: missingSlug }),
    });
    expect(missing.status).toBe(404);
    expect(await missing.json()).toEqual({ error: 'not_found', message: 'chart not found' });
    expect(missing.headers.get('cache-control')).toBeNull();

    const slug = chartKeyToSlug(QUERY_Q1);
    const ok = await GET(new Request(`http://localhost/api/chart/${slug}`), {
      params: Promise.resolve({ slug }),
    });
    expect(ok.status).toBe(200);
    expect(ok.headers.get('cache-control')).toBe(READ_API_CACHE_CONTROL);
    const body = (await ok.json()) as { display_name: string; commits: unknown[] };
    expect(body.display_name).toBe('tpch sf=1 Q1 [nvme]');
    expect(body.commits).toHaveLength(3);
  });

  it('GET /api/chart/{slug}?n=1 narrows the commit window', async () => {
    const slug = chartKeyToSlug(QUERY_Q1);
    const res = await GET(new Request(`http://localhost/api/chart/${slug}?n=1`), {
      params: Promise.resolve({ slug }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { commits: unknown[] };
    expect(body.commits).toHaveLength(1);
  });

  it('GET /api/chart/{slug} returns 200 (not 500) for ?n=all and a malformed ?n', async () => {
    // With only 3 fixture commits this cannot discriminate WHICH window the
    // route threads (all/default/malformed all return 3 commits); it pins the
    // 200-path only. The discriminating all-vs-default route assertion lives in
    // the 125-commit history-placement suite below.
    const slug = chartKeyToSlug(QUERY_Q1);
    const all = await GET(new Request(`http://localhost/api/chart/${slug}?n=all`), {
      params: Promise.resolve({ slug }),
    });
    expect(all.status).toBe(200);
    expect(((await all.json()) as { commits: unknown[] }).commits).toHaveLength(3);

    // A malformed ?n falls back to the default window (the 3-commit fixture fits).
    const bad = await GET(new Request(`http://localhost/api/chart/${slug}?n=banana`), {
      params: Promise.resolve({ slug }),
    });
    expect(bad.status).toBe(200);
    expect(((await bad.json()) as { commits: unknown[] }).commits).toHaveLength(3);
  });

  it('tags random-access and compression-size series with format only (no engine)', async () => {
    // The Rust oracle calls `tag(&format, None, Some(&format))` for these
    // families, so each series carries a format tag and no engine.
    const ra = await chartPayload({ k: 'RandomAccess', dataset: 'taxi' }, parseCommitWindow(null));
    expect(ra?.series_meta?.['vortex-file-compressed']).toEqual({
      format: 'vortex-file-compressed',
    });
    expect(ra?.series_meta?.parquet).toEqual({ format: 'parquet' });
    expect(ra?.series_meta?.parquet.engine).toBeUndefined();

    const cs = await chartPayload(
      { k: 'CompressionSize', dataset: 'tpch-lineitem', dataset_variant: null },
      parseCommitWindow(null),
    );
    expect(cs?.series_meta?.['vortex-file-compressed']).toEqual({
      format: 'vortex-file-compressed',
    });
    expect(cs?.series_meta?.parquet).toEqual({ format: 'parquet' });
  });
});

// The product-visible group ordering (GROUP_ORDER + compareGroupSortKey) is
// exercised end-to-end only by the Docker-gated collectGroups test; this no-DB
// test pins the ordering directly so it stays verifiable without testcontainers.
describe('compareGroupSortKey canonical group ordering (no DB)', () => {
  it('places Random Access directly below PolarSignals Profiling', () => {
    const sorted = ['Random Access', 'PolarSignals Profiling', 'Clickbench'].sort(
      compareGroupSortKey,
    );
    expect(sorted).toEqual(['Clickbench', 'PolarSignals Profiling', 'Random Access']);
    // Adjacency: nothing sorts between PolarSignals Profiling and Random Access.
    const polar = sorted.indexOf('PolarSignals Profiling');
    expect(sorted[polar + 1]).toBe('Random Access');
  });

  it('sorts listed groups before unknown groups, unknowns alphabetically last', () => {
    const sorted = ['cohere-large-10m / partitioned', 'Random Access', 'Compression'].sort(
      compareGroupSortKey,
    );
    expect(sorted).toEqual(['Compression', 'Random Access', 'cohere-large-10m / partitioned']);
  });
});

// Slug-decode rejection short-circuits to a 400 before any DB call, so these
// run without Docker. The forged `query_idx` cases pin the i32-validation fix:
// a non-i32 query_idx must be a 400 malformed slug (matching the Rust serde i32
// path), not an unhandled 500 from the Postgres integer bind.
describe('GET /api/chart/[slug] input validation (no DB)', () => {
  function forgeSlug(payload: unknown): string {
    return `qm.${Buffer.from(JSON.stringify(payload), 'utf8').toString('base64url')}`;
  }
  const baseQuery = {
    k: 'QueryMeasurement',
    dataset: 'tpch',
    dataset_variant: null,
    scale_factor: '1',
    storage: 'nvme',
  };

  it('returns 400 for a structurally malformed slug', async () => {
    const slug = 'not-a-slug';
    const res = await GET(new Request(`http://localhost/api/chart/${slug}`), {
      params: Promise.resolve({ slug }),
    });
    expect(res.status).toBe(400);
  });

  it.each([1.5, 2_147_483_648, -2_147_483_649])(
    'returns 400 (not 500) for a forged non-i32 query_idx %s',
    async (queryIdx) => {
      const slug = forgeSlug({ ...baseQuery, query_idx: queryIdx });
      const res = await GET(new Request(`http://localhost/api/chart/${slug}`), {
        params: Promise.resolve({ slug }),
      });
      expect(res.status).toBe(400);
    },
  );
});

// A long synthetic history exercises the bounded-vs-full window placement math
// (`history.start_index` / `total_commits` / `complete`), mirroring
// `chart_api.rs::chart_api_reports_virtual_history_for_bounded_and_full_windows`.
describe.skipIf(!dockerAvailable())('chartPayload history placement (long synthetic run)', () => {
  let container: StartedPostgreSqlContainer;
  const N = 125;
  const RANDOM_ACCESS: ChartKey = { k: 'RandomAccess', dataset: 'taxi' };

  beforeAll(async () => {
    container = await startBenchContainer();
    const pool = getPool();
    let id = 0;
    for (let i = 0; i < N; i += 1) {
      const sha = i.toString(16).padStart(40, '0');
      const ts = `2025-01-01T${String(Math.floor(i / 60) % 24).padStart(2, '0')}:${String(
        i % 60,
      ).padStart(2, '0')}:00Z`;
      await pool.query(
        `INSERT INTO commits (commit_sha, timestamp, message, tree_sha, url)
         VALUES ($1, $2::timestamptz, 'synthetic', $3, $4)`,
        [sha, ts, TREE_SHA, commitUrl(sha)],
      );
      id += 1;
      await pool.query(
        `INSERT INTO random_access_times
           (measurement_id, commit_sha, dataset, format, value_ns, all_runtimes_ns)
         VALUES ($1, $2, 'taxi', 'vortex-file-compressed', $3, '{1}'::bigint[])`,
        [id, sha, 500 + i],
      );
    }
  });

  afterAll(async () => {
    await resetPool();
    await container.stop();
  });

  it('reports virtual history for the default bounded window', async () => {
    const bounded = await chartPayload(RANDOM_ACCESS, parseCommitWindow(null));
    expect(bounded?.commits).toHaveLength(100);
    expect(bounded?.history).toEqual({
      total_commits: 125,
      start_index: 25,
      loaded_commits: 100,
      complete: false,
    });
  });

  it('reports complete history for ?n=all', async () => {
    const all = await chartPayload(RANDOM_ACCESS, parseCommitWindow('all'));
    expect(all?.commits).toHaveLength(125);
    expect(all?.history).toEqual({
      total_commits: 125,
      start_index: 0,
      loaded_commits: 125,
      complete: true,
    });
  });

  it('GET /api/chart/{slug} threads ?n=all vs the default window (discriminating)', async () => {
    // 125 fixture commits > the default window of 100, so all-vs-default
    // produces different lengths at the route layer (unlike the 3-commit
    // fixture's 200-path smoke test above).
    const slug = chartKeyToSlug(RANDOM_ACCESS);
    const all = await GET(new Request(`http://localhost/api/chart/${slug}?n=all`), {
      params: Promise.resolve({ slug }),
    });
    expect(((await all.json()) as { commits: unknown[] }).commits).toHaveLength(125);

    const dflt = await GET(new Request(`http://localhost/api/chart/${slug}`), {
      params: Promise.resolve({ slug }),
    });
    expect(((await dflt.json()) as { commits: unknown[] }).commits).toHaveLength(100);
  });
});

// Tailored fixtures pinning the subtle seeded-commit-window invariants that the
// shared fixture (every commit has every row) cannot exercise: the null-gap
// x-axis seeding, pre-history exclusion, IS NOT DISTINCT FROM NULL-dim equality,
// and the commit_sha tie-break under identical timestamps.
describe.skipIf(!dockerAvailable())('chartPayload seeded-window semantics', () => {
  let container: StartedPostgreSqlContainer;
  // Distinct SHAs. C is pre-history purely by its earlier timestamp (its SHA
  // rank is irrelevant); the tie-break test relies only on E > D lexically.
  const C = 'c'.repeat(40);
  const A = 'a'.repeat(40);
  const B = 'b'.repeat(40);
  const D = 'd'.repeat(40);
  const E = 'e'.repeat(40);

  beforeAll(async () => {
    container = await startBenchContainer();
    const pool = getPool();
    // C is pre-history for the `qn`/`qnull` charts; D and E share a timestamp.
    const commits: ReadonlyArray<readonly [string, string]> = [
      [C, '2026-05-01T12:00:00Z'],
      [A, '2026-05-02T12:00:00Z'],
      [B, '2026-05-03T12:00:00Z'],
      [D, '2026-05-04T00:00:00Z'],
      [E, '2026-05-04T00:00:00Z'],
    ];
    for (const [sha, ts] of commits) {
      await pool.query(
        `INSERT INTO commits (commit_sha, timestamp, message, tree_sha, url)
         VALUES ($1, $2::timestamptz, 'm', $3, $4)`,
        [sha, ts, TREE_SHA, commitUrl(sha)],
      );
    }
    let id = 0;
    const mid = (): number => {
      id += 1;
      return id;
    };
    // `qn` Q1 chart: a single row on commit A only. B/D/E fall in its window
    // (timestamp >= A) with no row -> null slots; C is pre-history -> excluded.
    // `commit_timestamp` is the denormalized copy of `commits.timestamp` that
    // every write path populates (the migration-006 backfill + the ingest
    // upsert); the read path now seeds + windows on it, so the fixture must
    // stamp it exactly as the writers do (a NULL here yields an empty chart).
    await pool.query(
      `INSERT INTO query_measurements
         (measurement_id, commit_sha, dataset, dataset_variant, scale_factor,
          query_idx, storage, engine, format, value_ns, all_runtimes_ns, commit_timestamp)
       VALUES ($1, $2, 'qn', NULL, '1', 1, 'nvme', 'datafusion', 'vortex-file-compressed',
               1000, '{1}'::bigint[], (SELECT timestamp FROM commits WHERE commit_sha = $2))`,
      [mid(), A],
    );
    // `qnull` Q1 chart with a NULL scale_factor, row on commit A only.
    await pool.query(
      `INSERT INTO query_measurements
         (measurement_id, commit_sha, dataset, dataset_variant, scale_factor,
          query_idx, storage, engine, format, value_ns, all_runtimes_ns, commit_timestamp)
       VALUES ($1, $2, 'qnull', NULL, NULL, 1, 'nvme', 'datafusion', 'vortex-file-compressed',
               2000, '{1}'::bigint[], (SELECT timestamp FROM commits WHERE commit_sha = $2))`,
      [mid(), A],
    );
    // `tie` random-access chart: rows on D and E, which share a timestamp.
    for (const sha of [D, E]) {
      await pool.query(
        `INSERT INTO random_access_times
           (measurement_id, commit_sha, dataset, format, value_ns, all_runtimes_ns)
         VALUES ($1, $2, 'tie', 'parquet', 7, '{1}'::bigint[])`,
        [mid(), sha],
      );
    }
  });

  afterAll(async () => {
    await resetPool();
    await container.stop();
  });

  it('keeps in-window commits with no fact row as null gaps, and excludes pre-history', async () => {
    const payload = await chartPayload(
      {
        k: 'QueryMeasurement',
        dataset: 'qn',
        dataset_variant: null,
        scale_factor: '1',
        storage: 'nvme',
        query_idx: 1,
      },
      parseCommitWindow(null),
    );
    const shas = payload?.commits.map((c) => c.sha) ?? [];
    // Commit A has the row; B is in-window (timestamp >= A) but has no `qn` row.
    expect(shas).toContain(A);
    expect(shas).toContain(B);
    // Pre-history commit C (timestamp < A's earliest) is excluded from the x-axis.
    expect(shas).not.toContain(C);
    const series = payload?.series['datafusion:vortex-file-compressed'] ?? [];
    expect(series[shas.indexOf(A)]).toBe(1000);
    expect(series[shas.indexOf(B)]).toBeNull();
  });

  it('matches a NULL scale_factor via IS NOT DISTINCT FROM, and a wrong value misses', async () => {
    const matched = await chartPayload(
      {
        k: 'QueryMeasurement',
        dataset: 'qnull',
        dataset_variant: null,
        scale_factor: null,
        storage: 'nvme',
        query_idx: 1,
      },
      parseCommitWindow(null),
    );
    expect(matched).not.toBeNull();
    expect(matched?.series['datafusion:vortex-file-compressed']).toContain(2000);
    // A non-NULL scale_factor must NOT match the NULL row (proves the predicate is
    // IS NOT DISTINCT FROM, not `=`, which would silently return no rows -> a 404).
    const missed = await chartPayload(
      {
        k: 'QueryMeasurement',
        dataset: 'qnull',
        dataset_variant: null,
        scale_factor: '1',
        storage: 'nvme',
        query_idx: 1,
      },
      parseCommitWindow(null),
    );
    expect(missed).toBeNull();
  });

  it('breaks identical timestamps by commit_sha (DESC keeps the larger; oldest-first by ASC)', async () => {
    // ?n=1 keeps the most recent commit; with equal timestamps the tie-break is
    // commit_sha DESC, so E (e > d) is kept.
    const one = await chartPayload({ k: 'RandomAccess', dataset: 'tie' }, parseCommitWindow('1'));
    expect(one?.commits.map((c) => c.sha)).toEqual([E]);
    // The full window orders oldest-first by (timestamp ASC, commit_sha ASC).
    const all = await chartPayload({ k: 'RandomAccess', dataset: 'tie' }, parseCommitWindow('all'));
    expect(all?.commits.map((c) => c.sha)).toEqual([D, E]);
  });
});
