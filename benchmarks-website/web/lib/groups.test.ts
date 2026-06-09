// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import type { StartedPostgreSqlContainer } from '@testcontainers/postgresql';
import {
  collectGroupCharts,
  collectGroups,
  type GroupChartsResponse,
  type GroupsResponse,
} from './queries';
import { READ_API_CACHE_CONTROL } from './cache';
import { collectGroupSummary } from './summary';
import { chartKeyToSlug, groupKeyFromSlug, groupKeyToSlug } from './slug';
import {
  commitUrl,
  dockerAvailable,
  seedChartFixture,
  startBenchContainer,
  TREE_SHA,
} from './test-harness';
import { parseCommitWindow } from './window';
import { getPool, resetPool } from './db';
import { GET as getGroups } from '@/app/api/groups/route';
import { GET as getGroup } from '@/app/api/group/[slug]/route';

function expectDefined<T>(value: T | null | undefined, message: string): T {
  if (value === null || value === undefined) {
    throw new Error(`expected ${message} to be defined`);
  }
  return value;
}

// The canonical fixture's summary maths (ratio 2.0, geomean 0.5, the query
// missing-series penalty) reproduce the v2-contract values asserted by
// `server/tests/group_api.rs`; see `seedChartFixture` in `./test-harness`.
describe.skipIf(!dockerAvailable())(
  'collectGroups + group endpoints (testcontainers Postgres)',
  () => {
    let container: StartedPostgreSqlContainer;

    beforeAll(async () => {
      container = await startBenchContainer();
      await seedChartFixture(getPool());
    });

    afterAll(async () => {
      await resetPool();
      await container.stop();
    });

    it('orders groups by the canonical GROUP_ORDER, with unknown groups last', async () => {
      const groups = await collectGroups();
      expect(groups.map((g) => g.name)).toEqual([
        'Random Access',
        'Compression',
        'Compression Size',
        'TPC-H (NVMe) (SF=1)',
        'cohere-large-10m / partitioned',
      ]);
    });

    it('computes the random-access summary (ratio to fastest)', async () => {
      const groups = await collectGroups();
      const summary = expectDefined(
        groups.find((g) => g.name === 'Random Access')?.summary,
        'random access summary',
      );
      if (summary.type !== 'randomAccess') {
        throw new Error(`expected randomAccess summary, got ${summary.type}`);
      }
      expect(summary.title).toBe('Random Access Performance');
      expect(summary.rankings[0].name).toBe('vortex-file-compressed');
      expect(summary.rankings[1].name).toBe('parquet');
      expect(summary.rankings[0].ratio).toBeCloseTo(1.0, 6);
      expect(summary.rankings[1].ratio).toBeCloseTo(2.0, 6);
    });

    it('computes the compression throughput summary (geomean speedup)', async () => {
      const groups = await collectGroups();
      const summary = expectDefined(
        groups.find((g) => g.name === 'Compression')?.summary,
        'compression summary',
      );
      if (summary.type !== 'compression') {
        throw new Error(`expected compression summary, got ${summary.type}`);
      }
      expect(summary.compressRatio).toBeCloseTo(2.0, 6);
      expect(summary.decompressRatio).toBeCloseTo(2.0, 6);
      expect(summary.datasetCount).toBe(1);
      // The wire shape carries camelCase variant fields per `dto.rs`.
      const json = JSON.stringify(summary);
      expect(json).toContain('"compressRatio"');
      expect(json).toContain('"datasetCount"');
      expect(json).not.toContain('compress_ratio');
    });

    it('computes the compression-size summary (geomean size ratio)', async () => {
      const groups = await collectGroups();
      const summary = expectDefined(
        groups.find((g) => g.name === 'Compression Size')?.summary,
        'compression size summary',
      );
      if (summary.type !== 'compressionSize') {
        throw new Error(`expected compressionSize summary, got ${summary.type}`);
      }
      expect(summary.minRatio).toBeCloseTo(0.5, 6);
      expect(summary.meanRatio).toBeCloseTo(0.5, 6);
      expect(summary.maxRatio).toBeCloseTo(0.5, 6);
      expect(summary.datasetCount).toBe(1);
    });

    it('computes the query-benchmark summary with v2 missing-series penalty', async () => {
      const groups = await collectGroups();
      const summary = expectDefined(
        groups.find((g) => g.name === 'TPC-H (NVMe) (SF=1)')?.summary,
        'query summary',
      );
      if (summary.type !== 'queryBenchmark') {
        throw new Error(`expected queryBenchmark summary, got ${summary.type}`);
      }
      expect(summary.rankings[0].name).toBe('datafusion:vortex-file-compressed');
      expect(summary.rankings[1].name).toBe('duckdb:parquet');
      expect(summary.rankings[0].score).toBeLessThan(summary.rankings[1].score);
      // Exact geomean scores pin the penalty model (not just the ordering). At the
      // latest commit datafusion has Q1=1_100_000, Q2=700_000 and duckdb has only
      // Q1=900_000; bestByQuery is {Q1: 900_000, Q2: 700_000}. datafusion scores
      // sqrt((10+1_100_000)/(10+900_000) * 1) = sqrt(1100010/900010); duckdb is
      // missing Q2 so it takes the penalty max(900_000,300_000)*2 = 1_800_000,
      // scoring sqrt(1 * (10+1_800_000)/(10+700_000)) = sqrt(1800010/700010). The
      // penalty is what pushes duckdb (best raw Q1) below datafusion.
      expect(summary.rankings[0].score).toBeCloseTo(1.10554, 5);
      expect(summary.rankings[1].score).toBeCloseTo(1.60356, 5);
      // datafusion series has Q1 + Q2; duckdb series has Q1 only.
      expect(summary.rankings[0].totalRuntime).toBeCloseTo(1_800_000, 6);
      expect(JSON.stringify(summary)).toContain('"totalRuntime"');
    });

    it('attaches editorial descriptions and omits them where v2 has none', async () => {
      const groups = await collectGroups();
      expect(groups.find((g) => g.name === 'Random Access')?.description).toBe(
        'Tests performance of selecting arbitrary row indices from a file on NVMe storage',
      );
      expect(groups.find((g) => g.name === 'TPC-H (NVMe) (SF=1)')?.description).toBe(
        'TPC-H benchmark queries on local NVMe storage at SF=1 (~1GB of data)',
      );
      const vector = expectDefined(
        groups.find((g) => g.name === 'cohere-large-10m / partitioned'),
        'vector-search group',
      );
      // Vector-search groups carry neither a summary nor a description.
      expect(vector.summary).toBeUndefined();
      expect(vector.description).toBeUndefined();
    });

    it('GET /api/groups returns the canonical group list with summaries', async () => {
      const res = await getGroups();
      expect(res.status).toBe(200);
      expect(res.headers.get('cache-control')).toBe(READ_API_CACHE_CONTROL);
      const body = (await res.json()) as GroupsResponse;
      expect(body.groups.map((g) => g.name)).toEqual([
        'Random Access',
        'Compression',
        'Compression Size',
        'TPC-H (NVMe) (SF=1)',
        'cohere-large-10m / partitioned',
      ]);
    });

    it('collectGroupCharts inlines flattened chart payloads for one group', async () => {
      const slug = groupKeyToSlug({
        k: 'QueryGroup',
        dataset: 'tpch',
        dataset_variant: null,
        scale_factor: '1',
        storage: 'nvme',
      });
      const group = expectDefined(
        await collectGroupCharts(groupKeyFromSlug(slug), parseCommitWindow(null)),
        'group charts',
      );
      expect(group.name).toBe('TPC-H (NVMe) (SF=1)');
      expect(group.summary?.type).toBe('queryBenchmark');
      expect(group.description).toBe(
        'TPC-H benchmark queries on local NVMe storage at SF=1 (~1GB of data)',
      );
      expect(group.charts.map((c) => c.name)).toEqual(['Q1', 'Q2']);
      const q1 = group.charts[0];
      // The chart link's slug round-trips to the same `/api/chart` slug.
      expect(q1.slug).toBe(
        chartKeyToSlug({
          k: 'QueryMeasurement',
          dataset: 'tpch',
          dataset_variant: null,
          scale_factor: '1',
          storage: 'nvme',
          query_idx: 1,
        }),
      );
      // The ChartResponse payload is flattened in alongside `name` / `slug`.
      expect(q1.display_name).toBe('tpch sf=1 Q1 [nvme]');
      expect(q1.commits).toHaveLength(3);
      expect(q1.series['datafusion:vortex-file-compressed']).toEqual([
        1_000_000, 1_050_000, 1_100_000,
      ]);
    });

    it('GET /api/group/{slug} returns the group, honoring ?n', async () => {
      const slug = groupKeyToSlug({
        k: 'QueryGroup',
        dataset: 'tpch',
        dataset_variant: null,
        scale_factor: '1',
        storage: 'nvme',
      });
      const full = await getGroup(new Request(`http://localhost/api/group/${slug}`), {
        params: Promise.resolve({ slug }),
      });
      expect(full.status).toBe(200);
      expect(full.headers.get('cache-control')).toBe(READ_API_CACHE_CONTROL);
      const body = (await full.json()) as GroupChartsResponse;
      expect(body.summary?.type).toBe('queryBenchmark');
      expect(body.charts).toHaveLength(2);

      const windowed = await getGroup(new Request(`http://localhost/api/group/${slug}?n=1`), {
        params: Promise.resolve({ slug }),
      });
      expect(windowed.status).toBe(200);
      const windowedBody = (await windowed.json()) as GroupChartsResponse;
      for (const chart of windowedBody.charts) {
        expect(chart.commits).toHaveLength(1);
      }
    });

    it('GET /api/group/{slug} returns 404 for a well-formed slug with no data', async () => {
      const slug = groupKeyToSlug({
        k: 'QueryGroup',
        dataset: 'nonexistent',
        dataset_variant: null,
        scale_factor: null,
        storage: 'nvme',
      });
      const res = await getGroup(new Request(`http://localhost/api/group/${slug}`), {
        params: Promise.resolve({ slug }),
      });
      expect(res.status).toBe(404);
      expect(await res.json()).toEqual({ error: 'not_found', message: 'group not found' });
      expect(res.headers.get('cache-control')).toBeNull();
    });
  },
);

// Isolated summary-math fidelity tests (each test truncates and seeds its own
// scenario). They pin two areas the shared fixture leaves under-covered:
//  Compression latest-timestamp semantics:
//  - sub-second timestamps must not be truncated to whole seconds (regression
//    for the `to_char(... SS)` round-trip that exact-equality-rebound a coarser
//    text value and silently dropped the summary);
//  - decode is aggregated at the encode-derived timestamp, not decode's own;
//  - the timestamp falls back to the latest decode pair when no encode exists.
//  Query-summary penalty model:
//  - the 300_000 ns penalty floor is exercised (all runtimes below it) so a
//    missing query flips the ranking, pinning the floor + ratio formula.
describe.skipIf(!dockerAvailable())('summary math fidelity (testcontainers Postgres)', () => {
  let container: StartedPostgreSqlContainer;
  let measurementId = 0;

  function nextId(): number {
    measurementId += 1;
    return measurementId;
  }

  async function insertCommit(sha: string, ts: string): Promise<void> {
    await getPool().query(
      `INSERT INTO commits (commit_sha, timestamp, message, tree_sha, url)
         VALUES ($1, $2::timestamptz, $3, $4, $5)`,
      [sha, ts, 'fidelity fixture', TREE_SHA, commitUrl(sha)],
    );
  }

  // One complete vortex/parquet compression-time pair for a single op.
  async function insertCompTimePair(
    sha: string,
    op: string,
    vortexNs: number,
    parquetNs: number,
  ): Promise<void> {
    await getPool().query(
      `INSERT INTO compression_times
           (measurement_id, commit_sha, dataset, dataset_variant, format, op,
            value_ns, all_runtimes_ns)
         VALUES ($1, $2, 'tpch-lineitem', NULL, 'vortex-file-compressed', $4, $5, '{1}'::bigint[]),
                ($3, $2, 'tpch-lineitem', NULL, 'parquet',                $4, $6, '{1}'::bigint[])`,
      [nextId(), sha, nextId(), op, vortexNs, parquetNs],
    );
  }

  // One complete vortex/parquet compression-size pair.
  async function insertCompSizePair(
    sha: string,
    vortexBytes: number,
    parquetBytes: number,
  ): Promise<void> {
    await getPool().query(
      `INSERT INTO compression_sizes
           (measurement_id, commit_sha, dataset, dataset_variant, format, value_bytes)
         VALUES ($1, $2, 'tpch-lineitem', NULL, 'vortex-file-compressed', $4),
                ($3, $2, 'tpch-lineitem', NULL, 'parquet',                $5)`,
      [nextId(), sha, nextId(), vortexBytes, parquetBytes],
    );
  }

  beforeAll(async () => {
    container = await startBenchContainer();
  });

  afterAll(async () => {
    await resetPool();
    await container.stop();
  });

  beforeEach(async () => {
    await getPool().query(
      'TRUNCATE compression_times, compression_sizes, query_measurements, commits',
    );
  });

  // One latest-value query_measurements row for the shared tpch query group.
  async function insertQuery(
    sha: string,
    queryIdx: number,
    engine: string,
    format: string,
    valueNs: number,
  ): Promise<void> {
    await getPool().query(
      `INSERT INTO query_measurements
           (measurement_id, commit_sha, dataset, dataset_variant, scale_factor,
            query_idx, storage, engine, format, value_ns, all_runtimes_ns)
         VALUES ($1, $2, 'tpch', NULL, '1', $3, 'nvme', $4, $5, $6, '{1}'::bigint[])`,
      [nextId(), sha, queryIdx, engine, format, valueNs],
    );
  }

  it('keeps a sub-second latest commit timestamp (no whole-second truncation)', async () => {
    // A single commit whose timestamp carries microseconds. The pre-fix code
    // rendered MAX(ts) to whole-second text and rebound it with exact
    // equality, so this pair no longer matched and the summary dropped.
    const sha = 'a'.repeat(40);
    await insertCommit(sha, '2026-04-23T12:00:00.123456Z');
    await insertCompTimePair(sha, 'encode', 1_000, 3_000);
    await insertCompTimePair(sha, 'decode', 1_000, 2_000);
    await insertCompSizePair(sha, 1_000, 4_000);

    const time = expectDefined(
      await collectGroupSummary({ k: 'CompressionTimeGroup' }, []),
      'compression-time summary',
    );
    if (time.type !== 'compression') {
      throw new Error(`expected compression summary, got ${time.type}`);
    }
    expect(time.compressRatio).toBeCloseTo(3.0, 6);
    expect(time.decompressRatio).toBeCloseTo(2.0, 6);
    expect(time.datasetCount).toBe(1);

    const size = expectDefined(
      await collectGroupSummary({ k: 'CompressionSizeGroup' }, []),
      'compression-size summary',
    );
    if (size.type !== 'compressionSize') {
      throw new Error(`expected compressionSize summary, got ${size.type}`);
    }
    expect(size.minRatio).toBeCloseTo(0.25, 6);
    expect(size.meanRatio).toBeCloseTo(0.25, 6);
    expect(size.maxRatio).toBeCloseTo(0.25, 6);
    expect(size.datasetCount).toBe(1);
  });

  it('aggregates decode at the encode-derived timestamp, not decode’s own latest', async () => {
    // Encode pair only at the OLDER commit; decode pair only at the NEWER
    // commit. The summary timestamp is encode-derived, so the decode geomean is
    // taken at the encode commit (no decode pair there) and is therefore
    // omitted, matching the Rust single-timestamp aggregation.
    const older = 'b'.repeat(40);
    const newer = 'c'.repeat(40);
    await insertCommit(older, '2026-04-23T12:00:00Z');
    await insertCommit(newer, '2026-04-24T12:00:00Z');
    await insertCompTimePair(older, 'encode', 1_000, 3_000);
    await insertCompTimePair(newer, 'decode', 1_000, 2_000);

    const summary = expectDefined(
      await collectGroupSummary({ k: 'CompressionTimeGroup' }, []),
      'compression-time summary',
    );
    if (summary.type !== 'compression') {
      throw new Error(`expected compression summary, got ${summary.type}`);
    }
    expect(summary.compressRatio).toBeCloseTo(3.0, 6);
    expect(summary.decompressRatio).toBeUndefined();
    expect(summary.datasetCount).toBe(1);
  });

  it('falls back to the latest decode timestamp when there is no encode pair', async () => {
    const sha = 'd'.repeat(40);
    await insertCommit(sha, '2026-04-23T12:00:00Z');
    await insertCompTimePair(sha, 'decode', 1_000, 2_000);

    const summary = expectDefined(
      await collectGroupSummary({ k: 'CompressionTimeGroup' }, []),
      'compression-time summary',
    );
    if (summary.type !== 'compression') {
      throw new Error(`expected compression summary, got ${summary.type}`);
    }
    expect(summary.compressRatio).toBeUndefined();
    expect(summary.decompressRatio).toBeCloseTo(2.0, 6);
    expect(summary.datasetCount).toBe(0);
  });

  it('applies the 300_000 penalty floor so a missing query flips the ranking', async () => {
    // All present runtimes are below the 300_000 ns floor, so the missing-query
    // penalty is max(maxRuntime, 300_000) * 2 = 600_000 for both series (the
    // floor, not maxRuntime, sets it). duckdb has the fastest Q1 (50_000) but is
    // missing Q2, so the floor penalty pushes it BELOW datafusion. This pins both
    // the penalty floor constant and the (10+v)/(10+best) ratio formula.
    const sha = 'e'.repeat(40);
    await insertCommit(sha, '2026-04-23T12:00:00Z');
    await insertQuery(sha, 1, 'datafusion', 'vortex-file-compressed', 100_000);
    await insertQuery(sha, 2, 'datafusion', 'vortex-file-compressed', 200_000);
    await insertQuery(sha, 1, 'duckdb', 'parquet', 50_000);

    const summary = expectDefined(
      await collectGroupSummary(
        {
          k: 'QueryGroup',
          dataset: 'tpch',
          dataset_variant: null,
          scale_factor: '1',
          storage: 'nvme',
        },
        [],
      ),
      'query summary',
    );
    if (summary.type !== 'queryBenchmark') {
      throw new Error(`expected queryBenchmark summary, got ${summary.type}`);
    }
    // bestByQuery = {Q1: 50_000, Q2: 200_000}.
    // datafusion: sqrt((10+100_000)/(10+50_000) * 1) = sqrt(100010/50010).
    // duckdb: missing Q2 -> floor penalty 600_000;
    //   sqrt(1 * (10+600_000)/(10+200_000)) = sqrt(600010/200010).
    expect(summary.rankings.map((r) => r.name)).toEqual([
      'datafusion:vortex-file-compressed',
      'duckdb:parquet',
    ]);
    expect(summary.rankings[0].score).toBeCloseTo(1.414143, 5);
    expect(summary.rankings[1].score).toBeCloseTo(1.732022, 5);
  });
});

// Slug-decode rejection short-circuits to a 400 before any DB call, so this
// runs without Docker, matching the chart route's input-validation contract.
describe('GET /api/group/[slug] input validation (no DB)', () => {
  it('returns 400 for a structurally malformed slug', async () => {
    const slug = 'not-a-slug';
    const res = await getGroup(new Request(`http://localhost/api/group/${slug}`), {
      params: Promise.resolve({ slug }),
    });
    expect(res.status).toBe(400);
    expect(await res.json()).toEqual({ error: 'bad_request', message: 'invalid group slug' });
    expect(res.headers.get('cache-control')).toBeNull();
  });
});
