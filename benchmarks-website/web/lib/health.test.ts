// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { StartedPostgreSqlContainer } from '@testcontainers/postgresql';
import { assembleHealth, buildRowCounts, collectHealth, type HealthResponse } from './health';
import { getPool, resetPool } from './db';
import { HEALTH_TABLES } from './families';
import { dockerAvailable, startBenchContainer } from './test-harness';

describe('buildRowCounts', () => {
  it('emits keys in HEALTH_TABLES (sorted BTreeMap) order', () => {
    // Insertion order here is deliberately scrambled to prove the output order
    // comes from HEALTH_TABLES, not from the input map.
    const counts = new Map<string, number>([
      ['vector_search_runs', 1],
      ['commits', 2],
      ['compression_times', 3],
      ['compression_sizes', 4],
      ['query_measurements', 5],
      ['random_access_times', 6],
    ]);
    expect(Object.keys(buildRowCounts(counts))).toEqual([...HEALTH_TABLES]);
  });

  it('throws loud when a table count is missing', () => {
    // Only `commits` is supplied; iteration is in sorted HEALTH_TABLES order, so
    // the first missing table reported is `compression_sizes`.
    expect(() => buildRowCounts(new Map([['commits', 1]]))).toThrow(/compression_sizes/);
  });
});

describe('assembleHealth', () => {
  it('builds the snake_case HealthResponse with status ok and schema_version 1', () => {
    const health: HealthResponse = assembleHealth({
      rowCounts: { commits: 3 },
      latestCommitTimestamp: '2024-01-15T10:30:45Z',
      dbPath: 'bench.example.rds.amazonaws.com',
      buildSha: 'abc123',
    });
    expect(health).toEqual({
      status: 'ok',
      db_path: 'bench.example.rds.amazonaws.com',
      schema_version: 1,
      build_sha: 'abc123',
      latest_commit_timestamp: '2024-01-15T10:30:45Z',
      row_counts: { commits: 3 },
    });
  });
});

describe.skipIf(!dockerAvailable())('collectHealth (testcontainers Postgres)', () => {
  let container: StartedPostgreSqlContainer;

  beforeAll(async () => {
    container = await startBenchContainer();
    delete process.env.VERCEL_GIT_COMMIT_SHA;
  });

  afterAll(async () => {
    await resetPool();
    await container.stop();
  });

  it('reports zero counts and a null timestamp against the empty schema', async () => {
    const health = await collectHealth();
    expect(health.status).toBe('ok');
    expect(health.schema_version).toBe(1);
    expect(health.db_path).toBe(container.getHost());
    expect(health.build_sha).toBe('unknown');
    expect(health.latest_commit_timestamp).toBeNull();
    expect(health.row_counts).toEqual({
      commits: 0,
      compression_sizes: 0,
      compression_times: 0,
      query_measurements: 0,
      random_access_times: 0,
      vector_search_runs: 0,
    });
  });

  it('reflects real row counts and the latest commit timestamp', async () => {
    await getPool().query(
      `INSERT INTO commits (commit_sha, timestamp, tree_sha, url)
       VALUES ('abc', '2024-01-15T10:30:45Z', 'tree', 'https://example/abc')`,
    );
    const health = await collectHealth();
    expect(health.row_counts.commits).toBe(1);
    expect(health.row_counts.query_measurements).toBe(0);
    expect(health.latest_commit_timestamp).toBe('2024-01-15T10:30:45Z');
  });
});
