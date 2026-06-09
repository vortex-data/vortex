// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * Shared testcontainers harness for the DB-backed vitest suites. This module
 * is imported only by `*.test.ts` files; it never reaches a production bundle.
 *
 * It centralizes the pieces the suites previously copy-pasted (and that had
 * already begun to drift in name): the Docker probe, the `migrations/001` DDL,
 * the container-boot + `BENCH_DB_*` env wiring, and the canonical three-commit
 * chart fixture mirroring `server/tests/common/mod.rs`.
 */

import { execSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { PostgreSqlContainer, type StartedPostgreSqlContainer } from '@testcontainers/postgresql';
import type { Pool } from 'pg';
import { getPool } from './db';

// Mirror the repo's Python `_docker_available()` precedent: the integration
// tests need a Docker daemon, so they are skipped (not failed) when one is
// absent.
export function dockerAvailable(): boolean {
  try {
    execSync('docker info', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/** The v4 schema DDL, read from the canonical migration the deploys apply. */
export const SCHEMA_DDL = readFileSync(
  fileURLToPath(new URL('../../../migrations/001_initial_schema.sql', import.meta.url)),
  'utf8',
);

/**
 * Start a `postgres:16-alpine` testcontainer, point the connection lib at it
 * via the `BENCH_DB_*` env vars (`BENCH_DB_PASSWORD` set means the IAM token
 * path is bypassed), and apply the `migrations/001` schema unless
 * `applySchema: false`. Callers own teardown: `await resetPool()` then
 * `await container.stop()` in `afterAll`.
 */
export async function startBenchContainer(
  options: { applySchema?: boolean } = {},
): Promise<StartedPostgreSqlContainer> {
  const container = await new PostgreSqlContainer('postgres:16-alpine').start();
  process.env.BENCH_DB_HOST = container.getHost();
  process.env.BENCH_DB_PORT = String(container.getPort());
  process.env.BENCH_DB_NAME = container.getDatabase();
  process.env.BENCH_DB_USER = container.getUsername();
  process.env.BENCH_DB_PASSWORD = container.getPassword();
  process.env.BENCH_DB_SSL = 'disable';
  if (options.applySchema !== false) {
    await getPool().query(SCHEMA_DDL);
  }
  return container;
}

// The canonical web-ui fixture, mirroring `server/tests/common/mod.rs`: three
// oldest-first commits, each carrying the same record set with a per-commit
// `bias` added so the series are non-flat. Fact rows are INSERTed directly
// rather than POSTed as ingest envelopes (the ingest path is still Rust-only),
// with synthetic `measurement_id`s since the read queries never read them.
export const COMMITS: ReadonlyArray<readonly [string, string, string]> = [
  ['1'.repeat(40), '2026-04-23T12:00:00Z', 'first commit'],
  ['2'.repeat(40), '2026-04-24T12:00:00Z', 'second commit'],
  ['3'.repeat(40), '2026-04-25T12:00:00Z', 'third commit'],
];

/** The fixture's shared (arbitrary) tree SHA. */
export const TREE_SHA = 'fedcba9876543210fedcba9876543210fedcba98';

/** Render the fixture commit URL for `sha`, matching the ingest writer. */
export function commitUrl(sha: string): string {
  return `https://github.com/vortex-data/vortex/commit/${sha}`;
}

/** Seed the canonical three-commit fixture (see [`COMMITS`]) into `pool`. */
export async function seedChartFixture(pool: Pool): Promise<void> {
  let id = 0;
  const mid = (): number => {
    id += 1;
    return id;
  };
  for (const [sha, ts, msg] of COMMITS) {
    await pool.query(
      `INSERT INTO commits (commit_sha, timestamp, message, tree_sha, url)
       VALUES ($1, $2::timestamptz, $3, $4, $5)`,
      [sha, ts, msg, TREE_SHA, commitUrl(sha)],
    );
  }
  for (let i = 0; i < COMMITS.length; i += 1) {
    const sha = COMMITS[i][0];
    const bias = i * 50_000;
    // query_measurements: Q1 has two engine/format series, Q2 has one.
    const qm: ReadonlyArray<readonly [number, string, string, number]> = [
      [1, 'datafusion', 'vortex-file-compressed', 1_000_000 + bias],
      [1, 'duckdb', 'parquet', 800_000 + bias],
      [2, 'datafusion', 'vortex-file-compressed', 600_000 + bias],
    ];
    for (const [queryIdx, engine, format, valueNs] of qm) {
      await pool.query(
        `INSERT INTO query_measurements
           (measurement_id, commit_sha, dataset, dataset_variant, scale_factor,
            query_idx, storage, engine, format, value_ns, all_runtimes_ns)
         VALUES ($1, $2, 'tpch', NULL, '1', $3, 'nvme', $4, $5, $6, '{1}'::bigint[])`,
        [mid(), sha, queryIdx, engine, format, valueNs],
      );
    }
    const compTimes: ReadonlyArray<readonly [string, string, number]> = [
      ['vortex-file-compressed', 'encode', 9_000 + bias],
      ['vortex-file-compressed', 'decode', 5_000 + bias],
      ['parquet', 'encode', 18_000 + 2 * bias],
      ['parquet', 'decode', 10_000 + 2 * bias],
    ];
    for (const [format, op, valueNs] of compTimes) {
      await pool.query(
        `INSERT INTO compression_times
           (measurement_id, commit_sha, dataset, dataset_variant, format, op,
            value_ns, all_runtimes_ns)
         VALUES ($1, $2, 'tpch-lineitem', NULL, $3, $4, $5, '{1}'::bigint[])`,
        [mid(), sha, format, op, valueNs],
      );
    }
    const compSizes: ReadonlyArray<readonly [string, number]> = [
      ['vortex-file-compressed', 4_000 + bias],
      ['parquet', 8_000 + 2 * bias],
    ];
    for (const [format, valueBytes] of compSizes) {
      await pool.query(
        `INSERT INTO compression_sizes
           (measurement_id, commit_sha, dataset, dataset_variant, format, value_bytes)
         VALUES ($1, $2, 'tpch-lineitem', NULL, $3, $4)`,
        [mid(), sha, format, valueBytes],
      );
    }
    const randomAccess: ReadonlyArray<readonly [string, number]> = [
      ['vortex-file-compressed', 500 + bias],
      ['parquet', 1_000 + 2 * bias],
    ];
    for (const [format, valueNs] of randomAccess) {
      await pool.query(
        `INSERT INTO random_access_times
           (measurement_id, commit_sha, dataset, format, value_ns, all_runtimes_ns)
         VALUES ($1, $2, 'taxi', $3, $4, '{1}'::bigint[])`,
        [mid(), sha, format, valueNs],
      );
    }
    await pool.query(
      `INSERT INTO vector_search_runs
         (measurement_id, commit_sha, dataset, layout, flavor, threshold, value_ns,
          all_runtimes_ns, matches, rows_scanned, bytes_scanned, iterations)
       VALUES ($1, $2, 'cohere-large-10m', 'partitioned', 'vortex-turboquant', 0.75, $3,
               '{1}'::bigint[], 42, 1000000, 5000000, 1)`,
      [mid(), sha, 7_000 + bias],
    );
  }
}
