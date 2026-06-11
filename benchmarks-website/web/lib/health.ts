// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * `/health` liveness probe plus a per-table row-count rollup, the TypeScript
 * port of `server/src/api/mod.rs::collect_health`.
 *
 * The wire shape preserves the Rust `HealthResponse`: snake_case field names,
 * a `row_counts` object keyed by table name in sorted (`BTreeMap`) order. Two
 * fields are adapted for the stateless Vercel deployment: `db_path` reports the
 * Postgres host (there is no local DuckDB file), and `build_sha` reports the
 * Vercel deployment commit SHA rather than a compile-time `env!`.
 */

import { getPool, sql } from './db';
import { HEALTH_TABLES } from './families';
import { SCHEMA_VERSION } from './schema-version';

/** Body of `GET /health`. Field names match the Rust `HealthResponse` exactly. */
export interface HealthResponse {
  status: string;
  db_path: string;
  schema_version: number;
  build_sha: string;
  latest_commit_timestamp: string | null;
  row_counts: Record<string, number>;
}

/**
 * Project per-table counts into the `row_counts` object, emitting keys in
 * [`HEALTH_TABLES`] order (sorted, matching the Rust `BTreeMap`). Throws if a
 * table's count is missing so a query gap fails loud rather than dropping a key.
 */
export function buildRowCounts(counts: ReadonlyMap<string, number>): Record<string, number> {
  const rowCounts: Record<string, number> = {};
  for (const table of HEALTH_TABLES) {
    const n = counts.get(table);
    if (n === undefined) {
      throw new Error(`missing row count for table \`${table}\``);
    }
    rowCounts[table] = n;
  }
  return rowCounts;
}

/** Assemble the `HealthResponse` from its parts. Pure, for unit testing. */
export function assembleHealth(args: {
  rowCounts: Record<string, number>;
  latestCommitTimestamp: string | null;
  dbPath: string;
  buildSha: string;
}): HealthResponse {
  return {
    status: 'ok',
    db_path: args.dbPath,
    schema_version: SCHEMA_VERSION,
    build_sha: args.buildSha,
    latest_commit_timestamp: args.latestCommitTimestamp,
    row_counts: args.rowCounts,
  };
}

async function countTable(table: string): Promise<number> {
  // `table` comes from `HEALTH_TABLES` (the `commits` dim table plus the closed
  // `FAMILIES` set), a compile-time constant set, never user input, so it is
  // safe in the identifier position. Re-assert membership defensively before
  // interpolating, mirroring the Rust `count_rows` "closed enum of literals".
  if (!HEALTH_TABLES.includes(table)) {
    throw new Error(`refusing to count unknown table \`${table}\``);
  }
  const result = await getPool().query<{ n: number }>(`SELECT COUNT(*)::int AS n FROM ${table}`);
  return result.rows[0].n;
}

/** Run the health queries against the pool and assemble the response. */
export async function collectHealth(): Promise<HealthResponse> {
  const entries = await Promise.all(
    HEALTH_TABLES.map(async (table) => [table, await countTable(table)] as const),
  );
  // Render the latest commit timestamp as a UTC RFC-3339 string. This drops
  // sub-second precision, which is safe because git commit timestamps are
  // whole-second; it diverges from the Rust server's DuckDB `CAST(... AS VARCHAR)`
  // format, but `/health` is a smoke-test field, not a wire-compat contract.
  const latest = await sql<{ ts: string | null }>`
    SELECT to_char(timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS ts
    FROM commits
    ORDER BY timestamp DESC
    LIMIT 1
  `;
  return assembleHealth({
    rowCounts: buildRowCounts(new Map(entries)),
    latestCommitTimestamp: latest.length > 0 ? latest[0].ts : null,
    dbPath: process.env.BENCH_DB_HOST ?? 'unknown',
    buildSha: process.env.VERCEL_GIT_COMMIT_SHA ?? 'unknown',
  });
}
