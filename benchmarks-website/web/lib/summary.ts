// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * v2-compatible per-group summary rollups, the TypeScript port of
 * `server/src/api/summary.rs`.
 *
 * Each `collect*Summary` runs a small set of focused SQL queries over a single
 * fact table and returns one [`Summary`] variant. The query-group summary is
 * gated on a v2 dataset allowlist via [`queryGroupHasV2Summary`].
 *
 * Behaviour-preservation notes (substrate migration, DuckDB -> Postgres):
 *  - `IS NOT DISTINCT FROM` gives `NULL == NULL` equality on the nullable
 *    `dataset_variant` / `scale_factor` dims, exactly as the DuckDB query did.
 *  - value columns are read `::float8` so node-postgres returns a JS `number`
 *    matching the Rust `CAST(... AS DOUBLE)`, rather than the bigint-as-string
 *    default.
 *  - the "latest timestamp" two-step (find the newest commit with a complete
 *    vortex/parquet pair, then aggregate at that timestamp) is preserved, but
 *    resolved entirely inside SQL via a CTE so `MAX(timestamp)` never round-trips
 *    through text. The Rust source rendered it via `CAST(MAX(ts) AS VARCHAR)` /
 *    `CAST(? AS TIMESTAMPTZ)`; DuckDB's VARCHAR cast preserves microseconds, but
 *    a text round-trip is fragile (a second-granularity render silently drops
 *    any sub-second commit timestamp), so the port keeps the timestamp in SQL.
 */

import { getPool } from './db';
import { compareCodeUnits } from './families';
import type { GroupKey } from './slug';

/** One random-access summary row. */
export interface RandomAccessRanking {
  /** Series name, normally the physical format. */
  name: string;
  /** Latest measured time in nanoseconds. */
  time: number;
  /** Ratio to the fastest series in the same chart. */
  ratio: number;
}

/** One query-benchmark summary row. */
export interface QueryRanking {
  /** Series name, normally `engine:format`. */
  name: string;
  /** Geomean ratio to the fastest observed value per query. */
  score: number;
  /** Sum of latest runtimes for the queries this series has. */
  totalRuntime: number;
}

/**
 * Server-computed group summary, the camelCase-tagged-union wire shape of the
 * Rust `Summary` enum (`#[serde(tag = "type")]` with camelCase variant names
 * and per-field renames). Optional ratio fields are omitted from the wire when
 * absent, matching the Rust `skip_serializing_if = "Option::is_none"`.
 */
export type Summary =
  | {
      type: 'randomAccess';
      title: string;
      rankings: RandomAccessRanking[];
      explanation: string;
    }
  | {
      type: 'compression';
      title: string;
      compressRatio?: number;
      decompressRatio?: number;
      datasetCount: number;
      explanation: string;
    }
  | {
      type: 'compressionSize';
      title: string;
      minRatio: number;
      meanRatio: number;
      maxRatio: number;
      datasetCount: number;
      explanation: string;
    }
  | {
      type: 'queryBenchmark';
      title: string;
      rankings: QueryRanking[];
      explanation: string;
    };

/**
 * Compute the v2-compatible summary for one group, if its kind has one.
 * `charts` is only consulted for the random-access path (which scans its chart
 * links for the latest populated dataset); the other paths query their fact
 * table directly. The structural `{ name }[]` accepts a `ChartLink[]`.
 */
export function collectGroupSummary(
  key: GroupKey,
  charts: readonly { readonly name: string }[],
): Promise<Summary | null> {
  switch (key.k) {
    case 'QueryGroup':
      if (queryGroupHasV2Summary(key.dataset)) {
        return collectQuerySummary(key.dataset, key.dataset_variant, key.scale_factor, key.storage);
      }
      return Promise.resolve(null);
    case 'CompressionTimeGroup':
      return collectCompressionSummary();
    case 'CompressionSizeGroup':
      return collectCompressionSizeSummary();
    case 'RandomAccessGroup':
      return collectRandomAccessSummary(charts);
    case 'VectorSearchGroup':
      return Promise.resolve(null);
  }
}

/** The v2 dataset allowlist for which a query group carries a summary. */
function queryGroupHasV2Summary(dataset: string): boolean {
  switch (dataset) {
    case 'clickbench':
    case 'statpopgen':
    case 'polarsignals':
    case 'tpch':
    case 'tpcds':
      return true;
    default:
      return false;
  }
}

/**
 * Geometric mean over the positive, finite values, or `null` when none
 * qualify. Computed in log space (`exp(mean(ln(v)))`) exactly as the Rust
 * `geo_mean`.
 */
function geoMean(values: readonly number[]): number | null {
  let sumLn = 0;
  let n = 0;
  for (const value of values) {
    if (value > 0 && Number.isFinite(value)) {
      sumLn += Math.log(value);
      n += 1;
    }
  }
  return n > 0 ? Math.exp(sumLn / n) : null;
}

async function collectRandomAccessSummary(
  charts: readonly { readonly name: string }[],
): Promise<Summary | null> {
  // Scan the group's chart links in order; the first chart with valid rows at
  // its latest commit wins (matching the Rust early-return loop).
  const text = `
    SELECT r.format AS name, r.value_ns::float8 AS value
      FROM random_access_times r
      JOIN commits c USING (commit_sha)
     WHERE r.dataset = $1
       AND r.value_ns > 0
       AND c.timestamp = (
            SELECT MAX(c2.timestamp)
              FROM random_access_times r2
              JOIN commits c2 USING (commit_sha)
             WHERE r2.dataset = $2
               AND r2.value_ns > 0
       )
     ORDER BY r.value_ns, r.format
  `;
  for (const chart of charts) {
    const rows = (
      await getPool().query<{ name: string; value: number }>(text, [chart.name, chart.name])
    ).rows;
    const rankings: RandomAccessRanking[] = rows.map((row) => ({
      name: row.name,
      time: row.value,
      ratio: 0,
    }));
    if (rankings.length === 0) {
      continue;
    }
    // Streaming min (loop, not a `Math.min(...)` spread) for consistency with
    // `collectCompressionSizeSummary` and to avoid a large-array call-argument
    // cliff; the Rust source uses `reduce(f64::min)` here.
    let minTime = Infinity;
    for (const r of rankings) {
      minTime = Math.min(minTime, r.time);
    }
    if (minTime <= 0 || !Number.isFinite(minTime)) {
      continue;
    }
    for (const r of rankings) {
      r.ratio = r.time / minTime;
    }
    rankings.sort((a, b) =>
      a.time < b.time ? -1 : a.time > b.time ? 1 : compareCodeUnits(a.name, b.name),
    );
    return {
      type: 'randomAccess',
      title: 'Random Access Performance',
      rankings,
      explanation: 'Random access time | Ratio to fastest (lower is better)',
    };
  }
  return null;
}

async function collectCompressionSummary(): Promise<Summary | null> {
  // Both the encode (compress) and decode (decompress) geomeans are evaluated at
  // the encode-derived latest timestamp, falling back to the decode timestamp
  // when no encode pair exists (Rust order). The shared timestamp is resolved
  // inside SQL so it never round-trips through text and loses sub-second
  // precision.
  const { compress, decompress } = await compressionSpeedups();
  if (compress.length === 0 && decompress.length === 0) {
    return null;
  }
  const summary: Extract<Summary, { type: 'compression' }> = {
    type: 'compression',
    title: 'Compression Throughput vs Parquet',
    datasetCount: compress.length,
    explanation: 'Inverse geomean of Vortex/Parquet ratios (higher is better)',
  };
  const compressRatio = geoMean(compress);
  if (compressRatio !== null) {
    summary.compressRatio = compressRatio;
  }
  const decompressRatio = geoMean(decompress);
  if (decompressRatio !== null) {
    summary.decompressRatio = decompressRatio;
  }
  return summary;
}

/**
 * Parquet/Vortex encode (`compress`) and decode (`decompress`) throughput ratios
 * at the shared latest commit timestamp. The timestamp is the newest commit with
 * a complete encode vortex/parquet pair, falling back to the newest decode pair
 * when no encode pair exists, matching `collect_compression_summary`. It is
 * resolved inside the query via a CTE, so `MAX(timestamp)` never round-trips
 * through text (a second-granularity text render silently drops any sub-second
 * commit timestamp). Decode ratios are taken at the encode-derived timestamp,
 * preserving the Rust behaviour of aggregating both ops at one timestamp.
 */
async function compressionSpeedups(): Promise<{ compress: number[]; decompress: number[] }> {
  const text = `
    WITH pairs AS (
      SELECT v.op AS op,
             c.timestamp AS ts,
             p.value_ns::float8 / v.value_ns::float8 AS ratio,
             v.dataset AS dataset,
             v.dataset_variant AS dataset_variant
        FROM compression_times v
        JOIN compression_times p
          ON p.commit_sha = v.commit_sha
         AND p.dataset = v.dataset
         AND p.dataset_variant IS NOT DISTINCT FROM v.dataset_variant
         AND p.op = v.op
        JOIN commits c ON c.commit_sha = v.commit_sha
       WHERE v.op IN ('encode', 'decode')
         AND v.format = 'vortex-file-compressed'
         AND p.format = 'parquet'
         AND v.value_ns > 0
         AND p.value_ns > 0
         AND lower(v.dataset) NOT LIKE '%wide table%'
    )
    SELECT pairs.op AS op, pairs.ratio AS ratio
      FROM pairs
      JOIN (
        SELECT COALESCE(
          (SELECT MAX(ts) FROM pairs WHERE op = 'encode'),
          (SELECT MAX(ts) FROM pairs WHERE op = 'decode')
        ) AS ts
      ) shared ON pairs.ts = shared.ts
     ORDER BY pairs.dataset, pairs.dataset_variant NULLS FIRST
  `;
  const rows = (await getPool().query<{ op: string; ratio: number }>(text)).rows;
  const compress: number[] = [];
  const decompress: number[] = [];
  for (const row of rows) {
    if (row.op === 'encode') {
      compress.push(row.ratio);
    } else if (row.op === 'decode') {
      decompress.push(row.ratio);
    }
  }
  return { compress, decompress };
}

async function collectCompressionSizeSummary(): Promise<Summary | null> {
  const ratios = await compressionSizeRatios();
  const meanRatio = geoMean(ratios);
  if (meanRatio === null) {
    return null;
  }
  // Streaming min/max fold (loop, not a `Math.min(...ratios)` spread) so a very
  // large dataset count cannot overflow the call-argument limit, matching the
  // Rust `fold(INFINITY, f64::min)` / `fold(NEG_INFINITY, f64::max)`.
  let minRatio = Infinity;
  let maxRatio = -Infinity;
  for (const ratio of ratios) {
    minRatio = Math.min(minRatio, ratio);
    maxRatio = Math.max(maxRatio, ratio);
  }
  return {
    type: 'compressionSize',
    title: 'Compression Size Summary',
    minRatio,
    meanRatio,
    maxRatio,
    datasetCount: ratios.length,
    explanation: 'Geomean of Vortex/Parquet size ratios (lower is better)',
  };
}

/**
 * Vortex/Parquet size ratios at the latest commit with a complete vortex/parquet
 * size pair. `MAX(timestamp)` is resolved inside the query (CTE) so it never
 * round-trips through text and drops sub-second commit timestamps.
 */
async function compressionSizeRatios(): Promise<number[]> {
  const text = `
    WITH pairs AS (
      SELECT c.timestamp AS ts,
             v.value_bytes::float8 / p.value_bytes::float8 AS ratio,
             v.dataset AS dataset,
             v.dataset_variant AS dataset_variant
        FROM compression_sizes v
        JOIN compression_sizes p
          ON p.commit_sha = v.commit_sha
         AND p.dataset = v.dataset
         AND p.dataset_variant IS NOT DISTINCT FROM v.dataset_variant
        JOIN commits c ON c.commit_sha = v.commit_sha
       WHERE v.format = 'vortex-file-compressed'
         AND p.format = 'parquet'
         AND v.value_bytes > 0
         AND p.value_bytes > 0
         AND lower(v.dataset) NOT LIKE '%wide table%'
    )
    SELECT ratio
      FROM pairs
     WHERE ts = (SELECT MAX(ts) FROM pairs)
     ORDER BY dataset, dataset_variant NULLS FIRST
  `;
  const rows = (await getPool().query<{ ratio: number }>(text)).rows;
  return rows.map((row) => row.ratio);
}

async function collectQuerySummary(
  dataset: string,
  datasetVariant: string | null,
  scaleFactor: string | null,
  storage: string,
): Promise<Summary | null> {
  // Latest value per (query_idx, engine, format), then v2's missing-series
  // penalty model: each series scores the geomean of `(10 + value) / (10 +
  // best)` over every query, imputing a penalty where the series has no value.
  const text = `
    WITH latest AS (
      SELECT q.query_idx,
             q.engine || ':' || q.format AS series,
             q.value_ns::float8 AS value_ns,
             row_number() OVER (
               PARTITION BY q.query_idx, q.engine, q.format
               ORDER BY c.timestamp DESC
             ) AS rn
        FROM query_measurements q
        JOIN commits c USING (commit_sha)
       WHERE q.dataset = $1
         AND q.dataset_variant IS NOT DISTINCT FROM $2
         AND q.scale_factor    IS NOT DISTINCT FROM $3
         AND q.storage = $4
         AND q.value_ns > 0
    )
    SELECT query_idx, series, value_ns
      FROM latest
     WHERE rn = 1
     ORDER BY query_idx, series
  `;
  const rows = (
    await getPool().query<{ query_idx: number; series: string; value_ns: number }>(text, [
      dataset,
      datasetVariant,
      scaleFactor,
      storage,
    ])
  ).rows;

  const queries = new Set<number>();
  const valuesBySeries = new Map<string, Map<number, number>>();
  for (const row of rows) {
    queries.add(row.query_idx);
    let series = valuesBySeries.get(row.series);
    if (series === undefined) {
      series = new Map<number, number>();
      valuesBySeries.set(row.series, series);
    }
    series.set(row.query_idx, row.value_ns);
  }
  if (valuesBySeries.size === 0) {
    return null;
  }

  // Sorted query indices match the Rust `BTreeSet<i32>` iteration order.
  const sortedQueries = [...queries].sort((a, b) => a - b);
  const bestByQuery = new Map<number, number>();
  for (const queryIdx of sortedQueries) {
    let best = Infinity;
    for (const series of valuesBySeries.values()) {
      const value = series.get(queryIdx);
      if (value !== undefined && value < best) {
        best = value;
      }
    }
    if (Number.isFinite(best)) {
      bestByQuery.set(queryIdx, best);
    }
  }

  const rankings: QueryRanking[] = [];
  // Sorted series keys match the Rust `BTreeMap<String, _>` iteration order.
  for (const name of [...valuesBySeries.keys()].sort(compareCodeUnits)) {
    const queryValues = valuesBySeries.get(name);
    if (queryValues === undefined) {
      continue;
    }
    let totalRuntime = 0;
    for (const queryIdx of [...queryValues.keys()].sort((a, b) => a - b)) {
      totalRuntime += queryValues.get(queryIdx) ?? 0;
    }
    let maxRuntime = -Infinity;
    for (const value of queryValues.values()) {
      if (value > maxRuntime) {
        maxRuntime = value;
      }
    }
    if (!Number.isFinite(maxRuntime)) {
      continue;
    }
    const penalty = Math.max(maxRuntime, 300_000) * 2;
    const ratios: number[] = [];
    for (const queryIdx of sortedQueries) {
      const base = bestByQuery.get(queryIdx);
      if (base === undefined) {
        continue;
      }
      const value = queryValues.get(queryIdx) ?? penalty;
      ratios.push((10 + value) / (10 + base));
    }
    const score = geoMean(ratios);
    if (score === null) {
      continue;
    }
    rankings.push({ name, score, totalRuntime });
  }
  rankings.sort((a, b) =>
    a.score < b.score ? -1 : a.score > b.score ? 1 : compareCodeUnits(a.name, b.name),
  );

  if (rankings.length === 0) {
    return null;
  }
  return {
    type: 'queryBenchmark',
    title: 'Performance Summary',
    rankings,
    explanation: 'Geomean of query time ratio to fastest (lower is better)',
  };
}
