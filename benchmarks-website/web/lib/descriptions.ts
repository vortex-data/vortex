// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * Editorial group descriptions, the TypeScript port of
 * `server/src/api/descriptions.rs` (originally ported from v2's
 * `src/config.js` `BENCHMARK_DESCRIPTIONS` + `src/utils.js`
 * `getBenchmarkDescription`).
 *
 * These strings are the source of truth for the hover tooltip rendered on each
 * group title. They are deliberately editorial and hand-maintained, derived
 * from the group *name* rather than from the database, so adding a new group's
 * blurb is a one-line edit here rather than a schema or ingest change.
 *
 * `TPC-H` / `TPC-DS` group names fan out by storage and scale factor, so their
 * descriptions are synthesised from the parsed name rather than hand-listed per
 * `(storage, sf)` pair.
 */

/**
 * Look up a short editorial description for a group display name. Returns
 * `null` when the group has no canonical description (e.g. vector-search
 * groups); callers render the title without a tooltip in that case.
 *
 * Mirrors the Rust `group_description`: the synthesised `TPC-*` blurb takes
 * precedence, then the hard-coded name-keyed table.
 */
export function groupDescription(name: string): string | null {
  const tpc = tpcDescription(name);
  if (tpc !== null) {
    return tpc;
  }
  return staticDescription(name);
}

/**
 * Hard-coded, name-keyed descriptions for the non-fan-out groups. These match
 * v2 verbatim where v2 had a description.
 */
function staticDescription(name: string): string | null {
  switch (name) {
    case 'Random Access':
      return (
        'Point lookups — selecting specific rows by position from an NVMe file. What feature ' +
        'stores, vector retrieval, and per-record serving actually do.'
      );
    case 'Compression':
      return (
        'Encode and decode throughput (MB/s) for Vortex vs Parquet (zstd page compression). ' +
        'Encode gates ingestion; decode gates every scan after.'
      );
    case 'Compression Size':
      return (
        'Compressed file size per format across a fixed set of datasets. A faster format that ' +
        'bloats on disk just trades one bill for another.'
      );
    case 'Clickbench':
      return (
        "ClickHouse's 43-query analytical suite over real web-analytics data — the field's " +
        'standard test for single-table scans, filters, and aggregations.'
      );
    case 'Statistical and Population Genetics':
      return (
        'Population-genetics queries over the gnomAD dataset — DuckDB-only, exercising the ' +
        'deeply-nested array operations real genomics pipelines run on.'
      );
    case 'PolarSignals Profiling':
      return (
        'Scan-layer benchmark modeled on PolarSignals/Parca: projection and filter pushdown ' +
        'over deeply-nested profile schemas — the shape continuous-profiling backends actually read.'
      );
    default:
      return null;
  }
}

/**
 * Suite-level TPC blurb (no storage / scale-factor dimensions) for the clustered
 * Historic-page section, where those become toggle buttons rather than part of
 * the heading. Returns `null` for non-TPC names.
 */
export function tpcSuiteDescription(suite: string): string | null {
  switch (suite) {
    case 'TPC-H':
      return 'TPC-H — 22 analytical queries against the canonical OLAP star schema.';
    case 'TPC-DS':
      return (
        'TPC-DS — the broader 99-query analytical suite, with larger schemas and skewed ' +
        'distributions.'
      );
    default:
      return null;
  }
}

/**
 * Derive a description for `TPC-H (NVMe|S3) (SF=N)` and `TPC-DS (NVMe) (SF=N)`
 * group names. The shape is fixed because the query-group name builder emits
 * exactly this format for tpch/tpcds. Returns `null` for any name that does not
 * start with `TPC-H ` or `TPC-DS `.
 */
function tpcDescription(name: string): string | null {
  let suite: 'TPC-H' | 'TPC-DS';
  let rest: string;
  if (name.startsWith('TPC-H ')) {
    suite = 'TPC-H';
    rest = name.slice('TPC-H '.length);
  } else if (name.startsWith('TPC-DS ')) {
    suite = 'TPC-DS';
    rest = name.slice('TPC-DS '.length);
  } else {
    return null;
  }
  let storage: 'nvme' | 's3';
  if (rest.startsWith('(NVMe)')) {
    storage = 'nvme';
  } else if (rest.startsWith('(S3)')) {
    storage = 's3';
  } else {
    return null;
  }
  const sf = parseSf(rest);
  if (sf === null) {
    return null;
  }
  return formatTpc(suite, storage, sf);
}

/**
 * Pull `SF=N` (leading digits only) out of strings like `(NVMe) (SF=10)`.
 * Returns `null` if there is no `SF=` substring or the digits don't parse.
 */
function parseSf(s: string): string | null {
  const idx = s.indexOf('SF=');
  if (idx === -1) {
    return null;
  }
  const after = s.slice(idx + 'SF='.length);
  let digits = '';
  for (const ch of after) {
    if (ch >= '0' && ch <= '9') {
      digits += ch;
    } else {
      break;
    }
  }
  return digits.length === 0 ? null : digits;
}

/**
 * Render the v2-compatible TPC blurb. The storage phrase comes from the parsed
 * group name; the scale-bytes annotation only renders for TPC-H (v2's TPC-DS
 * descriptions did not annotate scale bytes).
 */
function formatTpc(suite: 'TPC-H' | 'TPC-DS', storage: 'nvme' | 's3', sf: string): string {
  const storagePhrase = storage === 's3' ? 'against S3' : 'on local NVMe';
  const bytes = scaleBytes(sf);
  if (suite === 'TPC-H') {
    const schema = 'TPC-H — 22 analytical queries against the canonical OLAP star schema —';
    if (bytes !== null) {
      return `${schema} at scale factor ${sf} (~${bytes}), ${storagePhrase}.`;
    }
    return `${schema} at scale factor ${sf}, ${storagePhrase}.`;
  }
  return (
    'TPC-DS — the broader 99-query analytical suite (larger schemas, skewed distributions) — ' +
    `at scale factor ${sf}, ${storagePhrase}.`
  );
}

/** The human-readable dataset size for the known TPC scale factors. */
function scaleBytes(sf: string): string | null {
  switch (sf) {
    case '1':
      return '1 GB';
    case '10':
      return '10 GB';
    case '100':
      return '100 GB';
    case '1000':
      return '1 TB';
    default:
      return null;
  }
}
