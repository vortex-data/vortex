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
      return 'Tests performance of selecting arbitrary row indices from a file on NVMe storage';
    case 'Compression':
      return (
        'Measures encoding and decoding throughput (MB/s) for Vortex files and Parquet ' +
        'files (with zstd page compression)'
      );
    case 'Compression Size':
      return (
        'Compares compressed file sizes and compression ratios across different encoding ' +
        'strategies'
      );
    case 'Clickbench':
      return (
        "ClickHouse's analytical benchmark suite testing real-world query patterns on web " +
        'analytics data'
      );
    case 'Statistical and Population Genetics':
      return 'A suite of Statistical and Population genetics queries using the gnomAD dataset';
    case 'PolarSignals Profiling':
      return (
        'Profiling data benchmark modeled on PolarSignals/Parca, exercising scan-layer ' +
        'performance with projection and filter pushdown on deeply nested schemas'
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
  const storagePhrase = storage === 's3' ? 'against S3 storage' : 'on local NVMe storage';
  const bytes = scaleBytes(sf);
  if (suite === 'TPC-H') {
    if (bytes !== null) {
      return `TPC-H benchmark queries ${storagePhrase} at SF=${sf} (~${bytes} of data)`;
    }
    return `TPC-H benchmark queries ${storagePhrase} at SF=${sf}`;
  }
  return `TPC-DS benchmark queries ${storagePhrase} at SF=${sf}`;
}

/** The human-readable dataset size for the known TPC scale factors. */
function scaleBytes(sf: string): string | null {
  switch (sf) {
    case '1':
      return '1GB';
    case '10':
      return '10GB';
    case '100':
      return '100GB';
    case '1000':
      return '1TB';
    default:
      return null;
  }
}
