// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * Per-fact-table registry, the TypeScript port of `server/src/family.rs`.
 *
 * Each of the five fact tables is a [`Family`] that ties together the Postgres
 * table name and the chart and group slug prefixes. The read endpoints
 * dispatch through this registry rather than hand-listing the families, so the
 * slug prefixes (consumed by [`./slug`]) and the table-name set (consumed by
 * `/health`'s row counts; the read queries name their tables in static SQL)
 * have a single source of truth, exactly as the Rust `family.rs` "spine" does.
 *
 * The order of [`FAMILIES`] mirrors `family.rs`'s `FAMILIES` constant and the
 * DDL apply order in `migrations/001_initial_schema.sql`.
 */

/** The five fact-table names, spelled exactly as in `migrations/001`. */
export type FamilyTable =
  | 'query_measurements'
  | 'compression_times'
  | 'compression_sizes'
  | 'random_access_times'
  | 'vector_search_runs';

/** Discriminant (`k`) of a [`./slug`].`ChartKey`; one per family. */
export type ChartKind =
  | 'QueryMeasurement'
  | 'CompressionTime'
  | 'CompressionSize'
  | 'RandomAccess'
  | 'VectorSearch';

/** Discriminant (`k`) of a [`./slug`].`GroupKey`; one per family. */
export type GroupKind =
  | 'QueryGroup'
  | 'CompressionTimeGroup'
  | 'CompressionSizeGroup'
  | 'RandomAccessGroup'
  | 'VectorSearchGroup';

/** One fact-table family: its table name, slug prefixes, and key discriminants. */
export interface Family {
  /** Postgres table name, matching `migrations/001_initial_schema.sql`. */
  tableName: FamilyTable;
  /** Slug prefix for individual charts in this family (e.g. `qm`). */
  chartSlugPrefix: string;
  /** Slug prefix for groups of this family's charts (e.g. `qmg`). */
  groupSlugPrefix: string;
  /** The `ChartKey` discriminant this family owns. */
  chartKind: ChartKind;
  /** The `GroupKey` discriminant this family owns. */
  groupKind: GroupKind;
}

/**
 * All five fact-table families, in the order `family.rs`'s `FAMILIES` declares
 * them (which is also the `migrations/001` DDL apply order).
 */
export const FAMILIES: readonly Family[] = [
  {
    tableName: 'query_measurements',
    chartSlugPrefix: 'qm',
    groupSlugPrefix: 'qmg',
    chartKind: 'QueryMeasurement',
    groupKind: 'QueryGroup',
  },
  {
    tableName: 'compression_times',
    chartSlugPrefix: 'ct',
    groupSlugPrefix: 'ctg',
    chartKind: 'CompressionTime',
    groupKind: 'CompressionTimeGroup',
  },
  {
    tableName: 'compression_sizes',
    chartSlugPrefix: 'cs',
    groupSlugPrefix: 'csg',
    chartKind: 'CompressionSize',
    groupKind: 'CompressionSizeGroup',
  },
  {
    tableName: 'random_access_times',
    chartSlugPrefix: 'rat',
    groupSlugPrefix: 'rag',
    chartKind: 'RandomAccess',
    groupKind: 'RandomAccessGroup',
  },
  {
    tableName: 'vector_search_runs',
    chartSlugPrefix: 'vsr',
    groupSlugPrefix: 'vsg',
    chartKind: 'VectorSearch',
    groupKind: 'VectorSearchGroup',
  },
];

/** Look up the family that owns a chart-key discriminant. */
export function familyForChartKind(kind: ChartKind): Family {
  const family = FAMILIES.find((f) => f.chartKind === kind);
  if (family === undefined) {
    throw new Error(`no family for chart kind \`${kind}\``);
  }
  return family;
}

/** Look up the family that owns a group-key discriminant. */
export function familyForGroupKind(kind: GroupKind): Family {
  const family = FAMILIES.find((f) => f.groupKind === kind);
  if (family === undefined) {
    throw new Error(`no family for group kind \`${kind}\``);
  }
  return family;
}

/**
 * The Postgres tables surfaced by `/health`, in `BTreeMap` (sorted) order to
 * match the Rust `HealthResponse.row_counts` wire shape: the `commits` dim
 * table plus every [`FAMILIES`] table name, sorted lexicographically.
 */
export const HEALTH_TABLES: readonly string[] = [
  'commits',
  ...FAMILIES.map((f) => f.tableName),
].sort();

/**
 * Byte-order string comparison matching Rust `String::cmp` (and so `BTreeMap`
 * key order); the ASCII series, format, and ranking names compared by the read
 * port order identically under JS code-unit comparison. Shared by the query
 * and summary modules so the two ports cannot drift apart.
 */
export function compareCodeUnits(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
