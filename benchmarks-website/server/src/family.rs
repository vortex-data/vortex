// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-fact-table registry.
//!
//! Each of the five fact tables (`query_measurements`, `compression_times`,
//! `compression_sizes`, `random_access_times`, `vector_search_runs`) is a
//! [`Family`]. Every Family declaration ties together the table name, the
//! chart and group slug prefixes, the `measurement_id_*` hash function, the
//! `apply_record` branch, the read-API chart and group collectors, and the
//! row-count query. Adding a sixth fact table is one new const [`Family`]
//! plus one entry in [`FAMILIES`]; the compiler enforces every required
//! hook is populated.
//!
//! See [`crate::schema`] for the DuckDB schema this registry is paired
//! with, [`crate::records::Record`] for the wire shape each Family
//! deserializes, and [`crate::slug`] for the slug variants each Family
//! claims (via [`Family::chart_slug_prefix`] / [`Family::group_slug_prefix`]).
//!
//! # The "spine" contract
//!
//! Every call site that varies per-fact-table dispatches through this
//! registry instead of hand-listing the families:
//!
//! - `crate::db::open` and `crate::ingest::apply_record` iterate
//!   [`FAMILIES`] for DDL apply and per-record dispatch.
//! - `crate::api::charts::chart_payload` and
//!   `crate::api::groups::collect_groups` dispatch through
//!   [`family_for_chart_key`] / per-family `collect_groups`.
//! - `crate::api::collect_health` iterates [`FAMILIES`] for `/health`'s
//!   `row_counts` map. The wire shape uses a `BTreeMap` so a new family
//!   appears in the response automatically; consumers index by table
//!   name (`row_counts["query_measurements"]`) just as before.
//! - `crate::slug::ChartKey::prefix` and `GroupKey::prefix` consult the
//!   registry rather than maintaining a parallel `PREFIX_*` const table.
//! - `crate::schema::TABLES` is derived from this registry at first use
//!   so the snapshot endpoint, the restore docs, and any future caller
//!   that needs the table-name set see the registry as the single source
//!   of truth.
//!
//! Per-family adapter functions in this file still trampoline into free
//! functions in [`mod@crate::api::charts`], [`mod@crate::api::groups`],
//! [`crate::db`], and [`crate::ingest`]. Inlining those bodies into the
//! adapters is mechanical and tracked as future cleanup; it does not
//! affect the spine contract above.

use anyhow::Result;
use duckdb::Connection;
use duckdb::Transaction;

use crate::api::ChartResponse;
use crate::api::CommitWindow;
use crate::api::Group;
use crate::api::charts;
use crate::api::groups;
use crate::db;
use crate::ingest;
use crate::ingest::RecordError;
use crate::records::Record;
use crate::slug::ChartKey;
use crate::slug::GroupKey;

/// One fact-table family.
///
/// Function-pointer fields (not closures) let the struct live in `const`
/// position so the registry is statically allocated. Adding a new family
/// means declaring one `const <NAME>_FAMILY: Family = Family { ... }` and
/// appending it to [`FAMILIES`].
pub struct Family {
    /// DuckDB table name. The `hasher_for(<table_name>)` seed in
    /// [`crate::db`] uses this string verbatim, so two families with
    /// otherwise-equal dim tuples still hash to distinct
    /// `measurement_id`s.
    pub table_name: &'static str,

    /// Slug prefix for individual charts in this family (e.g. `qm` for
    /// `query_measurements`). Paired with the JSON-encoded
    /// [`ChartKey`] variant - see [`crate::slug`].
    pub chart_slug_prefix: &'static str,

    /// Slug prefix for groups of this family's charts (e.g. `qmg`).
    pub group_slug_prefix: &'static str,

    /// `CREATE TABLE IF NOT EXISTS <table_name> (...)` for this family.
    /// [`crate::db::open`] runs the [`crate::schema::COMMITS_DDL`] dim
    /// table first, then iterates [`FAMILIES`] and applies this DDL for
    /// every family. Adding a sixth fact table means one new const DDL
    /// in [`crate::schema`] plus pointing this field at it; no new edit
    /// in `db::open()`.
    pub schema_ddl: &'static str,

    /// Compute the deterministic `measurement_id` for a record of this
    /// family. The caller is responsible for ensuring `record` is this
    /// family's variant (see [`family_for_record`]); calling with a
    /// foreign variant panics.
    pub measurement_id: fn(&Record) -> i64,

    /// Insert one record into this family's fact table inside an open
    /// transaction. Returns `true` if the row was an upsert (existing
    /// `measurement_id`) and `false` if it was a fresh insert. The
    /// caller is responsible for ensuring `record` is this family's
    /// variant; calling with a foreign variant panics.
    ///
    /// `pub(crate)` because [`RecordError`] is itself `pub(crate)` (it
    /// is intra-server error plumbing, not part of the public API).
    pub(crate) apply_record: fn(&Transaction<'_>, &Record) -> Result<bool, RecordError>,

    /// Collect this family's chart payload for the supplied [`ChartKey`].
    /// Returns `Ok(None)` if no rows exist for the key; the caller maps
    /// that to a 404. The caller is responsible for ensuring `key` is
    /// this family's variant (see [`family_for_chart_key`]); calling
    /// with a foreign variant panics.
    pub collect_chart_for_key:
        fn(&Connection, &ChartKey, &CommitWindow) -> Result<Option<ChartResponse>>,

    /// Discovery pass: enumerate every group of this family present in
    /// the DB. The returned `Vec` may be empty if no rows exist. The
    /// three singleton-group families (compression-times, compression-
    /// sizes, random-access) return a zero-or-one-element `Vec` here.
    pub collect_groups: fn(&Connection) -> Result<Vec<Group>>,

    /// Row count of this family's fact table; used by `/health` to
    /// surface per-table counts without spelling out the table names in
    /// the handler.
    pub row_count: fn(&Connection) -> Result<i64>,
}

/// All five fact-table families, in the order [`crate::schema::TABLES`]
/// applies their DDL. Adding a family appends one entry here.
pub const FAMILIES: &[&Family] = &[
    &QUERY_MEASUREMENTS,
    &COMPRESSION_TIMES,
    &COMPRESSION_SIZES,
    &RANDOM_ACCESS_TIMES,
    &VECTOR_SEARCH_RUNS,
];

/// Look up the family that owns `record`'s variant.
pub fn family_for_record(record: &Record) -> &'static Family {
    match record {
        Record::QueryMeasurement(_) => &QUERY_MEASUREMENTS,
        Record::CompressionTime(_) => &COMPRESSION_TIMES,
        Record::CompressionSize(_) => &COMPRESSION_SIZES,
        Record::RandomAccessTime(_) => &RANDOM_ACCESS_TIMES,
        Record::VectorSearchRun(_) => &VECTOR_SEARCH_RUNS,
    }
}

/// Look up the family that owns `key`'s variant.
pub fn family_for_chart_key(key: &ChartKey) -> &'static Family {
    match key {
        ChartKey::QueryMeasurement { .. } => &QUERY_MEASUREMENTS,
        ChartKey::CompressionTime { .. } => &COMPRESSION_TIMES,
        ChartKey::CompressionSize { .. } => &COMPRESSION_SIZES,
        ChartKey::RandomAccess { .. } => &RANDOM_ACCESS_TIMES,
        ChartKey::VectorSearch { .. } => &VECTOR_SEARCH_RUNS,
    }
}

/// Look up the family that owns `key`'s variant.
pub fn family_for_group_key(key: &GroupKey) -> &'static Family {
    match key {
        GroupKey::QueryGroup { .. } => &QUERY_MEASUREMENTS,
        GroupKey::CompressionTimeGroup => &COMPRESSION_TIMES,
        GroupKey::CompressionSizeGroup => &COMPRESSION_SIZES,
        GroupKey::RandomAccessGroup => &RANDOM_ACCESS_TIMES,
        GroupKey::VectorSearchGroup { .. } => &VECTOR_SEARCH_RUNS,
    }
}

// -----------------------------------------------------------------------
// Per-family declarations.
//
// Each Family's function-pointer fields point at small adapter functions
// that pattern-match the outer Record / ChartKey enums against the
// family's variant and delegate to the existing free functions in
// `crate::api::charts`, `crate::api::groups`, `crate::db`, and
// `crate::ingest`. Inlining those bodies into the adapters is mechanical
// future cleanup; the registry above is already the spine for the call
// sites that vary per-family.
// -----------------------------------------------------------------------

/// Family for `query_measurements`.
pub const QUERY_MEASUREMENTS: Family = Family {
    table_name: "query_measurements",
    chart_slug_prefix: "qm",
    group_slug_prefix: "qmg",
    schema_ddl: crate::schema::QUERY_MEASUREMENTS_DDL,
    measurement_id: query_measurement_id,
    apply_record: query_apply_record,
    collect_chart_for_key: query_collect_chart,
    collect_groups: groups::collect_query_groups,
    row_count: query_row_count,
};

/// Family for `compression_times`.
pub const COMPRESSION_TIMES: Family = Family {
    table_name: "compression_times",
    chart_slug_prefix: "ct",
    group_slug_prefix: "ctg",
    schema_ddl: crate::schema::COMPRESSION_TIMES_DDL,
    measurement_id: compression_time_measurement_id,
    apply_record: compression_time_apply_record,
    collect_chart_for_key: compression_time_collect_chart,
    collect_groups: compression_time_collect_groups,
    row_count: compression_time_row_count,
};

/// Family for `compression_sizes`.
pub const COMPRESSION_SIZES: Family = Family {
    table_name: "compression_sizes",
    chart_slug_prefix: "cs",
    group_slug_prefix: "csg",
    schema_ddl: crate::schema::COMPRESSION_SIZES_DDL,
    measurement_id: compression_size_measurement_id,
    apply_record: compression_size_apply_record,
    collect_chart_for_key: compression_size_collect_chart,
    collect_groups: compression_size_collect_groups,
    row_count: compression_size_row_count,
};

/// Family for `random_access_times`.
pub const RANDOM_ACCESS_TIMES: Family = Family {
    table_name: "random_access_times",
    chart_slug_prefix: "rat",
    group_slug_prefix: "rag",
    schema_ddl: crate::schema::RANDOM_ACCESS_TIMES_DDL,
    measurement_id: random_access_measurement_id,
    apply_record: random_access_apply_record,
    collect_chart_for_key: random_access_collect_chart,
    collect_groups: random_access_collect_groups,
    row_count: random_access_row_count,
};

/// Family for `vector_search_runs`.
pub const VECTOR_SEARCH_RUNS: Family = Family {
    table_name: "vector_search_runs",
    chart_slug_prefix: "vsr",
    group_slug_prefix: "vsg",
    schema_ddl: crate::schema::VECTOR_SEARCH_RUNS_DDL,
    measurement_id: vector_search_measurement_id,
    apply_record: vector_search_apply_record,
    collect_chart_for_key: vector_search_collect_chart,
    collect_groups: groups::collect_vector_search_groups,
    row_count: vector_search_row_count,
};

// -----------------------------------------------------------------------
// Per-family adapter functions. Each is a thin pattern-match-and-delegate
// wrapper over the existing free function with the matching name in
// `crate::api::charts`, `crate::api::groups`, `crate::db`, or
// `crate::ingest`.
// -----------------------------------------------------------------------

fn query_measurement_id(record: &Record) -> i64 {
    let Record::QueryMeasurement(r) = record else {
        panic!("query_measurement_id called with non-query record");
    };
    db::measurement_id_query(r)
}

fn query_apply_record(tx: &Transaction<'_>, record: &Record) -> Result<bool, RecordError> {
    let Record::QueryMeasurement(r) = record else {
        panic!("query_apply_record called with non-query record");
    };
    ingest::insert_query_measurement(tx, r)
}

fn query_collect_chart(
    conn: &Connection,
    key: &ChartKey,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let ChartKey::QueryMeasurement {
        dataset,
        dataset_variant,
        scale_factor,
        storage,
        query_idx,
    } = key
    else {
        panic!("query_collect_chart called with non-query key");
    };
    charts::collect_query_chart(
        conn,
        dataset,
        dataset_variant,
        scale_factor,
        storage,
        *query_idx,
        window,
    )
}

fn query_row_count(conn: &Connection) -> Result<i64> {
    count_rows(conn, "query_measurements")
}

fn compression_time_measurement_id(record: &Record) -> i64 {
    let Record::CompressionTime(r) = record else {
        panic!("compression_time_measurement_id called with non-compression-time record");
    };
    db::measurement_id_compression_time(r)
}

fn compression_time_apply_record(
    tx: &Transaction<'_>,
    record: &Record,
) -> Result<bool, RecordError> {
    let Record::CompressionTime(r) = record else {
        panic!("compression_time_apply_record called with non-compression-time record");
    };
    ingest::insert_compression_time(tx, r)
}

fn compression_time_collect_chart(
    conn: &Connection,
    key: &ChartKey,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let ChartKey::CompressionTime {
        dataset,
        dataset_variant,
    } = key
    else {
        panic!("compression_time_collect_chart called with non-compression-time key");
    };
    charts::collect_compression_time_chart(conn, dataset, dataset_variant, window)
}

fn compression_time_collect_groups(conn: &Connection) -> Result<Vec<Group>> {
    Ok(groups::collect_compression_time_group(conn)?
        .into_iter()
        .collect())
}

fn compression_time_row_count(conn: &Connection) -> Result<i64> {
    count_rows(conn, "compression_times")
}

fn compression_size_measurement_id(record: &Record) -> i64 {
    let Record::CompressionSize(r) = record else {
        panic!("compression_size_measurement_id called with non-compression-size record");
    };
    db::measurement_id_compression_size(r)
}

fn compression_size_apply_record(
    tx: &Transaction<'_>,
    record: &Record,
) -> Result<bool, RecordError> {
    let Record::CompressionSize(r) = record else {
        panic!("compression_size_apply_record called with non-compression-size record");
    };
    ingest::insert_compression_size(tx, r)
}

fn compression_size_collect_chart(
    conn: &Connection,
    key: &ChartKey,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let ChartKey::CompressionSize {
        dataset,
        dataset_variant,
    } = key
    else {
        panic!("compression_size_collect_chart called with non-compression-size key");
    };
    charts::collect_compression_size_chart(conn, dataset, dataset_variant, window)
}

fn compression_size_collect_groups(conn: &Connection) -> Result<Vec<Group>> {
    Ok(groups::collect_compression_size_group(conn)?
        .into_iter()
        .collect())
}

fn compression_size_row_count(conn: &Connection) -> Result<i64> {
    count_rows(conn, "compression_sizes")
}

fn random_access_measurement_id(record: &Record) -> i64 {
    let Record::RandomAccessTime(r) = record else {
        panic!("random_access_measurement_id called with non-random-access record");
    };
    db::measurement_id_random_access(r)
}

fn random_access_apply_record(tx: &Transaction<'_>, record: &Record) -> Result<bool, RecordError> {
    let Record::RandomAccessTime(r) = record else {
        panic!("random_access_apply_record called with non-random-access record");
    };
    ingest::insert_random_access(tx, r)
}

fn random_access_collect_chart(
    conn: &Connection,
    key: &ChartKey,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let ChartKey::RandomAccess { dataset } = key else {
        panic!("random_access_collect_chart called with non-random-access key");
    };
    charts::collect_random_access_chart(conn, dataset, window)
}

fn random_access_collect_groups(conn: &Connection) -> Result<Vec<Group>> {
    Ok(groups::collect_random_access_group(conn)?
        .into_iter()
        .collect())
}

fn random_access_row_count(conn: &Connection) -> Result<i64> {
    count_rows(conn, "random_access_times")
}

fn vector_search_measurement_id(record: &Record) -> i64 {
    let Record::VectorSearchRun(r) = record else {
        panic!("vector_search_measurement_id called with non-vector-search record");
    };
    db::measurement_id_vector_search(r)
}

fn vector_search_apply_record(tx: &Transaction<'_>, record: &Record) -> Result<bool, RecordError> {
    let Record::VectorSearchRun(r) = record else {
        panic!("vector_search_apply_record called with non-vector-search record");
    };
    ingest::insert_vector_search(tx, r)
}

fn vector_search_collect_chart(
    conn: &Connection,
    key: &ChartKey,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let ChartKey::VectorSearch {
        dataset,
        layout,
        threshold,
    } = key
    else {
        panic!("vector_search_collect_chart called with non-vector-search key");
    };
    charts::collect_vector_search_chart(conn, dataset, layout, *threshold, window)
}

fn vector_search_row_count(conn: &Connection) -> Result<i64> {
    count_rows(conn, "vector_search_runs")
}

/// `SELECT COUNT(*) FROM <table>`. The table name comes from a const in a
/// closed enum of literals above, never user input.
fn count_rows(conn: &Connection, table: &'static str) -> Result<i64> {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    let n: i64 = conn.query_row(&sql, [], |row| row.get(0))?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `schema::TABLES` is derived from `FAMILIES` (see [`crate::schema`]),
    /// so the order MUST be `commits` followed by every family in
    /// declaration order. This test pins the derivation rule explicitly
    /// in case someone replaces `TABLES` with a hand-written list again.
    #[test]
    fn schema_tables_derived_from_families() {
        let mut expected: Vec<&'static str> = Vec::with_capacity(1 + FAMILIES.len());
        expected.push("commits");
        expected.extend(FAMILIES.iter().map(|f| f.table_name));
        let actual: Vec<&'static str> = crate::schema::TABLES.iter().copied().collect();
        assert_eq!(
            actual, expected,
            "schema::TABLES must be derived from FAMILIES (commits + each family's table_name)"
        );
    }

    /// Slug prefixes are how the client and migrate path distinguish
    /// families. They must be distinct.
    #[test]
    fn slug_prefixes_are_distinct() {
        let chart_prefixes: Vec<&'static str> =
            FAMILIES.iter().map(|f| f.chart_slug_prefix).collect();
        let group_prefixes: Vec<&'static str> =
            FAMILIES.iter().map(|f| f.group_slug_prefix).collect();
        let mut sorted = chart_prefixes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            chart_prefixes.len(),
            "chart prefixes distinct"
        );
        let mut sorted = group_prefixes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            group_prefixes.len(),
            "group prefixes distinct"
        );
    }

    /// Per-fact-table `measurement_id` hashes carry the family's
    /// `table_name` as a per-domain seed (see `crate::db::hasher_for`),
    /// so two families sharing the same `commit_sha + dataset + format`
    /// dim tuple still produce distinct row keys. This regression test
    /// builds a [`crate::records::CompressionTime`] and a
    /// [`crate::records::CompressionSize`] over the same overlapping
    /// dim subset and asserts the hashes are non-zero and distinct.
    #[test]
    fn measurement_id_per_family_seed_keeps_overlapping_dims_distinct() {
        use crate::records::CompressionSize;
        use crate::records::CompressionTime;

        let ct = Record::CompressionTime(CompressionTime {
            commit_sha: "0000000000000000000000000000000000000000".into(),
            dataset: "shared".into(),
            dataset_variant: None,
            format: "vortex".into(),
            op: "encode".into(),
            value_ns: 0,
            all_runtimes_ns: vec![],
            env_triple: None,
        });
        let cs = Record::CompressionSize(CompressionSize {
            commit_sha: "0000000000000000000000000000000000000000".into(),
            dataset: "shared".into(),
            dataset_variant: None,
            format: "vortex".into(),
            value_bytes: 0,
        });
        let ct_id = (COMPRESSION_TIMES.measurement_id)(&ct);
        let cs_id = (COMPRESSION_SIZES.measurement_id)(&cs);
        assert_ne!(ct_id, 0, "compression_times hash must not be zero");
        assert_ne!(cs_id, 0, "compression_sizes hash must not be zero");
        assert_ne!(
            ct_id, cs_id,
            "per-family table_name seed must keep overlapping-dim hashes distinct \
             across families"
        );
    }
}
