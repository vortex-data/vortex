// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB connection management plus the deterministic `measurement_id` hash.
//!
//! The server holds an [`r2d2::Pool`] of [`duckdb::Connection`]s. All DB
//! work runs inside `spawn_blocking`, where a connection is checked out
//! from the pool for the duration of the call. This lets Axum's concurrent
//! requests run on independent DuckDB connections instead of serializing
//! through a single mutex.
//!
//! `measurement_id` is a server-internal xxhash64 over `commit_sha` plus
//! each table's dimensional tuple. Including `commit_sha` makes every
//! (commit, dim) pair a distinct row, which is what the chart pages render
//! as a time series; re-ingest of the same pair is the upsert case. The
//! hash never crosses a process boundary, so the exact byte layout below
//! is private to this server.

use std::hash::Hasher as _;
use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Connection;
use duckdb::DuckdbConnectionManager;
use r2d2::Pool;
use twox_hash::XxHash64;

use crate::records::CompressionSize;
use crate::records::CompressionTime;
use crate::records::QueryMeasurement;
use crate::records::RandomAccessTime;
use crate::records::VectorSearchRun;
use crate::schema::SCHEMA_DDL;

/// A pool of DuckDB connections shared across all handlers. Cheap to clone
/// (internally an `Arc`).
pub type DbHandle = Pool<DuckdbConnectionManager>;

/// Open the DuckDB file at `path` (creating it if absent), apply the schema
/// DDL once, and return a connection pool ready to be cloned into the Axum
/// state.
pub fn open<P: AsRef<Path>>(path: P) -> Result<DbHandle> {
    let manager = DuckdbConnectionManager::file(path.as_ref())
        .with_context(|| format!("opening DuckDB at {}", path.as_ref().display()))?;
    let pool = Pool::builder()
        .build(manager)
        .context("building DuckDB connection pool")?;
    let conn = pool
        .get()
        .context("acquiring DuckDB connection from pool")?;
    conn.execute_batch(SCHEMA_DDL)
        .context("applying schema DDL")?;
    Ok(pool)
}

/// Run a synchronous DB operation on the blocking pool. A connection is
/// checked out from the r2d2 pool inside `spawn_blocking` (which may block
/// if the pool is saturated) and released when the closure returns.
pub async fn run_blocking<F, T>(handle: &DbHandle, f: F) -> Result<T>
where
    F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    let handle = handle.clone();
    tokio::task::spawn_blocking(move || {
        let mut conn = handle
            .get()
            .context("acquiring DuckDB connection from pool")?;
        f(&mut conn)
    })
    .await
    .context("DB task panicked")?
}

/// Finalize the hash and bit-cast to `i64` because DuckDB's `BIGINT` is
/// signed.
fn finish(hasher: XxHash64) -> i64 {
    hasher.finish() as i64
}

fn write_str(hasher: &mut XxHash64, s: &str) {
    hasher.write_u64(s.len() as u64);
    hasher.write(s.as_bytes());
}

fn write_opt_str(hasher: &mut XxHash64, s: Option<&str>) {
    match s {
        Some(s) => {
            hasher.write_u8(1);
            write_str(hasher, s);
        }
        None => hasher.write_u8(0),
    }
}

fn write_i32(hasher: &mut XxHash64, v: i32) {
    hasher.write_i32(v);
}

fn write_f64(hasher: &mut XxHash64, v: f64) {
    hasher.write_u64(v.to_bits());
}

/// Initialize a hasher seeded with a per-table tag so two fact tables that
/// happen to share the same dim values still produce distinct
/// `measurement_id`s.
fn hasher_for(tag: &'static str) -> XxHash64 {
    let mut h = XxHash64::with_seed(0);
    h.write(tag.as_bytes());
    h.write_u8(0);
    h
}

/// Hash for `query_measurements` rows. Includes `commit_sha` so each
/// (commit, dim tuple) pair gets a distinct row; re-emission of the same
/// pair is the upsert case.
pub fn measurement_id_query(r: &QueryMeasurement) -> i64 {
    let mut h = hasher_for("query_measurements");
    write_str(&mut h, &r.commit_sha);
    write_str(&mut h, &r.dataset);
    write_opt_str(&mut h, r.dataset_variant.as_deref());
    write_opt_str(&mut h, r.scale_factor.as_deref());
    write_i32(&mut h, r.query_idx);
    write_str(&mut h, &r.storage);
    write_str(&mut h, &r.engine);
    write_str(&mut h, &r.format);
    finish(h)
}

/// Hash for `compression_times` rows.
pub fn measurement_id_compression_time(r: &CompressionTime) -> i64 {
    let mut h = hasher_for("compression_times");
    write_str(&mut h, &r.commit_sha);
    write_str(&mut h, &r.dataset);
    write_opt_str(&mut h, r.dataset_variant.as_deref());
    write_str(&mut h, &r.format);
    write_str(&mut h, &r.op);
    finish(h)
}

/// Hash for `compression_sizes` rows.
pub fn measurement_id_compression_size(r: &CompressionSize) -> i64 {
    let mut h = hasher_for("compression_sizes");
    write_str(&mut h, &r.commit_sha);
    write_str(&mut h, &r.dataset);
    write_opt_str(&mut h, r.dataset_variant.as_deref());
    write_str(&mut h, &r.format);
    finish(h)
}

/// Hash for `random_access_times` rows.
pub fn measurement_id_random_access(r: &RandomAccessTime) -> i64 {
    let mut h = hasher_for("random_access_times");
    write_str(&mut h, &r.commit_sha);
    write_str(&mut h, &r.dataset);
    write_str(&mut h, &r.format);
    finish(h)
}

/// Hash for `vector_search_runs` rows. `iterations` is intentionally not
/// part of the dim tuple — it is a side count, not a dimension.
pub fn measurement_id_vector_search(r: &VectorSearchRun) -> i64 {
    let mut h = hasher_for("vector_search_runs");
    write_str(&mut h, &r.commit_sha);
    write_str(&mut h, &r.dataset);
    write_str(&mut h, &r.layout);
    write_str(&mut h, &r.flavor);
    write_f64(&mut h, r.threshold);
    finish(h)
}
