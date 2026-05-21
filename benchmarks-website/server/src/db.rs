// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB connection management plus the deterministic `measurement_id` hash.
//!
//! The server keeps one root [`duckdb::Connection`] and clones a fresh
//! connection from it for each blocking DB task. All DB work runs inside
//! `spawn_blocking` so the Tokio runtime is never blocked on synchronous
//! DuckDB calls.
//!
//! `measurement_id` is a server-internal xxhash64 over `commit_sha` plus
//! each table's dimensional tuple. Including `commit_sha` makes every
//! (commit, dim) pair a distinct row, which is what the chart pages render
//! as a time series; re-ingest of the same pair is the upsert case. The
//! hash never crosses a process boundary, so the exact byte layout below
//! is private to this server.

use std::hash::Hasher as _;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Connection;
use parking_lot::Mutex;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;
use twox_hash::XxHash64;

use crate::records::CompressionSize;
use crate::records::CompressionTime;
use crate::records::QueryMeasurement;
use crate::records::RandomAccessTime;
use crate::records::VectorSearchRun;
use crate::schema::SCHEMA_DDL;

const READ_CONCURRENCY_LIMIT: usize = 4;

/// Shared DuckDB handle. Cloning the handle is cheap; each DB task clones a
/// task-local [`Connection`] before doing work.
#[derive(Clone)]
pub struct DbHandle {
    root: Arc<Mutex<Connection>>,
    read_permits: Arc<Semaphore>,
}

impl DbHandle {
    fn new(root: Connection) -> Self {
        Self {
            root: Arc::new(Mutex::new(root)),
            read_permits: Arc::new(Semaphore::new(READ_CONCURRENCY_LIMIT)),
        }
    }

    pub(crate) fn connection(&self) -> Result<Connection> {
        let root = self.root.lock();
        root.try_clone().context("cloning DuckDB connection")
    }
}

/// Open the DuckDB file at `path` (creating it if absent) and apply the
/// schema DDL. Returns a handle ready to be cloned into the Axum state.
pub fn open<P: AsRef<Path>>(path: P) -> Result<DbHandle> {
    let conn = Connection::open(path.as_ref())
        .with_context(|| format!("opening DuckDB at {}", path.as_ref().display()))?;
    conn.execute_batch(SCHEMA_DDL)
        .context("applying schema DDL")?;
    Ok(DbHandle::new(conn))
}

/// Run a synchronous DB operation on the blocking pool using a task-local
/// DuckDB connection cloned from the shared database handle.
pub async fn run_blocking<F, T>(handle: &DbHandle, f: F) -> Result<T>
where
    F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    run_blocking_inner(handle, None, f).await
}

/// Run a read-side DB operation on the blocking pool, capped by the read
/// concurrency semaphore so a hydration burst cannot flood DuckDB with
/// unbounded concurrent scans.
pub async fn run_read_blocking<F, T>(handle: &DbHandle, f: F) -> Result<T>
where
    F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    let permit = handle
        .read_permits
        .clone()
        .acquire_owned()
        .await
        .context("read semaphore closed")?;
    run_blocking_inner(handle, Some(permit), f).await
}

async fn run_blocking_inner<F, T>(
    handle: &DbHandle,
    permit: Option<OwnedSemaphorePermit>,
    f: F,
) -> Result<T>
where
    F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    let handle = handle.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut conn = handle.connection()?;
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use anyhow::Result;
    use tempfile::TempDir;

    use super::*;

    fn record_max(max_active: &AtomicUsize, value: usize) {
        let mut current = max_active.load(Ordering::SeqCst);
        while value > current {
            match max_active.compare_exchange(current, value, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn run_read_blocking_limits_concurrent_db_tasks() -> Result<()> {
        let tmp = TempDir::new()?;
        let handle = open(tmp.path().join("bench.duckdb"))?;
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();

        for _ in 0..12 {
            let active = Arc::clone(&active);
            let max_active = Arc::clone(&max_active);
            let handle = handle.clone();
            tasks.push(tokio::spawn(async move {
                run_read_blocking(&handle, move |_conn| {
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    record_max(&max_active, now);
                    std::thread::sleep(Duration::from_millis(25));
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(())
                })
                .await
            }));
        }

        for task in tasks {
            task.await??;
        }

        assert!(
            max_active.load(Ordering::SeqCst) <= 4,
            "read DB tasks should be capped at four concurrent workers"
        );
        Ok(())
    }
}
