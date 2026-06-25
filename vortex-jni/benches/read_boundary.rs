// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native "floor" for the `VortexJniReadBenchmark` JMH lanes.
//!
//! This reads the SAME canonical `.vortex` file the JMH benchmark reads (see [`canonical`]) and runs
//! the same lanes — full scan, native projection, and native filter — but entirely in Rust:
//! `scan -> Arrow RecordBatch`, with no JNI crossing and no Arrow C Data export. Comparing these
//! numbers against the JMH `ops/s` isolates the cost of the JNI + Arrow C Data boundary from the
//! underlying format read.
//!
//! Like the JMH side, every lane scans the full table and reports the produced row count, so
//! `ItemsCount::new(ROWS)` makes Divan print **input rows scanned per second** — directly comparable
//! to JMH's `@OperationsPerInvocation(ROWS)` `ops/s`. Each chunk is materialized to Arrow (forcing
//! the decode) and the batch row count is taken; no per-value work is done, so the numbers reflect
//! scan + materialization, not consume-side arithmetic.
//!
//! Each lane runs in two modes: single-threaded (no pool workers — the consuming thread drives the
//! scan, mirroring the JNI default) and `*_pooled` (a background `CurrentThreadWorkerPool` sized to
//! `available_parallelism() - 1`). The Vortex -> Arrow conversion runs inside the scan's `map`, which
//! executes on the handle-spawned split tasks, so the pool parallelizes both the decode and the Arrow
//! conversion. Utf8View columns are downgraded to flat Arrow `Utf8` via a stripped target field,
//! exactly as the JNI path does, so both benches materialize the same types.

#![expect(clippy::unwrap_used)]

mod canonical;

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::Array;
use arrow_schema::DataType;
use arrow_schema::Field;
use divan::Bencher;
use divan::counter::ItemsCount;
use vortex::VortexSessionDefault;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowSessionExt;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::current::CurrentThreadWorkerPool;
use vortex::io::session::RuntimeSessionExt;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::session::VortexSession;
use vortex::utils::parallelism::get_available_parallelism;

use crate::canonical::ROWS;

/// Shared current-thread runtime and its background worker pool, mirroring the JNI's static
/// `RUNTIME`/`POOL`. One long-lived pool is used (it has no `Drop`, so a per-sample pool would leak
/// worker threads); `set_workers` adjusts the count per lane — `0` for the single-threaded lanes,
/// `available_parallelism() - 1` for the pooled ones.
static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);
static POOL: LazyLock<CurrentThreadWorkerPool> = LazyLock::new(|| RUNTIME.new_pool());

fn main() {
    divan::main();
}

/// The lanes, mirroring `VortexJniReadBenchmark`.
#[derive(Clone, Copy)]
enum Lane {
    /// Read all six columns.
    FullScan,
    /// Native projection of `id, y`.
    Projection,
    /// Native filter `cat = 'alpha'` (~1/16 selectivity).
    SelectiveFilter,
}

/// Read state opened once per Divan sample (outside the timed region) and reused across iterations.
struct Env {
    session: VortexSession,
    file: VortexFile,
}

impl Env {
    fn open() -> VortexResult<Self> {
        let path = canonical::default_path();
        canonical::ensure_canonical(&path)?;

        let session = VortexSession::default().with_handle(RUNTIME.handle());
        let file = RUNTIME.block_on(
            session
                .open_options()
                .with_layout_reader_cache()
                .open_path(&path),
        )?;
        Ok(Self { session, file })
    }

    /// Scan the table for `lane`, materialize each chunk to Arrow, and return the total row count so
    /// the read is observable without per-value work.
    ///
    /// The Vortex -> Arrow conversion runs inside the scan's `map`, which executes within each split
    /// task spawned on the session's runtime handle — so with pool workers configured the decode AND
    /// the Arrow conversion run in parallel in the background, rather than inline on this thread.
    fn run(&self, lane: Lane) -> VortexResult<u64> {
        let mut builder = self.file.scan()?;
        match lane {
            Lane::Projection => builder = builder.with_projection(project_id_y()),
            Lane::SelectiveFilter => builder = builder.with_filter(filter_cat_alpha()),
            Lane::FullScan => {}
        }

        // Downgrade Utf8View -> Utf8 (and BinaryView -> Binary), matching the JNI Arrow boundary.
        let target = stripped_target(&builder.dtype()?)?;
        let session = self.session.clone();

        let mut rows = 0u64;
        for batch in builder
            .map(move |array| {
                let mut ctx = session.create_execution_ctx();
                let arrow = session
                    .arrow()
                    .execute_arrow(array, Some(&target), &mut ctx)?;
                Ok(arrow.len() as u64)
            })
            .into_iter(&*RUNTIME)?
        {
            rows += batch?;
        }
        Ok(rows)
    }
}

/// Build the Arrow target field for `execute_arrow`, replacing view string/binary types with their
/// flat equivalents so the materialized types match what the JNI path hands to Java.
fn stripped_target(dtype: &DType) -> VortexResult<Field> {
    let schema = dtype.to_arrow_schema()?;
    let stripped = strip_views(DataType::Struct(schema.fields().clone()));
    let DataType::Struct(fields) = stripped else {
        return Err(vortex_err!("scan dtype did not export as an Arrow struct"));
    };
    Ok(Field::new_struct("", fields, false))
}

fn strip_views(data_type: DataType) -> DataType {
    match data_type {
        DataType::Utf8View => DataType::Utf8,
        DataType::BinaryView => DataType::Binary,
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| {
                    Arc::new(Field::new(
                        f.name(),
                        strip_views(f.data_type().clone()),
                        f.is_nullable(),
                    ))
                })
                .collect(),
        ),
        other => other,
    }
}

fn project_id_y() -> Expression {
    select(vec![FieldName::from("id"), FieldName::from("y")], root())
}

fn filter_cat_alpha() -> Expression {
    Binary.new_expr(
        Operator::Eq,
        [get_item(FieldName::from("cat"), root()), lit("alpha")],
    )
}

/// Background worker count for the pooled lanes: one fewer than the available parallelism (the
/// consuming thread also drives the executor), at least one.
fn worker_threads() -> usize {
    get_available_parallelism()
        .map(|n| n.saturating_sub(1).max(1))
        .unwrap_or(1)
}

fn run_lane(bencher: Bencher<'_, '_>, lane: Lane, workers: usize) {
    POOL.set_workers(workers);
    bencher
        .with_inputs(|| Env::open().unwrap())
        .input_counter(|_| ItemsCount::new(ROWS))
        .bench_refs(move |env| env.run(lane).unwrap());
}

#[divan::bench]
fn full_scan(bencher: Bencher) {
    run_lane(bencher, Lane::FullScan, 0);
}

#[divan::bench]
fn projection(bencher: Bencher) {
    run_lane(bencher, Lane::Projection, 0);
}

#[divan::bench]
fn selective_filter(bencher: Bencher) {
    run_lane(bencher, Lane::SelectiveFilter, 0);
}

#[divan::bench]
fn full_scan_pooled(bencher: Bencher) {
    run_lane(bencher, Lane::FullScan, worker_threads());
}

#[divan::bench]
fn projection_pooled(bencher: Bencher) {
    run_lane(bencher, Lane::Projection, worker_threads());
}

#[divan::bench]
fn selective_filter_pooled(bencher: Bencher) {
    run_lane(bencher, Lane::SelectiveFilter, worker_threads());
}
