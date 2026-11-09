// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan of a file for callers without an engine of their own.
//!
//! A planner says what it needs: compute, or reads. This driver runs the scan's composed planners
//! ([`PreparedScan::planner`]) on a [`ParallelDriver`]: each chunk on a task of its own, up to
//! the concurrency, or every chunk on one run at a concurrency of one. Every read is a `spawn_io`
//! task of the file's segment source. Batches the caller wants as Arrow are converted on the task
//! that produced them, beside the scan's other chunks.

use std::future;
use std::sync::Arc;
use std::task::Poll;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_layout::scan::v2::PreparedScan;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_scan::planning::drive::MapFn;
use vortex_scan::planning::drive::ParallelDriver;
use vortex_utils::parallelism::get_available_parallelism;

/// Runs a V2 scan on the session's runtime.
pub struct ScanDriver {
    handle: Handle,
    concurrency: usize,
    ordered: bool,
}

impl ScanDriver {
    /// A driver spawning on `handle`, running `concurrency` chunks beside each other per worker
    /// thread, or every chunk on one run when that is at most one chunk in all (so at zero), and
    /// yielding batches in row order when `ordered`, which holds them until the scan is done.
    pub fn new(handle: Handle, concurrency: usize, ordered: bool) -> Self {
        Self {
            handle,
            concurrency,
            ordered,
        }
    }

    /// Prepares `builder` where compute runs, composes its planners, and streams their batches.
    /// Preparation errors are the stream's first item.
    pub fn run(self, builder: ScanBuilder) -> BoxStream<'static, VortexResult<ArrayRef>> {
        self.drive(builder, Arc::new(Ok))
    }

    /// Like [`ScanDriver::run`], yielding each batch executed to an Arrow [`RecordBatch`] of
    /// `schema`, or of the schema the session derives from the scan's dtype when `None`.
    ///
    /// Execution to Arrow is where a scan's batches are decompressed, so the driver runs it on
    /// the task that produced each batch, in parallel across chunks, rather than leaving it to
    /// the single task polling the stream.
    pub fn run_record_batches(
        self,
        builder: ScanBuilder,
        schema: Option<SchemaRef>,
    ) -> BoxStream<'static, VortexResult<RecordBatch>> {
        let session = builder.session().clone();
        let schema = match schema {
            Some(schema) => schema,
            None => match session.arrow().to_arrow_schema(builder.dtype()) {
                Ok(schema) => Arc::new(schema),
                Err(error) => return stream::once(future::ready(Err(error))).boxed(),
            },
        };
        let field = Field::new_struct("", schema.fields().clone(), false);
        self.drive(
            builder,
            Arc::new(move |batch| {
                let mut ctx = session.create_execution_ctx();
                let arrow = batch.execute_arrow(Some(&field), &mut ctx)?;
                Ok(RecordBatch::from(arrow.as_struct().clone()))
            }),
        )
    }

    /// Runs the scan, applying `map_fn` to each batch on the task that produced it.
    fn drive<A: Send + 'static>(
        self,
        builder: ScanBuilder,
        map_fn: MapFn<A>,
    ) -> BoxStream<'static, VortexResult<A>> {
        let handle = self.handle.clone();
        stream::once(async move {
            handle
                .spawn_cpu(move || {
                    let scan = Arc::new(builder.prepare()?);
                    VortexResult::Ok(self.run_prepared(scan, map_fn))
                })
                .await
        })
        .try_flatten()
        .boxed()
    }

    /// Streams the batches of a scan the caller prepared, applying `map_fn` to each on the task
    /// that produced it.
    pub fn run_prepared<A: Send + 'static>(
        self,
        scan: Arc<PreparedScan>,
        map_fn: MapFn<A>,
    ) -> BoxStream<'static, VortexResult<A>> {
        let batches = self
            .driver()
            .execute_with(scan.plan_root(), map_fn)
            .map_ok(|scoped| scoped.batch);
        // The prepared scan holds the file's shared cache: keep it until the stream ends, so
        // scans of the file that start meanwhile find it.
        batches
            .chain(stream::poll_fn(move |_| {
                let _ = &scan;
                Poll::Ready(None)
            }))
            .boxed()
    }

    /// The driver that runs `concurrency` chunks per worker thread at once, each on a run of its
    /// own, or every chunk on one run when that is at most one chunk.
    fn driver(&self) -> ParallelDriver {
        let in_flight = self.concurrency * get_available_parallelism().unwrap_or(1);
        let driver = ParallelDriver::new(self.handle.clone()).with_ordered(self.ordered);
        if in_flight <= 1 {
            driver.with_workers(1).with_roots_per_worker(usize::MAX)
        } else {
            // One more worker for the root planner, so it never takes a chunk's place while it
            // prunes and cuts chunks.
            driver.with_workers(in_flight + 1).with_roots_per_worker(1)
        }
    }
}

#[cfg(test)]
mod tests;
