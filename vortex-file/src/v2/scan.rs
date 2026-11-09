// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan of a file for callers without an engine of their own.
//!
//! A planner says what it needs: compute, or reads. This driver decides where each runs. It
//! composes the planners over a [`PreparedScan`], the pruning plans first when the scan has a
//! filter and the chunks of the rows they keep after, and runs them on planning runs it owns:
//! the root run stepped on `spawn_cpu` tasks as the stream is polled, each spawned chunk on a
//! task of its own, up to the concurrency, or on the root run at a concurrency of one. Every
//! read is a `spawn_io` task of the file's segment source. Batches the caller wants as Arrow are
//! converted on the task that produced them, beside the scan's other chunks.

use std::future;
use std::sync::Arc;
use std::task::Poll;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use futures::FutureExt;
use futures::SinkExt;
use futures::StreamExt;
use futures::TryFutureExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoSource;
use vortex_io::runtime::Handle;
use vortex_layout::scan::v2::PreparedScan;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_utils::parallelism::get_available_parallelism;

/// Runs a V2 scan on the session's runtime.
pub struct ScanDriver {
    handle: Handle,
    concurrency: usize,
    ordered: bool,
    limit: Option<u64>,
}

impl ScanDriver {
    /// A driver spawning on `handle`, running `concurrency` chunks beside each other per worker
    /// thread, or every chunk on the scan's own run at one, and yielding batches in row order
    /// when `ordered`.
    ///
    /// An ordered scan holds each chunk's batches until the chunk is done and every earlier chunk
    /// was yielded, so it holds at most the chunks running beside each other.
    pub fn new(handle: Handle, concurrency: usize, ordered: bool) -> Self {
        assert!(concurrency > 0);
        Self {
            handle,
            concurrency,
            ordered,
            limit: None,
        }
    }

    /// Stops the scan once it yielded `limit` rows, slicing the last batch to fit. An ordered
    /// scan yields the first `limit` rows it selects; an unordered scan yields any `limit` of
    /// them.
    pub fn with_limit(mut self, limit: Option<u64>) -> Self {
        self.limit = limit;
        self
    }

    /// Prepares `builder` where compute runs, composes its planners, and streams their batches.
    /// Preparation errors are the stream's first item.
    pub fn run(self, builder: ScanBuilder) -> BoxStream<'static, VortexResult<ArrayRef>> {
        let limit = self.limit;
        let batches = self.drive(builder, Arc::new(Ok));
        limit_rows(
            batches,
            limit,
            |batch| batch.len(),
            |batch, rows| batch.slice(0..rows),
        )
    }

    /// Like [`ScanDriver::run`], over a scan prepared already. Use it to reuse one preparation,
    /// or to handle preparation errors before the scan starts.
    pub fn run_prepared(
        self,
        scan: Arc<PreparedScan>,
    ) -> BoxStream<'static, VortexResult<ArrayRef>> {
        let limit = self.limit;
        let batches = self.execute(scan, Arc::new(Ok));
        limit_rows(
            batches,
            limit,
            |batch| batch.len(),
            |batch, rows| batch.slice(0..rows),
        )
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
        let limit = self.limit;
        let batches = self.drive(
            builder,
            Arc::new(move |batch| {
                let mut ctx = session.create_execution_ctx();
                let arrow = session
                    .arrow()
                    .execute_arrow(batch, Some(&field), &mut ctx)?;
                Ok(RecordBatch::from(arrow.as_struct().clone()))
            }),
        );
        limit_rows(batches, limit, RecordBatch::num_rows, |batch, rows| {
            Ok(batch.slice(0, rows))
        })
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
                    VortexResult::Ok(self.execute(scan, map_fn))
                })
                .await
        })
        .try_flatten()
        .boxed()
    }

    fn execute<A: Send + 'static>(
        self,
        scan: Arc<PreparedScan>,
        map_fn: MapFn<A>,
    ) -> BoxStream<'static, VortexResult<A>> {
        let in_flight = self.concurrency * get_available_parallelism().unwrap_or(1);
        // An ordered scan runs each chunk on a task of its own, so it can yield a chunk once it
        // and every chunk before it are done.
        let inline = in_flight <= 1 && !self.ordered;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: scan.rows().clone(),
        };
        let io = scan.io();
        let root = scan.planner();
        let (chunk_handle, chunk_io, chunk_map) =
            (self.handle.clone(), Arc::clone(&io), Arc::clone(&map_fn));
        let yielded = run_stream(self.handle, root, scope, io.session(), map_fn, inline);

        let batches = if self.ordered {
            yielded
                .map_ok(move |yielded| match yielded {
                    Yielded::Batch(start, batch) => future::ready(Ok(vec![(start, batch)])).boxed(),
                    Yielded::Spawn(scope, chunk) => chunk_stream(
                        &chunk_handle,
                        chunk,
                        scope,
                        chunk_io.session(),
                        Arc::clone(&chunk_map),
                    )
                    .try_collect::<Vec<_>>()
                    .map_ok(|mut held| {
                        // Morsels finish in whatever order their reads arrive.
                        held.sort_by_key(|(start, _)| *start);
                        held
                    })
                    .boxed(),
                })
                // The root run spawns chunks in row order, and buffering keeps that order.
                .try_buffered(in_flight)
                .map_ok(|held| stream::iter(held.into_iter().map(Ok)))
                .try_flatten()
                .boxed()
        } else {
            yielded
                .map_ok(move |yielded| match yielded {
                    Yielded::Batch(start, batch) => {
                        stream::once(future::ready(Ok((start, batch)))).boxed()
                    }
                    Yielded::Spawn(scope, chunk) => chunk_stream(
                        &chunk_handle,
                        chunk,
                        scope,
                        chunk_io.session(),
                        Arc::clone(&chunk_map),
                    ),
                })
                .try_flatten_unordered(in_flight)
                .boxed()
        };

        // The prepared scan holds the file's shared cache: keep it until the stream ends, so
        // scans of the file that start meanwhile find it.
        batches
            .chain(stream::poll_fn(move |_| {
                let _ = &scan;
                Poll::Ready(None)
            }))
            .map_ok(|(_, batch)| batch)
            .boxed()
    }
}

/// Ends `batches` once they hold `limit` rows, slicing the last batch to fit. Dropping the scan's
/// stream cancels the chunks still running.
fn limit_rows<A: Send + 'static>(
    batches: BoxStream<'static, VortexResult<A>>,
    limit: Option<u64>,
    len: fn(&A) -> usize,
    slice: fn(A, usize) -> VortexResult<A>,
) -> BoxStream<'static, VortexResult<A>> {
    let Some(limit) = limit else {
        return batches;
    };

    stream::unfold((batches, limit), move |(mut batches, left)| async move {
        if left == 0 {
            return None;
        }
        let batch = match batches.next().await? {
            Ok(batch) => batch,
            Err(error) => return Some((Err(error), (batches, 0))),
        };
        let rows = len(&batch) as u64;
        if rows <= left {
            return Some((Ok(batch), (batches, left - rows)));
        }
        let rows = usize::try_from(left).vortex_expect("fewer rows than the batch fit usize");
        Some((slice(batch, rows), (batches, 0)))
    })
    .boxed()
}

/// A mapping applied to each batch on the task that produced it.
type MapFn<A> = Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>;

/// What the root run's stream yields: a mapped batch with the row it starts at, or a chunk a
/// planner of the run spawned.
enum Yielded<A> {
    Batch(u64, A),
    Spawn(WorkScope, Box<dyn Planner>),
}

/// One CPU step of a run, taken on a `spawn_cpu` task with the run.
enum Step<A> {
    Batch(u64, A),
    Spawn(WorkScope, Box<dyn Planner>),
    RootDone,
    Idle,
    Waiting,
}

/// The state of the root run's stream.
struct RunState<A> {
    handle: Handle,
    /// Taken while a CPU step runs on its task and put back with its result.
    run: Option<Run>,
    io: Arc<dyn IoSource>,
    map_fn: MapFn<A>,
    /// Whether a spawned chunk runs on this run, as another root, instead of being yielded.
    inline: bool,
}

/// Runs `root` over `scope` on a planning run of its own, reading through `io`, each CPU step
/// on a `spawn_cpu` task, and yields its mapped batches and, unless `inline`, the chunks it
/// spawns.
fn run_stream<A: Send + 'static>(
    handle: Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
    map_fn: MapFn<A>,
    inline: bool,
) -> BoxStream<'static, VortexResult<Yielded<A>>> {
    let mut run = Run::new();
    run.admit(root, scope, Arc::clone(&io));
    let state = RunState {
        handle,
        run: Some(run),
        io,
        map_fn,
        inline,
    };
    stream::try_unfold(state, |mut state| async move {
        loop {
            let run = state
                .run
                .take()
                .ok_or_else(|| vortex_err!("the scan's run is missing"))?;
            let map_fn = Arc::clone(&state.map_fn);
            let (run, step) = state
                .handle
                .spawn_cpu(move || {
                    let mut run = run;
                    let step = run.advance().and_then(|progress| {
                        Ok(match progress {
                            Progress::Batch(batch) => {
                                Step::Batch(batch.scope.rows.start, map_fn(batch.array)?)
                            }
                            Progress::Spawn(_, scope, child) => Step::Spawn(scope, child),
                            Progress::RootDone(_) => Step::RootDone,
                            Progress::Idle => Step::Idle,
                            Progress::Waiting => Step::Waiting,
                        })
                    });
                    (run, step)
                })
                .await;
            let run = state.run.insert(run);
            match step? {
                Step::Batch(start, batch) => {
                    return Ok(Some((Yielded::Batch(start, batch), state)));
                }
                Step::Spawn(scope, child) if state.inline => {
                    run.admit(child, scope, Arc::clone(&state.io));
                }
                Step::Spawn(scope, child) => {
                    return Ok(Some((Yielded::Spawn(scope, child), state)));
                }
                Step::RootDone if run.live_roots() > 0 => {}
                Step::RootDone | Step::Idle => return Ok(None),
                Step::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
    })
    .boxed()
}

/// Runs a spawned chunk's `root` over `scope` on a planning run of its own, on one task that
/// drives the run to its end and awaits only its reads, and yields its mapped batches with the
/// row each starts at. A planner the chunk spawns in turn runs as another root of the same run.
fn chunk_stream<A: Send + 'static>(
    handle: &Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
    map_fn: MapFn<A>,
) -> BoxStream<'static, VortexResult<(u64, A)>> {
    let (mut send, recv) = mpsc::channel(1);
    let task = handle.spawn(async move {
        let mut run = Run::new();
        run.admit(root, scope, Arc::clone(&io));
        loop {
            match run.advance()? {
                Progress::Batch(batch) => {
                    let start = batch.scope.rows.start;
                    let batch = map_fn(batch.array);
                    let failed = batch.is_err();
                    if send.send(batch.map(|batch| (start, batch))).await.is_err() || failed {
                        return Ok(());
                    }
                }
                Progress::Spawn(_, scope, child) => {
                    run.admit(child, scope, Arc::clone(&io));
                }
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => return Ok(()),
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
    });
    let errors = stream::once(task)
        .filter_map(|result: VortexResult<()>| future::ready(result.err().map(Err)));
    stream::select(recv, errors).boxed()
}

#[cfg(test)]
mod tests;
