// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan of a file for callers without an engine of their own.
//!
//! A planner says what it needs: compute, or reads. This driver decides where each runs. It
//! composes the planners over a [`PreparedScan`], the pruning plans first when the scan has a
//! filter and the chunks of the rows they keep after, and runs them on planning runs it owns:
//! the root run stepped on `spawn_cpu` tasks by a task of the runtime, each spawned chunk on a
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
use futures::SinkExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoSource;
use vortex_io::runtime::Handle;
use vortex_layout::scan::v2::ChunkPlanner;
use vortex_layout::scan::v2::PreparedScan;
use vortex_layout::scan::v2::PrunePlanner;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_mask::Mask;
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
}

impl ScanDriver {
    /// A driver spawning on `handle`, running `concurrency` chunks beside each other per worker
    /// thread, or every chunk on the scan's own run at one, and yielding batches in row order
    /// when `ordered`, which holds them until the scan is done.
    pub fn new(handle: Handle, concurrency: usize, ordered: bool) -> Self {
        assert!(concurrency > 0);
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
        let pump_handle = self.handle.clone();
        let capacity = self.concurrency * get_available_parallelism().unwrap_or(1);
        let started = Arc::new(xp::ScanClock::default());
        let clock = Arc::clone(&started);
        let inner = stream::once(async move {
            clock.start();
            let clock2 = Arc::clone(&clock);
            handle
                .spawn_cpu(move || {
                    let scan = Arc::new(builder.prepare()?);
                    VortexResult::Ok(self.execute(scan, map_fn, clock2))
                })
                .await
        })
        .try_flatten()
        .boxed();
        let inner = if xp::mode() == "pump" {
            on_runtime(&pump_handle, inner, capacity)
        } else {
            inner
        };
        inner
            .chain(stream::poll_fn(move |_| {
                started.finish();
                Poll::Ready(None)
            }))
            .boxed()
    }

    fn execute<A: Send + 'static>(
        self,
        scan: Arc<PreparedScan>,
        map_fn: MapFn<A>,
        clock: Arc<xp::ScanClock>,
    ) -> BoxStream<'static, VortexResult<A>> {
        let in_flight = self.concurrency * get_available_parallelism().unwrap_or(1);
        let in_flight = if xp::mode() == "inline16" { in_flight.min(16) } else { in_flight };
        let inline = in_flight <= 1;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: scan.rows().clone(),
        };
        let io = scan.io();
        let root = compose(&scan);
        let (chunk_handle, chunk_io, chunk_map) =
            (self.handle.clone(), Arc::clone(&io), Arc::clone(&map_fn));
        let root = run_stream(self.handle.clone(), root, scope, io.session(), map_fn, inline);
        let root = if xp::mode() == "root" {
            on_runtime(&self.handle, root, in_flight)
        } else {
            root
        };
        let batches = root
            .map_ok(move |yielded| match yielded {
                Yielded::Batch(start, batch) => {
                    stream::once(future::ready(Ok((start, batch)))).boxed()
                }
                Yielded::Spawn(scope, chunk) => {
                    clock.chunk_spawned();
                    chunk_stream(
                    &chunk_handle,
                    chunk,
                    scope,
                    chunk_io.session(),
                    Arc::clone(&chunk_map),
                )
                }
            })
            .try_flatten_unordered(in_flight);
        // The prepared scan holds the file's shared cache: keep it until the stream ends, so
        // scans of the file that start meanwhile find it.
        let batches = batches.chain(stream::poll_fn(move |_| {
            let _ = &scan;
            Poll::Ready(None)
        }));
        if !self.ordered {
            return batches.map_ok(|(_, batch)| batch).boxed();
        }
        stream::once(async move {
            let mut held: Vec<(u64, A)> = batches.try_collect().await?;
            // Morsels finish in whatever order their reads arrive.
            held.sort_by_key(|(start, _)| *start);
            VortexResult::Ok(stream::iter(held.into_iter().map(|(_, batch)| Ok(batch))))
        })
        .try_flatten()
        .boxed()
    }
}

/// The planners of a scan, composed: with a filter, its pruning plans first over the whole scan,
/// continuing into the chunks of the rows they keep; without, the chunks directly.
fn compose(scan: &Arc<PreparedScan>) -> Box<dyn Planner> {
    let rows = scan.rows().clone();
    let chunks = {
        let scan = Arc::clone(scan);
        let rows = rows.clone();
        move |kept: Option<Mask>| -> Box<dyn Planner> {
            let selection = scan.selection().clone();
            Box::new(ChunkPlanner::new(scan, rows, selection, kept))
        }
    };
    if scan.filter().is_some() {
        Box::new(PrunePlanner::new(Arc::clone(scan), rows, Box::new(chunks)))
    } else {
        chunks(None)
    }
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
            let rtt = std::time::Instant::now();
            let step_fn = move || {
                    let cpu = std::time::Instant::now();
                    let mut run = run;
                    let step = run.advance().and_then(|progress| {
                        Ok(match progress {
                            Progress::Batch(batch) => {
                                Step::Batch(batch.scope.rows.start, map_fn(batch.array)?)
                            }
                            Progress::Spawn(scope, child) => Step::Spawn(scope, child),
                            Progress::RootDone(_) => Step::RootDone,
                            Progress::Idle => Step::Idle,
                            Progress::Waiting => Step::Waiting,
                        })
                    });
                    xp::add(&xp::STATS.root_cpu_ns, cpu.elapsed());
                    (run, step)
                };
            let (run, step) = if xp::mode().starts_with("inline") {
                step_fn()
            } else {
                state.handle.spawn_cpu(step_fn).await
            };
            xp::STATS.root_steps.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            xp::add(&xp::STATS.root_rtt_ns, rtt.elapsed());
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

/// Polls `stream` on a task of the runtime and forwards its items through a channel of
/// `capacity`.
///
/// The root run only takes its next step when its stream is polled. Stepping it on its own task
/// keeps the caller, which may poll from outside the runtime, from paying a wake-up round trip
/// for every chunk the root run launches. The chunks themselves are still spawned by the caller,
/// so they reach the runtime as before, and the task, which only the root run's own steps and
/// reads wake, is never queued behind a chunk's work. The task starts on the first poll and is aborted when the
/// returned stream is dropped; a panic in it is re-raised when the stream reaches its end.
fn on_runtime<T: Send + 'static>(
    handle: &Handle,
    stream: BoxStream<'static, T>,
    capacity: usize,
) -> BoxStream<'static, T> {
    let handle = handle.clone();
    stream::once(async move {
        let (mut send, recv) = mpsc::channel(capacity);
        let task = handle.spawn(async move {
            let mut stream = stream;
            while let Some(item) = stream.next().await {
                if send.send(item).await.is_err() {
                    return;
                }
            }
        });
        recv.chain(stream::once(task).filter_map(|()| future::ready(None)))
    })
    .flatten()
    .boxed()
}

/// EXPERIMENT: diagnostics for the CI A/B. Not for merge.
#[doc(hidden)]
#[allow(clippy::disallowed_methods, missing_docs)]
pub mod xp {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::Duration;
    use std::time::Instant;

    pub fn mode() -> String {
        std::env::var("XP_MODE").unwrap_or_else(|_| "base".to_string())
    }

    #[derive(Default)]
    pub struct Stats {
        pub scans: AtomicU64,
        pub scan_ns: AtomicU64,
        pub root_steps: AtomicU64,
        pub root_cpu_ns: AtomicU64,
        pub root_rtt_ns: AtomicU64,
        pub chunks: AtomicU64,
        pub chunk_wait_ns: AtomicU64,
        pub chunk_run_ns: AtomicU64,
        pub first_spawn_ns: AtomicU64,
        pub last_spawn_ns: AtomicU64,
        pub chunk_cpu_ns: AtomicU64,
        pub chunk_map_ns: AtomicU64,
        pub chunk_send_ns: AtomicU64,
        pub chunk_io_ns: AtomicU64,
        pub chunk_waits: AtomicU64,
        pub chunk_tcpu_ns: AtomicU64,
    }

    /// This thread's CPU time in nanoseconds.
    #[allow(unsafe_code)]
    pub fn thread_cpu_ns() -> u64 {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: `ts` is a valid, writable timespec.
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        u64::try_from(ts.tv_sec).unwrap_or(0) * 1_000_000_000 + u64::try_from(ts.tv_nsec).unwrap_or(0)
    }

    pub static STATS: Stats = Stats {
        scans: AtomicU64::new(0),
        scan_ns: AtomicU64::new(0),
        root_steps: AtomicU64::new(0),
        root_cpu_ns: AtomicU64::new(0),
        root_rtt_ns: AtomicU64::new(0),
        chunks: AtomicU64::new(0),
        chunk_wait_ns: AtomicU64::new(0),
        chunk_run_ns: AtomicU64::new(0),
        first_spawn_ns: AtomicU64::new(0),
        last_spawn_ns: AtomicU64::new(0),
        chunk_cpu_ns: AtomicU64::new(0),
        chunk_map_ns: AtomicU64::new(0),
        chunk_send_ns: AtomicU64::new(0),
        chunk_io_ns: AtomicU64::new(0),
        chunk_waits: AtomicU64::new(0),
        chunk_tcpu_ns: AtomicU64::new(0),
    };

    pub fn add(counter: &AtomicU64, d: Duration) {
        counter.fetch_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX), Relaxed);
    }

    pub struct Timer<'a>(&'a AtomicU64, Instant);
    impl<'a> Timer<'a> {
        pub fn new(c: &'a AtomicU64) -> Self {
            Self(c, Instant::now())
        }
    }
    impl Drop for Timer<'_> {
        fn drop(&mut self) {
            add(self.0, self.1.elapsed());
        }
    }

    /// One scan's start, and its first and last chunk launch offsets.
    #[derive(Default)]
    pub struct ScanClock {
        start: parking_lot::Mutex<Option<Instant>>,
        first: AtomicU64,
        last: AtomicU64,
        done: std::sync::atomic::AtomicBool,
    }
    impl ScanClock {
        pub fn start(&self) {
            *self.start.lock() = Some(Instant::now());
        }
        pub fn chunk_spawned(&self) {
            if let Some(s) = *self.start.lock() {
                let ns = u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX);
                let _ = self.first.compare_exchange(0, ns, Relaxed, Relaxed);
                self.last.store(ns, Relaxed);
            }
        }
        pub fn finish(&self) {
            if self.done.swap(true, Relaxed) {
                return;
            }
            if let Some(s) = *self.start.lock() {
                STATS.scans.fetch_add(1, Relaxed);
                add(&STATS.scan_ns, s.elapsed());
                STATS.first_spawn_ns.fetch_add(self.first.load(Relaxed), Relaxed);
                STATS.last_spawn_ns.fetch_add(self.last.load(Relaxed), Relaxed);
            }
        }
    }

    pub fn reset() {
        for c in [
            &STATS.scans, &STATS.scan_ns, &STATS.root_steps, &STATS.root_cpu_ns,
            &STATS.root_rtt_ns, &STATS.chunks, &STATS.chunk_wait_ns, &STATS.chunk_run_ns,
            &STATS.first_spawn_ns, &STATS.last_spawn_ns, &STATS.chunk_cpu_ns,
            &STATS.chunk_map_ns, &STATS.chunk_send_ns, &STATS.chunk_io_ns, &STATS.chunk_waits,
            &STATS.chunk_tcpu_ns,
        ] {
            c.store(0, Relaxed);
        }
        vortex_layout::scan::v2::xp_layout::reset();
    }

    /// Per-scan averages, microseconds unless counted.
    pub fn summary() -> String {
        let g = |c: &AtomicU64| c.load(Relaxed) as f64;
        let scans = g(&STATS.scans).max(1.0);
        let chunks = g(&STATS.chunks).max(1.0);
        let steps = g(&STATS.root_steps).max(1.0);
        format!(
            "scans={} scan_us={:.1} root_steps={:.1} root_cpu_us/step={:.2} root_rtt_us/step={:.2} chunks={:.1} chunk_wait_us={:.2} chunk_run_us={:.1} first_spawn_us={:.1} last_spawn_us={:.1} chunk_cpu_us={:.1} chunk_map_us={:.1} chunk_send_us={:.1} chunk_io_us={:.1} chunk_waits={:.2} chunk_tcpu_us={:.1} requests={:.1} request_us={:.2} retain_read_wait_us={:.2} retain_write_wait_us={:.2}",
            g(&STATS.scans),
            g(&STATS.scan_ns) / scans / 1e3,
            g(&STATS.root_steps) / scans,
            g(&STATS.root_cpu_ns) / steps / 1e3,
            g(&STATS.root_rtt_ns) / steps / 1e3,
            g(&STATS.chunks) / scans,
            g(&STATS.chunk_wait_ns) / chunks / 1e3,
            g(&STATS.chunk_run_ns) / chunks / 1e3,
            g(&STATS.first_spawn_ns) / scans / 1e3,
            g(&STATS.last_spawn_ns) / scans / 1e3,
            g(&STATS.chunk_cpu_ns) / chunks / 1e3,
            g(&STATS.chunk_map_ns) / chunks / 1e3,
            g(&STATS.chunk_send_ns) / chunks / 1e3,
            g(&STATS.chunk_io_ns) / chunks / 1e3,
            g(&STATS.chunk_waits) / chunks,
            g(&STATS.chunk_tcpu_ns) / chunks / 1e3,
            {
                use vortex_layout::scan::v2::xp_layout as l;
                g(&l::REQUESTS) / scans
            },
            {
                use vortex_layout::scan::v2::xp_layout as l;
                g(&l::REQUEST_NS) / g(&l::REQUESTS).max(1.0) / 1e3
            },
            g(&vortex_layout::scan::v2::xp_layout::RETAIN_READ_WAIT_NS) / scans / 1e3,
            g(&vortex_layout::scan::v2::xp_layout::RETAIN_WRITE_WAIT_NS) / scans / 1e3,
        )
    }
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
    let spawned = std::time::Instant::now();
    let task = handle.spawn(async move {
        xp::add(&xp::STATS.chunk_wait_ns, spawned.elapsed());
        xp::STATS.chunks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _run_timer = xp::Timer::new(&xp::STATS.chunk_run_ns);
        let mut run = Run::new();
        run.admit(root, scope, Arc::clone(&io));
        loop {
            let t = std::time::Instant::now();
            let c = xp::thread_cpu_ns();
            let progress = run.advance()?;
            xp::add(&xp::STATS.chunk_cpu_ns, t.elapsed());
            xp::STATS.chunk_tcpu_ns.fetch_add(xp::thread_cpu_ns().saturating_sub(c), std::sync::atomic::Ordering::Relaxed);
            match progress {
                Progress::Batch(batch) => {
                    let start = batch.scope.rows.start;
                    let t = std::time::Instant::now();
                    let batch = map_fn(batch.array);
                    xp::add(&xp::STATS.chunk_map_ns, t.elapsed());
                    let failed = batch.is_err();
                    let t = std::time::Instant::now();
                    let sent = send.send(batch.map(|batch| (start, batch))).await;
                    xp::add(&xp::STATS.chunk_send_ns, t.elapsed());
                    if sent.is_err() || failed {
                        return Ok(());
                    }
                }
                Progress::Spawn(scope, child) => {
                    run.admit(child, scope, Arc::clone(&io));
                }
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => return Ok(()),
                Progress::Waiting => {
                    let t = std::time::Instant::now();
                    future::poll_fn(|cx| run.poll_completion(cx)).await?;
                    xp::add(&xp::STATS.chunk_io_ns, t.elapsed());
                    xp::STATS.chunk_waits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    });
    let errors = stream::once(task)
        .filter_map(|result: VortexResult<()>| future::ready(result.err().map(Err)));
    stream::select(recv, errors).boxed()
}

#[cfg(test)]
mod tests;
