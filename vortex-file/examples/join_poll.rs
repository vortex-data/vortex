// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A microbenchmark of DuckDB's join-and-poll pattern over Vortex scans, without DuckDB or I/O.
//! Diagnostic, not for merging.
//!
//! N threads scan M in-memory files of K parts each the way DuckDB's multi-file reader drives
//! Vortex: a thread joins a file under one global lock, then polls what it joined until it is
//! done, and joins again. Four drivers reproduce `vortex-duckdb`'s scan:
//!
//! - `dev-v1`: develop's V1. Each file's split futures, from `ScanBuilder::prepare().execute()`,
//!   in a list a join pops from; one `block_on` per split.
//! - `dev-v2`: develop's V2 (`scan_v2.rs`). Each file's parts in a queue cut up front; a join is
//!   allowed while the queue is not exhausted; each thread keeps one `Run` across files and
//!   enters `block_on` only when its run waits for reads.
//! - `head-v1` / `head-v2`: the head's `FileScanState`. Each file's `ScanStream`, opened from
//!   `LayoutReaderDataSource` or `VortexFile::plan_data_source`, behind a mutex; a join clones
//!   it, bounded by its claims; one `block_on(stream.next())` per batch.
//!
//! `--fast` polls each future once before `block_on`, as #11896 does, and `--force-pend` makes
//! every first poll pend so that the fast path cannot skip the executor.
//!
//! ```text
//! cargo run --release -p vortex-file --example join_poll -- --threads 1,8,18,48,96
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation
)]

use std::collections::VecDeque;
use std::future;
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::pin;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Poll;
use std::time::Duration;
use std::time::Instant;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_btrblocks::CompressionSession;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::request::IoService;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutReaderRef;
use vortex_layout::scan::layout::LayoutReaderDataSource;
use vortex_layout::scan::scan_builder::ScanBuilder as V1ScanBuilder;
use vortex_layout::session::LayoutSession;
use vortex_scan::DataSource;
use vortex_scan::DataSourceRef;
use vortex_scan::ScanRequest;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::prepare::ScanBuilder as V2ScanBuilder;
use vortex_scan::prepare::part_root;
use vortex_scan::stream::Claims;
use vortex_scan::stream::ScanStream;
use vortex_session::VortexSession;

static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>()
        .with::<RuntimeSession>()
        .with::<CompressionSession>()
        .with::<LayoutSession>()
        .with_handle(RUNTIME.handle());
    vortex_file::register_default_encodings(&session);
    session
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    DevV1,
    DevV2,
    HeadV1,
    HeadV2,
}

impl Path {
    fn parse(s: &str) -> Self {
        match s {
            "dev-v1" => Self::DevV1,
            "dev-v2" => Self::DevV2,
            "head-v1" => Self::HeadV1,
            "head-v2" => Self::HeadV2,
            other => panic!("unknown path {other}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::DevV1 => "dev-v1",
            Self::DevV2 => "dev-v2",
            Self::HeadV1 => "head-v1",
            Self::HeadV2 => "head-v2",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// No filter: every part returns its rows.
    Tiny,
    /// A filter that zone maps cannot prune and that no row passes: every part returns empty.
    Empty,
}

struct Args {
    paths: Vec<Path>,
    threads: Vec<usize>,
    files: usize,
    parts: usize,
    rows: usize,
    mode: Mode,
    fast: bool,
    force_pend: bool,
    reps: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        paths: vec![Path::DevV1, Path::HeadV1, Path::DevV2, Path::HeadV2],
        threads: vec![1, 8, 18, 48, 96],
        files: 64,
        parts: 32,
        rows: 256,
        mode: Mode::Tiny,
        fast: false,
        force_pend: false,
        reps: 3,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().expect("flag needs a value");
        match flag.as_str() {
            "--paths" => args.paths = value().split(',').map(Path::parse).collect(),
            "--threads" => args.threads = value().split(',').map(|n| n.parse().unwrap()).collect(),
            "--files" => args.files = value().parse().unwrap(),
            "--parts" => args.parts = value().parse().unwrap(),
            "--rows" => args.rows = value().parse().unwrap(),
            "--mode" => {
                args.mode = match value().as_str() {
                    "tiny" => Mode::Tiny,
                    "empty" => Mode::Empty,
                    other => panic!("unknown mode {other}"),
                }
            }
            "--fast" => args.fast = true,
            "--force-pend" => args.force_pend = true,
            "--reps" => args.reps = value().parse().unwrap(),
            "--help" | "-h" => {
                eprintln!(
                    "flags: --paths dev-v1,head-v1,dev-v2,head-v2 --threads 1,8,18,48,96 \
                     --files 64 --parts 32 --rows 256 --mode tiny|empty --fast --force-pend \
                     --reps 3"
                );
                std::process::exit(0);
            }
            other => panic!("unknown flag {other}"),
        }
    }
    args
}

fn uncompressed(chunk: &ArrayRef, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    Ok(chunk.clone())
}

/// One file of `parts` chunks of `rows` rows. Values are even and alternate in sign, so every zone
/// of two rows or more spans 1, which zone maps cannot prune `v = 1` from, and no row passes it.
fn make_file(parts: usize, rows: usize) -> VortexResult<VortexFile> {
    let chunks = (0..parts).map(|part| {
        let start = (part * rows) as i64;
        let v = Buffer::from_iter(
            (start..start + rows as i64).map(|i| if i % 2 == 0 { -2 * i - 2 } else { 2 * i }),
        )
        .into_array();
        StructArray::from_fields(&[("v", v)]).unwrap().into_array()
    });
    let array = ChunkedArray::from_iter(chunks).into_array();
    let strategy = vortex_file::WriteStrategyBuilder::from_session(&SESSION)
        .with_compressor(uncompressed)
        .with_row_block_size(rows)
        .with_data_block_target_bytes(None)
        .build();
    let mut buf = ByteBufferMut::empty();
    RUNTIME.block_on(
        SESSION
            .write_options()
            .disable_editions()
            .with_strategy(strategy)
            .write(&mut buf, array.to_array_stream()),
    )?;
    Ok(SESSION.open_options().open_buffer(buf)?.with_caching())
}

/// What each file keeps across reps.
struct File {
    file: VortexFile,
    reader: LayoutReaderRef,
    plan_source: DataSourceRef,
}

/// Counts of one thread, summed over the run.
#[derive(Default, Clone, Copy)]
struct Stats {
    joins: u64,
    /// Joins whose split or stream returned no batch.
    wasted_joins: u64,
    batches: u64,
    empty_batches: u64,
    /// Time holding the global lock to join, including the join itself.
    join_ns: u64,
    /// Time in the first poll after a join, until it returned a batch or ended.
    first_poll_ns: u64,
    /// Time in every later poll.
    poll_ns: u64,
    /// `block_on` entries, and how many of their first polls pended.
    block_ons: u64,
    pended: u64,
}

impl Stats {
    fn add(&mut self, other: &Self) {
        self.joins += other.joins;
        self.wasted_joins += other.wasted_joins;
        self.batches += other.batches;
        self.empty_batches += other.empty_batches;
        self.join_ns += other.join_ns;
        self.first_poll_ns += other.first_poll_ns;
        self.poll_ns += other.poll_ns;
        self.block_ons += other.block_ons;
        self.pended += other.pended;
    }

    fn batch(&mut self, array: &ArrayRef) {
        self.batches += 1;
        if array.is_empty() {
            self.empty_batches += 1;
        }
    }
}

/// A future that pends once, waking itself, and then completes.
fn yield_once() -> impl Future<Output = ()> {
    let mut yielded = false;
    future::poll_fn(move |cx| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
}

#[derive(Clone, Copy)]
struct Drive {
    fast: bool,
    force_pend: bool,
}

impl Drive {
    /// Blocks on `fut` on the shared runtime the way `vortex-duckdb` does, counting whether its
    /// first poll pends. With `fast`, polls it once first and enters the executor only if it pends.
    fn block_on<F: Future>(self, fut: F, stats: &mut Stats) -> F::Output {
        stats.block_ons += 1;
        let force = self.force_pend;
        let mut fut = pin!(async move {
            if force {
                yield_once().await;
            }
            fut.await
        });
        if self.fast {
            if let Some(output) = fut.as_mut().now_or_never() {
                return output;
            }
            stats.pended += 1;
            return RUNTIME.block_on(fut);
        }
        let mut first = true;
        let mut pended = false;
        let output = RUNTIME.block_on(future::poll_fn(|cx| {
            let poll = fut.as_mut().poll(cx);
            if first {
                first = false;
                pended = poll.is_pending();
            }
            poll
        }));
        stats.pended += u64::from(pended);
        output
    }
}

// ---- develop V1 -------------------------------------------------------------------------------

type SplitTask = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

struct DevV1File {
    splits: Mutex<Vec<SplitTask>>,
}

fn dev_v1_init(
    file: &File,
    projection: &BoundExpression,
    filter: Option<&BoundExpression>,
) -> VortexResult<DevV1File> {
    let scan = V1ScanBuilder::new(SESSION.clone(), Arc::clone(&file.reader))
        .with_projection(projection.clone())
        .with_some_filter(filter.cloned())
        .prepare()?;
    let mut splits = scan.execute(None)?;
    splits.reverse();
    Ok(DevV1File {
        splits: Mutex::new(splits),
    })
}

// ---- develop V2 -------------------------------------------------------------------------------

enum Work {
    Chunk(WorkScope, Box<dyn Planner>),
    Batch(ArrayRef),
}

/// Develop's `scan_v2::FileScan`, on the head's `part_root`.
struct DevV2File {
    io: Arc<dyn IoService>,
    queue: Mutex<VecDeque<Work>>,
    exhausted: AtomicBool,
}

fn dev_v2_init(
    file: &File,
    projection: &BoundExpression,
    filter: Option<&BoundExpression>,
    drive: Drive,
) -> VortexResult<DevV2File> {
    let splits = V1ScanBuilder::new(SESSION.clone(), Arc::clone(&file.reader))
        .with_projection(projection.clone())
        .with_some_filter(filter.cloned())
        .full_file_splits()?;
    let mut scan_file = file.file.scan_file();
    scan_file.cache.get_or_insert_default();
    let scan = Arc::new(
        V2ScanBuilder::new(SESSION.clone(), scan_file)
            .with_projection(projection.clone())
            .with_some_filter(filter.cloned())
            .with_row_range(0..file.file.row_count())
            .prepare()?,
    );
    let io = scan.io();
    let mut root = Run::new();
    root.admit(
        part_root(&scan, &splits, false),
        WorkScope {
            file_ordinal: 0,
            rows: scan.rows().clone(),
        },
        io.session(),
    );
    let mut queue = VecDeque::new();
    let mut ignored = Stats::default();
    loop {
        match root.advance()? {
            Progress::Batch(batch) => queue.push_back(Work::Batch(batch.array)),
            Progress::Spawn(scope, chunk) => queue.push_back(Work::Chunk(scope, chunk)),
            Progress::RootDone(_) if root.live_roots() > 0 => {}
            Progress::RootDone(_) | Progress::Idle => break,
            Progress::Waiting => {
                drive.block_on(future::poll_fn(|cx| root.poll_completion(cx)), &mut ignored)?;
            }
        }
    }
    Ok(DevV2File {
        io,
        exhausted: AtomicBool::new(queue.is_empty()),
        queue: Mutex::new(queue),
    })
}

impl DevV2File {
    fn next_work(&self) -> Option<Work> {
        let mut queue = self.queue.lock();
        let work = queue.pop_front()?;
        if queue.is_empty() {
            self.exhausted.store(true, Ordering::Release);
        }
        Some(work)
    }

    fn next_array(
        &self,
        run: &mut Run,
        drive: Drive,
        stats: &mut Stats,
    ) -> VortexResult<Option<ArrayRef>> {
        loop {
            while run.live_roots() < 1 {
                match self.next_work() {
                    Some(Work::Chunk(scope, chunk)) => run.admit(chunk, scope, self.io.session()),
                    Some(Work::Batch(array)) => return Ok(Some(array)),
                    None => break,
                }
            }
            match run.advance()? {
                Progress::Batch(batch) => return Ok(Some(batch.array)),
                Progress::Spawn(scope, child) => run.admit(child, scope, self.io.session()),
                Progress::RootDone(_) => {}
                Progress::Idle => return Ok(None),
                Progress::Waiting => {
                    drive.block_on(future::poll_fn(|cx| run.poll_completion(cx)), stats)?;
                }
            }
        }
    }
}

// ---- head -------------------------------------------------------------------------------------

/// The head's `FileScanState`, as far as joining and polling go.
struct HeadFile {
    stream: Mutex<ScanStream>,
    finished: AtomicBool,
    joined: AtomicUsize,
    joins_chunks: bool,
}

impl HeadFile {
    fn next_split(&self) -> Option<ScanStream> {
        if self.finished.load(Ordering::Acquire) {
            return None;
        }
        let stream = self.stream.lock();
        if let Some(Claims { claimed, total }) = stream.claims()
            && (claimed >= total || self.joined.load(Ordering::Relaxed) >= total)
        {
            return None;
        }
        self.joined.fetch_add(1, Ordering::Relaxed);
        Some(stream.clone())
    }
}

fn head_init(
    file: &File,
    v2: bool,
    projection: &Expression,
    filter: Option<&Expression>,
    drive: Drive,
) -> VortexResult<HeadFile> {
    let request = ScanRequest {
        projection: projection.clone(),
        filter: filter.cloned(),
        row_range: Some(0..file.file.row_count()),
        ordered: false,
        split_concurrency: Some(NonZeroUsize::MIN),
        ..ScanRequest::default()
    };
    let mut ignored = Stats::default();
    let stream = if v2 {
        drive
            .block_on(file.plan_source.scan(request), &mut ignored)?
            .open()?
    } else {
        let source = LayoutReaderDataSource::new(Arc::clone(&file.reader), SESSION.clone());
        drive.block_on(source.scan(request), &mut ignored)?.open()?
    };
    Ok(HeadFile {
        stream: Mutex::new(stream),
        finished: AtomicBool::new(false),
        joined: AtomicUsize::new(0),
        joins_chunks: v2,
    })
}

// ---- the run ----------------------------------------------------------------------------------

enum Files {
    DevV1(Vec<DevV1File>),
    DevV2(Vec<DevV2File>),
    Head(Vec<HeadFile>),
}

/// One thread's scan, joining files in order under `cursor` until every file is done.
fn scan_thread(files: &Files, cursor: &Mutex<usize>, drive: Drive) -> VortexResult<Stats> {
    let mut stats = Stats::default();
    let mut run = Run::new();
    loop {
        let joined_at = Instant::now();
        let joined = {
            let mut current = cursor.lock();
            loop {
                let idx = *current;
                let joined = match files {
                    Files::DevV1(files) => match files.get(idx) {
                        None => break None,
                        Some(file) => file.splits.lock().pop().map(Joined::DevV1),
                    },
                    Files::DevV2(files) => match files.get(idx) {
                        None => break None,
                        Some(file) => {
                            (!file.exhausted.load(Ordering::Acquire)).then_some(Joined::DevV2(idx))
                        }
                    },
                    Files::Head(files) => match files.get(idx) {
                        None => break None,
                        Some(file) => file.next_split().map(|stream| Joined::Head(idx, stream)),
                    },
                };
                match joined {
                    Some(joined) => break Some(joined),
                    None => *current += 1,
                }
            }
        };
        stats.join_ns += joined_at.elapsed().as_nanos() as u64;
        let Some(joined) = joined else {
            return Ok(stats);
        };
        stats.joins += 1;
        let batches_before = stats.batches;
        let mut first = true;
        match joined {
            Joined::DevV1(task) => {
                let started = Instant::now();
                let array = drive.block_on(task, &mut stats)?;
                stats.first_poll_ns += started.elapsed().as_nanos() as u64;
                if let Some(array) = array {
                    stats.batch(&array);
                }
            }
            Joined::DevV2(idx) => {
                let Files::DevV2(files) = files else {
                    unreachable!()
                };
                let file = &files[idx];
                loop {
                    let started = Instant::now();
                    let array = file.next_array(&mut run, drive, &mut stats)?;
                    let elapsed = started.elapsed().as_nanos() as u64;
                    if std::mem::take(&mut first) {
                        stats.first_poll_ns += elapsed;
                    } else {
                        stats.poll_ns += elapsed;
                    }
                    let Some(array) = array else { break };
                    batch_chunks(&array, true, &mut stats);
                }
            }
            Joined::Head(idx, mut stream) => {
                let Files::Head(files) = files else {
                    unreachable!()
                };
                let file = &files[idx];
                loop {
                    let started = Instant::now();
                    let array = drive.block_on(stream.next(), &mut stats).transpose()?;
                    let elapsed = started.elapsed().as_nanos() as u64;
                    if std::mem::take(&mut first) {
                        stats.first_poll_ns += elapsed;
                    } else {
                        stats.poll_ns += elapsed;
                    }
                    let Some(array) = array else {
                        file.finished.store(true, Ordering::Release);
                        break;
                    };
                    batch_chunks(&array, file.joins_chunks, &mut stats);
                }
                drop(stream);
            }
        }
        if stats.batches == batches_before {
            stats.wasted_joins += 1;
        }
    }
}

/// Counts `array` as the batches DuckDB would see: a chunked V2 batch is exported chunk by chunk.
fn batch_chunks(array: &ArrayRef, joins_chunks: bool, stats: &mut Stats) {
    match array.as_opt::<Chunked>().filter(|_| joins_chunks) {
        Some(chunked) => {
            for chunk in chunked.chunks() {
                stats.batch(&chunk);
            }
        }
        None => stats.batch(array),
    }
}

enum Joined {
    DevV1(SplitTask),
    DevV2(usize),
    Head(usize, ScanStream),
}

struct Measured {
    wall: Duration,
    setup: Duration,
    stats: Stats,
}

fn run_once(path: Path, threads: usize, files: &[File], args: &Args) -> VortexResult<Measured> {
    let drive = Drive {
        fast: args.fast,
        force_pend: args.force_pend,
    };
    let dtype = files[0].file.dtype().clone();
    let projection_expr = root();
    let filter_expr = (args.mode == Mode::Empty).then(|| eq(col("v"), lit(1i64)));
    let projection = projection_expr.bind(&dtype)?.optimize_recursive()?;
    let filter = filter_expr
        .as_ref()
        .map(|f| f.bind(&dtype)?.optimize_recursive())
        .transpose()?;

    let setup_start = Instant::now();
    let scans = match path {
        Path::DevV1 => Files::DevV1(
            files
                .iter()
                .map(|f| dev_v1_init(f, &projection, filter.as_ref()))
                .collect::<VortexResult<_>>()?,
        ),
        Path::DevV2 => Files::DevV2(
            files
                .iter()
                .map(|f| dev_v2_init(f, &projection, filter.as_ref(), drive))
                .collect::<VortexResult<_>>()?,
        ),
        Path::HeadV1 | Path::HeadV2 => Files::Head(
            files
                .iter()
                .map(|f| {
                    head_init(
                        f,
                        path == Path::HeadV2,
                        &projection_expr,
                        filter_expr.as_ref(),
                        drive,
                    )
                })
                .collect::<VortexResult<_>>()?,
        ),
    };
    let setup = setup_start.elapsed();

    let cursor = Mutex::new(0usize);
    let barrier = Barrier::new(threads + 1);
    let (wall, stats) = std::thread::scope(|scope| -> VortexResult<_> {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (scans, cursor, barrier) = (&scans, &cursor, &barrier);
                scope.spawn(move || {
                    barrier.wait();
                    scan_thread(scans, cursor, drive)
                })
            })
            .collect();
        barrier.wait();
        let started = Instant::now();
        let mut total = Stats::default();
        for handle in handles {
            total.add(&handle.join().expect("scan thread panicked")?);
        }
        Ok((started.elapsed(), total))
    })?;
    Ok(Measured { wall, setup, stats })
}

fn main() -> VortexResult<()> {
    let args = parse_args();
    let files = (0..args.files)
        .map(|_| {
            let file = make_file(args.parts, args.rows)?;
            Ok(File {
                reader: file.layout_reader()?,
                plan_source: file.plan_data_source(),
                file,
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;
    println!(
        "# files={} parts/file={} rows/part={} mode={:?} fast={} force_pend={} reps={} (median wall)",
        args.files, args.parts, args.rows, args.mode, args.fast, args.force_pend, args.reps
    );
    println!(
        "{:<8} {:>3} {:>9} {:>9} {:>6} {:>7} {:>8} {:>7} {:>10} {:>12} {:>10} {:>6}",
        "path",
        "N",
        "wall_ms",
        "setup_us",
        "joins",
        "wasted",
        "batches",
        "empty",
        "ns/join",
        "ns/1st_poll",
        "ns/batch",
        "pend%"
    );
    for &path in &args.paths {
        for &threads in &args.threads {
            let mut runs = (0..args.reps)
                .map(|_| run_once(path, threads, &files, &args))
                .collect::<VortexResult<Vec<_>>>()?;
            runs.sort_by_key(|run| run.wall);
            let run = &runs[runs.len() / 2];
            let s = run.stats;
            let per = |ns: u64, n: u64| if n == 0 { 0.0 } else { ns as f64 / n as f64 };
            println!(
                "{:<8} {:>3} {:>9.2} {:>9.1} {:>6} {:>7} {:>8} {:>7} {:>10.0} {:>12.0} {:>10.0} {:>6.1}",
                path.name(),
                threads,
                run.wall.as_secs_f64() * 1e3,
                run.setup.as_secs_f64() * 1e6 / args.files as f64,
                s.joins,
                s.wasted_joins,
                s.batches,
                s.empty_batches,
                per(s.join_ns, s.joins),
                per(s.first_poll_ns, s.joins),
                per(s.first_poll_ns + s.poll_ns, s.batches),
                100.0 * per(s.pended, s.block_ons),
            );
        }
    }
    Ok(())
}
