// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::TryStreamExt;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_io::request::IoSlot;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;

use super::*;
use crate::planning::gather::GatherPlanner;
use crate::planning::gather::Then;
use crate::planning::morsel::Morsel;
use crate::planning::morsel::MorselOutput;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;

/// A byte source that answers every fetch at once, counting the sessions opened on it and the
/// fetches it served.
#[derive(Default)]
struct InstantService {
    sessions: AtomicUsize,
    fetches: Arc<AtomicUsize>,
    /// Fails every fetch, to test error propagation.
    fail: bool,
}

impl InstantService {
    fn failing() -> Self {
        Self {
            fail: true,
            ..Default::default()
        }
    }
}

impl IoService for InstantService {
    fn session(&self) -> Arc<dyn IoSource> {
        self.sessions.fetch_add(1, Ordering::Relaxed);
        Arc::new(InstantSource {
            ready: Mutex::default(),
            fetches: Arc::clone(&self.fetches),
            fail: self.fail,
        })
    }
}

struct InstantSource {
    ready: Mutex<VecDeque<Completion>>,
    fetches: Arc<AtomicUsize>,
    fail: bool,
}

impl IoSource for InstantSource {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        for request in batch {
            if request.intent != IoIntent::Fetch {
                continue;
            }
            self.fetches.fetch_add(1, Ordering::Relaxed);
            let IoTarget::Range { len, .. } = request.target else {
                vortex_bail!("only ranges");
            };
            let result = if self.fail {
                Err(vortex_err!("read failed"))
            } else {
                Ok(IoResult::Bytes(BufferHandle::new_host(ByteBuffer::from(
                    vec![0u8; len],
                ))))
            };
            self.ready.lock().push_back(Completion {
                owner,
                request: request.request,
                result,
            });
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(self.ready.lock().pop_front())
    }

    fn poll_completion(&self, _cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        Poll::Ready(self.wait())
    }

    fn wait(&self) -> VortexResult<Completion> {
        self.ready
            .lock()
            .pop_front()
            .ok_or_else(|| vortex_err!("nothing outstanding"))
    }

    fn release(&self, owner: IoOwnerId) {
        self.ready.lock().retain(|c| c.owner != owner);
    }

    fn clear(&self) {
        self.ready.lock().clear();
    }
}

/// Reads one byte, then yields a batch of `rows` values counting from its scope's first row.
struct ReadThenBatch {
    scope: WorkScope,
    slot: IoSlot,
    issued: bool,
    done: bool,
}

impl ReadThenBatch {
    fn new(scope: WorkScope) -> Self {
        Self {
            scope,
            slot: IoSlot::default(),
            issued: false,
            done: false,
        }
    }
}

impl IoConsumer for ReadThenBatch {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.slot.deliver(request, result);
    }
}

impl Morsel for ReadThenBatch {
    fn state(&self) -> State {
        if self.done {
            State::Done
        } else if self.slot.is_waiting() {
            State::Waiting
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        if !self.issued {
            self.issued = true;
            return Ok(MorselOutput::NeedsIO(
                self.slot.issue(IoTarget::range(self.scope.rows.start, 1)),
            ));
        }
        self.slot.take();
        self.done = true;
        let start = u32::try_from(self.scope.rows.start)?;
        let end = u32::try_from(self.scope.rows.end)?;
        Ok(MorselOutput::Batch(
            PrimitiveArray::from_iter(start..end).into_array(),
        ))
    }
}

/// Emits its outputs in order, then finishes.
struct Emit(VecDeque<PlannerOutput>);

impl IoConsumer for Emit {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for Emit {
    fn state(&self) -> State {
        State::NeedsCompute
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        Ok(self.0.pop_front().unwrap_or(PlannerOutput::Done))
    }
}

fn scope(file: u64, rows: std::ops::Range<u64>) -> WorkScope {
    WorkScope {
        file_ordinal: file,
        rows,
    }
}

/// A file planner that spawns one independent chunk per `chunk` rows of `rows`, each a morsel.
fn file_planner(file: u64, rows: u64, chunk: u64) -> Box<dyn Planner> {
    let outputs = (0..rows.div_ceil(chunk))
        .map(|i| {
            let rows = i * chunk..((i + 1) * chunk).min(rows);
            let morsel = PlannerOutput::Morsel(
                scope(file, rows.clone()),
                Box::new(ReadThenBatch::new(scope(file, rows.clone()))),
            );
            PlannerOutput::Spawn(scope(file, rows), Box::new(Emit(VecDeque::from([morsel]))))
        })
        .collect();
    Box::new(Emit(outputs))
}

/// A table planner that opens `files` files of `rows` rows over their own services.
fn table(services: &[Arc<InstantService>], rows: u64, chunk: u64) -> PlanRoot {
    let outputs = services
        .iter()
        .enumerate()
        .map(|(file, service)| {
            let file = file as u64;
            let io: Arc<dyn IoService> = Arc::clone(service) as _;
            PlannerOutput::Open(scope(file, 0..rows), file_planner(file, rows, chunk), io)
        })
        .collect();
    let manifest: Arc<dyn IoService> = Arc::new(InstantService::default());
    PlanRoot::new(scope(0, 0..0), Box::new(Emit(outputs)), manifest)
}

fn services(files: usize) -> Vec<Arc<InstantService>> {
    (0..files)
        .map(|_| Arc::new(InstantService::default()))
        .collect()
}

/// `(file, first row, len)` of each batch.
fn shape(batches: &[ScopedBatch]) -> Vec<(u64, u64, usize)> {
    batches
        .iter()
        .map(|b| (b.scope.file_ordinal, b.scope.rows.start, b.batch.len()))
        .collect()
}

fn expected(files: u64, rows: u64, chunk: u64) -> Vec<(u64, u64, usize)> {
    (0..files)
        .flat_map(|file| {
            (0..rows.div_ceil(chunk)).map(move |i| {
                let start = i * chunk;
                (
                    file,
                    start,
                    usize::try_from(chunk.min(rows - start)).unwrap_or(0),
                )
            })
        })
        .collect()
}

#[test]
fn sequential_runs_every_file_through_its_own_service() -> VortexResult<()> {
    let services = services(3);
    let runtime = CurrentThreadRuntime::new();
    let batches: Vec<ScopedBatch> = runtime.block_on(
        SequentialDriver::new()
            .execute(table(&services, 10, 4))
            .try_collect(),
    )?;
    let mut shape = shape(&batches);
    shape.sort();
    assert_eq!(shape, expected(3, 10, 4));
    for service in &services {
        // The file's root, and each of its three chunks, read through a session of their own.
        assert_eq!(service.sessions.load(Ordering::Relaxed), 4);
        assert_eq!(service.fetches.load(Ordering::Relaxed), 3);
    }
    Ok(())
}

#[rstest]
#[case::one_worker(1, 1)]
#[case::many_workers(4, 2)]
#[case::more_workers_than_work(32, 3)]
fn parallel_runs_everything(#[case] workers: usize, #[case] roots: usize) -> VortexResult<()> {
    let services = services(5);
    let runtime = CurrentThreadRuntime::new();
    let pool = runtime.new_pool();
    pool.set_workers(4);
    let driver = ParallelDriver::new(runtime.handle())
        .with_workers(workers)
        .with_roots_per_worker(roots);
    let batches: Vec<ScopedBatch> =
        runtime.block_on(driver.execute(table(&services, 100, 7)).try_collect())?;
    let mut shape = shape(&batches);
    shape.sort();
    assert_eq!(shape, expected(5, 100, 7));
    Ok(())
}

#[test]
fn parallel_ordered_yields_by_file_and_row() -> VortexResult<()> {
    let services = services(4);
    let runtime = CurrentThreadRuntime::new();
    let pool = runtime.new_pool();
    pool.set_workers(4);
    let driver = ParallelDriver::new(runtime.handle())
        .with_workers(4)
        .with_ordered(true);
    let batches: Vec<ScopedBatch> =
        runtime.block_on(driver.execute(table(&services, 50, 6)).try_collect())?;
    assert_eq!(shape(&batches), expected(4, 50, 6));
    Ok(())
}

#[test]
fn parallel_maps_on_workers() -> VortexResult<()> {
    let services = services(2);
    let runtime = CurrentThreadRuntime::new();
    let driver = ParallelDriver::new(runtime.handle()).with_workers(2);
    let lens: Vec<ScopedBatch<usize>> = runtime.block_on(
        driver
            .execute_with(table(&services, 9, 3), Arc::new(|array| Ok(array.len())))
            .try_collect(),
    )?;
    assert_eq!(lens.iter().map(|b| b.batch).sum::<usize>(), 18);
    Ok(())
}

#[rstest]
#[case::sequential(false)]
#[case::parallel(true)]
fn read_errors_fail_the_stream(#[case] parallel: bool) {
    let services = vec![
        Arc::new(InstantService::default()),
        Arc::new(InstantService::failing()),
    ];
    let runtime = CurrentThreadRuntime::new();
    let root = table(&services, 8, 2);
    let stream = if parallel {
        ParallelDriver::new(runtime.handle())
            .with_workers(3)
            .execute(root)
    } else {
        SequentialDriver::new().execute(root)
    };
    let result: VortexResult<Vec<ScopedBatch>> = runtime.block_on(stream.try_collect());
    let error = result.err().map(|e| e.to_string());
    assert!(
        error.as_deref().is_some_and(|e| e.contains("read failed")),
        "{error:?}"
    );
}

/// An index read gathered inside a planner: its reads go through the index's service, its
/// batches reach only the continuation, in row order, and the continuation opens the files.
#[rstest]
#[case::sequential(false)]
#[case::parallel(true)]
fn a_gathered_read_decides_which_files_to_open(#[case] parallel: bool) -> VortexResult<()> {
    let index = Arc::new(InstantService::default());
    let files = services(3);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let then: Then = {
        let seen = Arc::clone(&seen);
        Box::new(move |batches| {
            // One file per gathered batch.
            seen.lock().extend(batches.iter().map(|batch| batch.len()));
            let opens = files
                .iter()
                .take(batches.len())
                .enumerate()
                .map(|(file, service)| {
                    let file = file as u64;
                    let io: Arc<dyn IoService> = Arc::clone(service) as _;
                    PlannerOutput::Open(scope(file, 0..10), file_planner(file, 10, 4), io)
                })
                .collect();
            Ok(Some(PlannerOutput::Planner(
                scope(0, 0..0),
                Box::new(Emit(opens)),
            )))
        })
    };
    // The index holds 25 rows in chunks of 10: three batches of 10, 10 and 5 rows.
    let gather = GatherPlanner::new(scope(9, 0..25), file_planner(9, 25, 10), then);
    let root = PlanRoot::new(scope(9, 0..25), Box::new(gather), Arc::clone(&index) as _);

    let runtime = CurrentThreadRuntime::new();
    let batches: Vec<ScopedBatch> = if parallel {
        let pool = runtime.new_pool();
        pool.set_workers(2);
        runtime.block_on(
            ParallelDriver::new(runtime.handle())
                .with_workers(2)
                .execute(root)
                .try_collect(),
        )?
    } else {
        runtime.block_on(SequentialDriver::new().execute(root).try_collect())?
    };

    assert_eq!(*seen.lock(), vec![10, 10, 5]);
    // The gathered index read is the planner's own: one session, one fetch per chunk.
    assert_eq!(index.sessions.load(Ordering::Relaxed), 1);
    assert_eq!(index.fetches.load(Ordering::Relaxed), 3);
    let mut shape = shape(&batches);
    shape.sort();
    assert_eq!(shape, expected(3, 10, 4));
    Ok(())
}
