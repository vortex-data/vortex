// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::StreamExt;
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

fn scope(rows: std::ops::Range<u64>) -> WorkScope {
    WorkScope {
        file_ordinal: 0,
        rows,
    }
}

/// A plan over `service` that spawns one independent chunk per `chunk` rows of `rows`, each a
/// morsel.
fn plan(service: &Arc<InstantService>, rows: u64, chunk: u64) -> PlanRoot {
    plan_with(service, rows, chunk, |scope| Box::new(ReadThenBatch::new(scope)))
}

/// Like [`plan`], with each chunk's morsel made by `morsel`.
fn plan_with(
    service: &Arc<InstantService>,
    rows: u64,
    chunk: u64,
    morsel: impl Fn(WorkScope) -> Box<dyn Morsel>,
) -> PlanRoot {
    let outputs = (0..rows.div_ceil(chunk))
        .map(|i| {
            let rows = i * chunk..((i + 1) * chunk).min(rows);
            let morsel = PlannerOutput::Morsel(scope(rows.clone()), morsel(scope(rows.clone())));
            PlannerOutput::Spawn(scope(rows), Box::new(Emit(VecDeque::from([morsel]))))
        })
        .collect();
    let io: Arc<dyn IoService> = Arc::clone(service) as _;
    PlanRoot::new(scope(0..rows), Box::new(Emit(outputs)), io)
}

/// `(first row, len)` of each batch.
fn shape(batches: &[ScopedBatch]) -> Vec<(u64, usize)> {
    batches
        .iter()
        .map(|b| (b.scope.rows.start, b.batch.len()))
        .collect()
}

fn expected(rows: u64, chunk: u64) -> Vec<(u64, usize)> {
    (0..rows.div_ceil(chunk))
        .map(|i| {
            let start = i * chunk;
            (start, usize::try_from(chunk.min(rows - start)).unwrap_or(0))
        })
        .collect()
}

#[test]
fn sequential_reads_each_root_through_its_own_session() -> VortexResult<()> {
    let service = Arc::new(InstantService::default());
    let runtime = CurrentThreadRuntime::new();
    let batches: Vec<ScopedBatch> = runtime.block_on(
        SequentialDriver::new()
            .execute(plan(&service, 10, 4))
            .try_collect(),
    )?;
    let mut shape = shape(&batches);
    shape.sort();
    assert_eq!(shape, expected(10, 4));
    // The root, and each of its three chunks, read through a session of their own.
    assert_eq!(service.sessions.load(Ordering::Relaxed), 4);
    assert_eq!(service.fetches.load(Ordering::Relaxed), 3);
    Ok(())
}

#[rstest]
#[case::one_worker(1, 1)]
#[case::many_workers(4, 2)]
#[case::more_workers_than_work(32, 3)]
fn parallel_runs_everything(#[case] workers: usize, #[case] roots: usize) -> VortexResult<()> {
    let service = Arc::new(InstantService::default());
    let runtime = CurrentThreadRuntime::new();
    let pool = runtime.new_pool();
    pool.set_workers(4);
    let driver = ParallelDriver::new(runtime.handle())
        .with_workers(workers)
        .with_roots_per_worker(roots);
    let batches: Vec<ScopedBatch> =
        runtime.block_on(driver.execute(plan(&service, 100, 7)).try_collect())?;
    let mut shape = shape(&batches);
    shape.sort();
    assert_eq!(shape, expected(100, 7));
    assert_eq!(service.sessions.load(Ordering::Relaxed), 1 + 15);
    Ok(())
}

#[test]
fn parallel_ordered_yields_by_row() -> VortexResult<()> {
    let service = Arc::new(InstantService::default());
    let runtime = CurrentThreadRuntime::new();
    let pool = runtime.new_pool();
    pool.set_workers(4);
    let driver = ParallelDriver::new(runtime.handle())
        .with_workers(4)
        .with_ordered(true);
    let batches: Vec<ScopedBatch> =
        runtime.block_on(driver.execute(plan(&service, 50, 6)).try_collect())?;
    assert_eq!(shape(&batches), expected(50, 6));
    Ok(())
}

#[test]
fn parallel_maps_on_workers() -> VortexResult<()> {
    let service = Arc::new(InstantService::default());
    let runtime = CurrentThreadRuntime::new();
    let driver = ParallelDriver::new(runtime.handle()).with_workers(2);
    let lens: Vec<ScopedBatch<usize>> = runtime.block_on(
        driver
            .execute_with(plan(&service, 9, 3), Arc::new(|array| Ok(array.len())))
            .try_collect(),
    )?;
    assert_eq!(lens.iter().map(|b| b.batch).sum::<usize>(), 9);
    Ok(())
}

#[rstest]
#[case::sequential(false)]
#[case::parallel(true)]
fn read_errors_fail_the_stream(#[case] parallel: bool) {
    let service = Arc::new(InstantService::failing());
    let runtime = CurrentThreadRuntime::new();
    let root = plan(&service, 8, 2);
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

/// Panics when first computed.
struct PanicMorsel;

impl IoConsumer for PanicMorsel {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Morsel for PanicMorsel {
    fn state(&self) -> State {
        State::NeedsCompute
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        panic!("morsel panicked");
    }
}

#[rstest]
#[case::one_worker(1)]
#[case::many_workers(3)]
fn parallel_reraises_a_worker_panic(#[case] workers: usize) {
    let service = Arc::new(InstantService::default());
    let runtime = CurrentThreadRuntime::new();
    let pool = runtime.new_pool();
    pool.set_workers(4);
    let root = plan_with(&service, 40, 4, |scope| -> Box<dyn Morsel> {
        if scope.rows.start == 8 {
            Box::new(PanicMorsel)
        } else {
            Box::new(ReadThenBatch::new(scope))
        }
    });
    let stream = ParallelDriver::new(runtime.handle())
        .with_workers(workers)
        .with_roots_per_worker(1)
        .execute(root);
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        runtime.block_on(stream.try_collect::<Vec<ScopedBatch>>())
    }));
    assert!(result.is_err(), "the panic must end the stream");
}

#[test]
fn parallel_error_stops_queued_roots() {
    let service = Arc::new(InstantService::failing());
    let runtime = CurrentThreadRuntime::new();
    let stream = ParallelDriver::new(runtime.handle())
        .with_workers(2)
        .with_roots_per_worker(1)
        .execute(plan(&service, 400, 4));
    let result: VortexResult<Vec<ScopedBatch>> = runtime.block_on(stream.try_collect());
    assert!(result.is_err());
    // Each worker reads at most the chunk it held when the first read failed.
    assert!(service.fetches.load(Ordering::Relaxed) <= 2);
}

#[test]
fn parallel_starts_on_first_poll() -> VortexResult<()> {
    let service = Arc::new(InstantService::default());
    let runtime = CurrentThreadRuntime::new();
    let stream = ParallelDriver::new(runtime.handle()).execute(plan(&service, 8, 4));
    assert_eq!(service.sessions.load(Ordering::Relaxed), 0);
    let batches: Vec<ScopedBatch> = runtime.block_on(stream.try_collect())?;
    assert_eq!(shape(&batches), expected(8, 4));
    Ok(())
}

#[test]
fn parallel_stream_dropped_mid_plan() -> VortexResult<()> {
    let service = Arc::new(InstantService::default());
    let runtime = CurrentThreadRuntime::new();
    let mut stream = ParallelDriver::new(runtime.handle())
        .with_workers(2)
        .execute(plan(&service, 400, 4));
    let first = runtime.block_on(stream.next());
    assert!(first.transpose()?.is_some());
    drop(stream);
    Ok(())
}
