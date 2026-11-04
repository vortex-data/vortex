// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::panic, clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use futures::task::noop_waker_ref;
use parking_lot::Mutex;
use rstest::rstest;
use tokio::sync::Notify;
use tokio::sync::Semaphore;
use tokio::sync::mpsc::unbounded_channel;
use tokio::time::timeout;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::ChunkedArray;
use vortex::array::buffer::BufferHandle;
use vortex::arrow::ArrowSessionExt;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::VortexReadAt;
use vortex::io::read_at2::FutureReadAt2;
use vortex::io::read_at2::ReadAt2IoService;
use vortex::io::request::IoConsumer;
use vortex::io::request::IoRequestId;
use vortex::io::request::IoResult;
use vortex::io::request::IoSlot;
use vortex::io::request::IoTarget;
use vortex::io::runtime::AbortHandle;
use vortex::io::runtime::AbortHandleRef;
use vortex::io::runtime::Executor;
use vortex::io::runtime::Handle;
use vortex::io::runtime::tokio::TokioRuntime;
use vortex::scan::planning::morsel::Morsel;
use vortex::scan::planning::morsel::MorselOutput;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::PlannerOutput;
use vortex::scan::planning::planner::State;
use vortex::scan::planning::planner::WorkScope;
use vortex::session::VortexSession;

use super::MapBatch;
use super::ScanBatch;
use super::chunk_stream;
use super::map_chunks;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

type CpuTask = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct QueuedExecutor {
    tasks: Mutex<Vec<Option<CpuTask>>>,
}

impl QueuedExecutor {
    fn run(&self, index: usize) {
        let task = self.tasks.lock()[index].take().unwrap();
        task();
    }

    fn count(&self) -> usize {
        self.tasks.lock().len()
    }

    fn handle(self: &Arc<Self>) -> Handle {
        let executor: Arc<dyn Executor> = Arc::clone(self) as _;
        Handle::new(Arc::downgrade(&executor))
    }
}

struct NoopAbort;

impl AbortHandle for NoopAbort {
    fn abort(self: Box<Self>) {}
}

impl Executor for QueuedExecutor {
    fn spawn(&self, _future: BoxFuture<'static, ()>) -> AbortHandleRef {
        panic!("Output mapping should only spawn CPU work")
    }

    fn spawn_cpu(&self, task: CpuTask) -> AbortHandleRef {
        self.tasks.lock().push(Some(task));
        Box::new(NoopAbort)
    }

    fn spawn_blocking_io(&self, _task: CpuTask) -> AbortHandleRef {
        panic!("Output mapping should not spawn blocking I/O")
    }
}

fn batch(values: Vec<Option<&str>>) -> VortexResult<RecordBatch> {
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)])),
        vec![Arc::new(StringArray::from(values))],
    )?)
}

fn array(batch: &RecordBatch) -> VortexResult<ArrayRef> {
    SESSION
        .arrow()
        .from_arrow_record_batch(batch.clone(), &batch.schema())
}

fn export(array: ArrayRef) -> VortexResult<RecordBatch> {
    assert!(
        array.as_opt::<Chunked>().is_none(),
        "Stored chunks were merged"
    );
    let mut ctx = SESSION.create_execution_ctx();
    let target = Field::new(
        "",
        DataType::Struct(vec![Field::new("s", DataType::Utf8, true)].into()),
        false,
    );
    let arrow = SESSION
        .arrow()
        .execute_arrow(array, Some(&target), &mut ctx)?;
    Ok(RecordBatch::from(arrow.as_struct().clone()))
}

#[rstest]
fn chunk_mapping_preserves_order_and_bounds_admitted_work(
    #[values(false, true)] ordered: bool,
) -> VortexResult<()> {
    let expected = [
        batch(vec![Some("first long string"), None])?,
        batch(vec![Some("second long string")])?,
        batch(vec![None, Some("third long string")])?,
    ];
    let arrays = expected
        .iter()
        .map(array)
        .collect::<VortexResult<Vec<_>>>()?;
    let grouped = ChunkedArray::try_new(arrays[..2].to_vec(), arrays[0].dtype().clone())?;
    let input = stream::iter([
        Ok(ScanBatch::Array(grouped.into_array())),
        Ok(ScanBatch::Array(arrays[2].clone())),
    ]);
    let executor = Arc::new(QueuedExecutor::default());
    let mut output = map_chunks(
        input.boxed(),
        executor.handle(),
        Arc::new(export),
        ordered,
        2,
    );
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(output.as_mut().poll_next(&mut cx).is_pending());
    assert_eq!(executor.count(), 2);

    executor.run(1);
    let second = output.as_mut().poll_next(&mut cx);
    if ordered {
        assert!(second.is_pending());
    } else {
        assert_eq!(
            second.map(|value| value.unwrap().unwrap()),
            Poll::Ready(expected[1].clone())
        );
    }
    executor.run(0);
    assert_eq!(
        output
            .as_mut()
            .poll_next(&mut cx)
            .map(|value| value.unwrap().unwrap()),
        Poll::Ready(expected[0].clone()),
    );
    if ordered {
        assert_eq!(
            output
                .as_mut()
                .poll_next(&mut cx)
                .map(|value| value.unwrap().unwrap()),
            Poll::Ready(expected[1].clone()),
        );
    }
    assert!(output.as_mut().poll_next(&mut cx).is_pending());
    assert_eq!(executor.count(), 3);
    executor.run(2);
    assert_eq!(
        output
            .as_mut()
            .poll_next(&mut cx)
            .map(|value| value.unwrap().unwrap()),
        Poll::Ready(expected[2].clone()),
    );
    assert!(matches!(
        output.as_mut().poll_next(&mut cx),
        Poll::Ready(None)
    ));
    Ok(())
}

#[rstest]
fn dropping_output_cancels_queued_mapping(
    #[values(false, true)] ordered: bool,
) -> VortexResult<()> {
    let input = array(&batch(vec![Some("cancelled")])?)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let mapped = Arc::clone(&calls);
    let executor = Arc::new(QueuedExecutor::default());
    let mut output = map_chunks(
        stream::iter((0..4).map(move |_| Ok(ScanBatch::Array(input.clone())))).boxed(),
        executor.handle(),
        Arc::new(move |array| {
            mapped.fetch_add(1, Ordering::Relaxed);
            export(array)
        }),
        ordered,
        2,
    );
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(output.as_mut().poll_next(&mut cx).is_pending());
    assert_eq!(executor.count(), 2);
    drop(output);
    executor.run(0);
    executor.run(1);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(executor.count(), 2);
    Ok(())
}

#[rstest]
fn chunk_mapping_propagates_export_errors(
    #[values(false, true)] ordered: bool,
) -> VortexResult<()> {
    let input = array(&batch(vec![Some("error")])?)?;
    let executor = Arc::new(QueuedExecutor::default());
    let mut output = map_chunks(
        stream::iter([Ok(ScanBatch::Array(input))]).boxed(),
        executor.handle(),
        Arc::new(|_| Err(vortex_err!("mapping failed"))),
        ordered,
        1,
    );
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(output.as_mut().poll_next(&mut cx).is_pending());
    executor.run(0);
    let Poll::Ready(Some(Err(error))) = output.as_mut().poll_next(&mut cx) else {
        panic!("Mapping error was not delivered");
    };
    assert!(error.to_string().contains("mapping failed"));
    Ok(())
}

struct BatchPlan {
    batches: Option<Batches>,
    scope: WorkScope,
}

impl IoConsumer for BatchPlan {
    fn set_io_result(&mut self, _: IoRequestId, _: IoResult) {
        panic!("Batch plan does not request I/O");
    }
}

impl Planner for BatchPlan {
    fn state(&self) -> State {
        if self.batches.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        Ok(PlannerOutput::Morsel(
            self.scope.clone(),
            Box::new(self.batches.take().unwrap()),
        ))
    }
}

struct Batches(std::vec::IntoIter<ArrayRef>);

impl IoConsumer for Batches {
    fn set_io_result(&mut self, _: IoRequestId, _: IoResult) {
        panic!("Batch morsel does not request I/O");
    }
}

impl Morsel for Batches {
    fn state(&self) -> State {
        if self.0.len() == 0 {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        Ok(MorselOutput::Batch(self.0.next().unwrap()))
    }
}

fn admitted_batches(
    admission: Arc<Semaphore>,
    arrays: Vec<ArrayRef>,
    map: MapBatch,
) -> BoxStream<'static, VortexResult<(u64, ScanBatch)>> {
    let scope = WorkScope {
        file_ordinal: 0,
        rows: 0..arrays.iter().map(|array| array.len() as u64).sum(),
    };
    let io = ReadAt2IoService::new(Arc::new(|| {
        Ok(Arc::new(FutureReadAt2::new(Arc::new(ByteBuffer::empty()))))
    }));
    chunk_stream(
        &TokioRuntime::current(),
        Box::new(BatchPlan {
            batches: Some(Batches(arrays.into_iter())),
            scope: scope.clone(),
        }),
        scope,
        Arc::new(io),
        map,
        Some(admission),
        false,
    )
}

#[tokio::test]
async fn chunk_admission_releases_permits_before_output_backpressure() -> VortexResult<()> {
    let expected = batch(vec![Some("admitted")])?;
    let input = array(&expected)?;
    let admission = Arc::new(Semaphore::new(1));
    let gate = admission.acquire().await.unwrap();
    let (mapped, mut notifications) = unbounded_channel();
    let first = admitted_batches(
        Arc::clone(&admission),
        vec![input.clone(); 3],
        Arc::new(move |array| {
            mapped.send(()).unwrap();
            export(array).map(ScanBatch::Arrow)
        }),
    );
    tokio::task::yield_now().await;
    assert!(notifications.try_recv().is_err());
    drop(gate);
    for _ in 0..2 {
        timeout(Duration::from_secs(5), notifications.recv())
            .await
            .unwrap()
            .unwrap();
    }
    // The first producer has filled its channel without the consumer being polled.
    let second = admitted_batches(
        Arc::clone(&admission),
        vec![input],
        Arc::new(|array| export(array).map(ScanBatch::Arrow)),
    );
    let second = timeout(Duration::from_secs(5), second.try_collect::<Vec<_>>())
        .await
        .unwrap()?;
    assert_eq!(second.len(), 1);
    let first = timeout(Duration::from_secs(5), first.try_collect::<Vec<_>>())
        .await
        .unwrap()?;
    assert_eq!(first.len(), 3);
    for (_, batch) in first.into_iter().chain(second) {
        let ScanBatch::Arrow(batch) = batch else {
            panic!("Expected mapped output");
        };
        assert_eq!(batch, expected);
    }
    assert_eq!(admission.available_permits(), 1);
    Ok(())
}

#[tokio::test]
async fn dropping_chunk_waiting_for_admission_cancels_work() -> VortexResult<()> {
    let admission = Arc::new(Semaphore::new(1));
    let gate = admission.acquire().await.unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let mapped = Arc::clone(&calls);
    let output = admitted_batches(
        Arc::clone(&admission),
        vec![array(&batch(vec![Some("cancelled")])?)?],
        Arc::new(move |array| {
            mapped.fetch_add(1, Ordering::Relaxed);
            export(array).map(ScanBatch::Arrow)
        }),
    );
    tokio::task::yield_now().await;
    drop(output);
    tokio::task::yield_now().await;
    drop(gate);
    tokio::task::yield_now().await;
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(admission.available_permits(), 1);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn chunk_admission_propagates_errors_and_releases_permits(
    #[values(false, true)] closed: bool,
) -> VortexResult<()> {
    let admission = Arc::new(Semaphore::new(1));
    if closed {
        admission.close();
    }
    let output = admitted_batches(
        Arc::clone(&admission),
        vec![array(&batch(vec![Some("error")])?)?],
        Arc::new(|_| Err(vortex_err!("mapping failed"))),
    );
    let result = timeout(Duration::from_secs(5), output.try_collect::<Vec<_>>())
        .await
        .unwrap();
    let Err(error) = result else {
        panic!("Expected failure");
    };
    assert!(error.to_string().contains(if closed {
        "Scan chunk admission closed"
    } else {
        "mapping failed"
    }));
    assert_eq!(admission.available_permits(), 1);
    Ok(())
}

struct DelayedSize(Arc<Notify>);

impl VortexReadAt for DelayedSize {
    fn concurrency(&self) -> usize {
        1
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let ready = Arc::clone(&self.0);
        async move {
            ready.notified().await;
            Ok(0)
        }
        .boxed()
    }

    fn read_at(
        &self,
        _: u64,
        _: usize,
        _: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        panic!("Only size is requested");
    }
}

struct IoBatchPlan {
    started: bool,
    slot: IoSlot,
    inner: BatchPlan,
}

impl IoConsumer for IoBatchPlan {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.slot.deliver(request, result);
    }
}

impl Planner for IoBatchPlan {
    fn state(&self) -> State {
        if self.slot.is_waiting() {
            State::Waiting
        } else {
            self.inner.state()
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if !self.started {
            self.started = true;
            return Ok(PlannerOutput::NeedsIO(self.slot.issue(IoTarget::Size)));
        }
        assert!(matches!(self.slot.take(), Some(IoResult::Size(0))));
        self.inner.compute()
    }
}

#[rstest]
#[tokio::test]
async fn admission_after_io_leaves_ready_chunks_unrestricted(
    #[values(false, true)] waits: bool,
) -> VortexResult<()> {
    let admission = Arc::new(Semaphore::new(1));
    let gate = admission.acquire().await.unwrap();
    let ready = Arc::new(Notify::new());
    let reader: Arc<dyn VortexReadAt> = Arc::new(DelayedSize(Arc::clone(&ready)));
    let io = ReadAt2IoService::new(Arc::new(move || {
        Ok(Arc::new(FutureReadAt2::new(Arc::clone(&reader))))
    }));
    let expected = batch(vec![Some("ready")])?;
    let scope = WorkScope {
        file_ordinal: 0,
        rows: 0..1,
    };
    let plan = BatchPlan {
        batches: Some(Batches(vec![array(&expected)?].into_iter())),
        scope: scope.clone(),
    };
    let plan: Box<dyn Planner> = if waits {
        Box::new(IoBatchPlan {
            started: false,
            slot: IoSlot::default(),
            inner: plan,
        })
    } else {
        Box::new(plan)
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let mapped = Arc::clone(&calls);
    let output = chunk_stream(
        &TokioRuntime::current(),
        plan,
        scope,
        Arc::new(io),
        Arc::new(move |array| {
            mapped.fetch_add(1, Ordering::Relaxed);
            export(array).map(ScanBatch::Arrow)
        }),
        Some(Arc::clone(&admission)),
        true,
    );
    tokio::task::yield_now().await;
    if waits {
        ready.notify_one();
        tokio::task::yield_now().await;
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        drop(gate);
    } else {
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        // Retain the unavailable permit through collection: a ready chunk never needs it.
    }
    let batches = timeout(Duration::from_secs(5), output.try_collect::<Vec<_>>())
        .await
        .unwrap()?;
    assert_eq!(batches.len(), 1);
    let ScanBatch::Arrow(batch) = &batches[0].1 else {
        panic!("Expected mapped output");
    };
    assert_eq!(batch, &expected);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod native_dispatch {
    use std::thread;

    use futures::future;
    use tokio::sync::Barrier;
    use vortex::io::request::IoService;
    use vortex::io::uring::UringOptions;
    use vortex::io::uring::UringWorkerPool;

    use super::*;
    use crate::persistent::scan_v2::Yielded;
    use crate::persistent::scan_v2::worker_chunk_stream;
    use crate::persistent::scan_v2::worker_stream;

    struct SplitPlan(std::vec::IntoIter<(WorkScope, Box<dyn Planner>)>);

    impl IoConsumer for SplitPlan {
        fn set_io_result(&mut self, _: IoRequestId, _: IoResult) {
            panic!("Split plan does not request I/O");
        }
    }

    impl Planner for SplitPlan {
        fn state(&self) -> State {
            if self.0.len() == 0 {
                State::Done
            } else {
                State::NeedsCompute
            }
        }

        fn compute(&mut self) -> VortexResult<PlannerOutput> {
            let (scope, plan) = self.0.next().unwrap();
            Ok(PlannerOutput::Spawn(scope, plan))
        }
    }

    struct PairedSize(Arc<Barrier>);

    impl VortexReadAt for PairedSize {
        fn concurrency(&self) -> usize {
            1
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            let barrier = Arc::clone(&self.0);
            async move {
                let owner = thread::current().id();
                barrier.wait().await;
                assert_eq!(thread::current().id(), owner);
                Ok(0)
            }
            .boxed()
        }

        fn read_at(
            &self,
            _: u64,
            _: usize,
            _: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            panic!("Only size is requested");
        }
    }

    #[rstest]
    #[tokio::test]
    async fn spawned_chunks_make_progress_on_separate_native_workers(
        #[values(false, true)] fail_mapping: bool,
        #[values(false, true)] reserved: bool,
    ) -> VortexResult<()> {
        let pool = UringWorkerPool::with_inherited_affinity(
            2 + usize::from(reserved),
            UringOptions {
                max_control_jobs: 1,
                max_active_jobs: 1,
                control_workers: usize::from(reserved),
                ..Default::default()
            },
        )?;
        let reader: Arc<dyn VortexReadAt> = Arc::new(PairedSize(Arc::new(Barrier::new(2))));
        let io: Arc<dyn IoService> = Arc::new(ReadAt2IoService::new(Arc::new(move || {
            Ok(Arc::new(FutureReadAt2::new(Arc::clone(&reader))))
        })));
        let expected = batch(vec![Some("native chunk")])?;
        let input = array(&expected)?;
        let children = (0..6)
            .map(|i| {
                let scope = WorkScope {
                    file_ordinal: 0,
                    rows: i..i + 1,
                };
                let plan: Box<dyn Planner> = Box::new(IoBatchPlan {
                    started: false,
                    slot: IoSlot::default(),
                    inner: BatchPlan {
                        scope: scope.clone(),
                        batches: Some(Batches(vec![input.clone(); 3].into_iter())),
                    },
                });
                (scope, plan)
            })
            .collect::<Vec<_>>();
        let map: MapBatch = Arc::new(move |array| {
            if fail_mapping {
                Err(vortex_err!("native mapping failed"))
            } else {
                export(array).map(ScanBatch::Arrow)
            }
        });
        let root = worker_stream(
            pool.clone(),
            Box::new(SplitPlan(children.into_iter())),
            WorkScope {
                file_ordinal: 0,
                rows: 0..6,
            },
            Arc::clone(&io),
            Arc::clone(&map),
            true,
        );
        let output = root
            .map_ok(move |yielded| match yielded {
                Yielded::Spawn(scope, child) => worker_chunk_stream(
                    pool.clone(),
                    child,
                    scope,
                    Arc::clone(&io),
                    Arc::clone(&map),
                ),
                Yielded::Batch(start, batch) => {
                    stream::once(future::ready(Ok((start, batch)))).boxed()
                }
            })
            .try_flatten_unordered(4);
        let collected = timeout(Duration::from_secs(5), output.try_collect::<Vec<_>>())
            .await
            .expect("native child admission must progress while its planner is active");
        if fail_mapping {
            assert!(
                collected
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("native mapping failed")
            );
        } else {
            let collected = collected?;
            assert_eq!(collected.len(), 18);
            let mut starts = Vec::new();
            for (start, batch) in collected {
                starts.push(start);
                let ScanBatch::Arrow(batch) = batch else {
                    panic!("Expected Arrow output")
                };
                assert_eq!(batch, expected);
            }
            starts.sort_unstable();
            assert_eq!(starts, (0..6).flat_map(|i| [i; 3]).collect::<Vec<_>>());
        }
        Ok(())
    }
}
