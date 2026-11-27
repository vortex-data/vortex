// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::oneshot;
use futures::executor::LocalPool;
use futures::future::BoxFuture;
use futures::stream;
use futures::task::LocalSpawnExt;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_expr::checked_add;
use vortex_expr::gt;
use vortex_expr::lit;
use vortex_expr::root;
use vortex_io::runtime::AbortHandle;
use vortex_io::runtime::AbortHandleRef;
use vortex_io::runtime::Executor;
use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::ChunkBudget;
use super::ScanDriver;
use super::order_batches;
use crate::execution::SegmentLocation;
use crate::execution::cache::ScanCache;
use crate::execution::chunk::MAX_ROWS_PER_CHUNK;
use crate::execution::pipeline::synthetic::RowSource;
use crate::plan::ConcatPlan;
use crate::plan::LazyPlanRef;
use crate::prepare::ScanBuilder;
use crate::source::ScanInput;
use crate::source::SegmentFuture;
use crate::source::SegmentId;
use crate::source::SegmentSource;
use crate::test::NoSegments;

fn builder() -> ScanBuilder {
    let plan = RowSource::plan(20, vec![4, 7, 9], None, false);
    ScanBuilder::new(
        VortexSession::empty().with_tokio(),
        ScanInput {
            dtype: plan.dtype().clone(),
            root: LazyPlanRef::ready(plan),
            locations: Arc::new([]),
            segments: Arc::new(NoSegments),
            cache: Some(ScanCache::default()),
        },
    )
}

fn driver() -> ScanDriver {
    let session = VortexSession::empty().with_tokio();
    ScanDriver::new(session.handle(), 2, true)
}

#[tokio::test]
async fn in_memory_source_uses_preparation_filter_projection_and_driver() -> VortexResult<()> {
    let builder = builder();
    let dtype = builder.dtype().clone();
    let builder = builder
        .with_some_filter(Some(gt(root(), lit(10_u64)).bind(&dtype)?))
        .with_projection(checked_add(root(), lit(100_u64)).bind(&dtype)?);
    let batches: Vec<ArrayRef> = driver().run(builder).try_collect().await?;
    let actual = ChunkedArray::try_new(batches, dtype)?.into_array();
    assert_arrays_eq!(
        actual,
        Buffer::from_iter(111_u64..120).into_array(),
        &mut VortexSession::empty().create_execution_ctx()
    );
    Ok(())
}

/// Per-batch output mapping must run on the driver's tasks: on the thread
/// polling the stream it is serialised, which is what slowed the decompress benchmark down.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_batches_are_mapped_off_the_polling_thread(
    #[values(0, 2)] concurrency: usize,
) -> VortexResult<()> {
    let polling = thread::current().id();
    let mapped_on = Arc::new(Mutex::new(Vec::new()));
    let map_fn = {
        let mapped_on = Arc::clone(&mapped_on);
        move |batch: ArrayRef| {
            mapped_on.lock().push(thread::current().id());
            Ok(batch)
        }
    };

    let builder = builder();
    let driver = ScanDriver::new(builder.session().handle(), concurrency, true);
    let batches: Vec<ArrayRef> = driver
        .run_prepared(builder.prepare()?, map_fn)
        .try_collect()
        .await?;

    let mapped_on = mapped_on.lock();
    assert!(!batches.is_empty());
    assert_eq!(mapped_on.len(), batches.len());
    assert!(mapped_on.iter().all(|thread| *thread != polling));
    Ok(())
}

const TEST_TIMEOUT: Duration = Duration::from_secs(10);

struct NoopAbort;

impl AbortHandle for NoopAbort {
    fn abort(self: Box<Self>) {}
}

/// Single-threaded executor driven explicitly via [`LocalPool`], so tests can advance all ready
/// tasks to a quiescent point before asserting how many chunks were admitted.
#[derive(Default)]
struct ControlledRuntime {
    spawned: Mutex<VecDeque<BoxFuture<'static, ()>>>,
}

impl ControlledRuntime {
    fn handle(self: &Arc<Self>) -> Handle {
        let executor: Arc<dyn Executor> = Arc::<Self>::clone(self);
        Handle::new(Arc::downgrade(&executor))
    }

    fn run_until_stalled(&self, pool: &mut LocalPool) {
        loop {
            while let Some(fut) = self.spawned.lock().pop_front() {
                pool.spawner()
                    .spawn_local(fut)
                    .expect("local pool accepts spawned task");
            }
            pool.run_until_stalled();
            if self.spawned.lock().is_empty() {
                break;
            }
        }
    }
}

impl Executor for ControlledRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        self.spawned.lock().push_back(fut);
        Box::new(NoopAbort)
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.spawn(async move { task() }.boxed())
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.spawn(async move { task() }.boxed())
    }
}

#[derive(Default)]
struct GatedSegmentsState {
    pending: VecDeque<(SegmentId, oneshot::Sender<()>)>,
    peak: usize,
}

/// Segment source whose reads stay pending until the test explicitly completes them.
#[derive(Default)]
struct GatedSegments {
    state: Mutex<GatedSegmentsState>,
}

impl GatedSegments {
    fn pending_count(&self) -> usize {
        self.state.lock().pending.len()
    }

    fn peak(&self) -> usize {
        self.state.lock().peak
    }

    fn complete(&self, count: usize) {
        let senders: Vec<_> = {
            let mut state = self.state.lock();
            assert!(
                state.pending.len() >= count,
                "requested to complete {count} reads, but only {} pending",
                state.pending.len()
            );
            (0..count)
                .map(|_| state.pending.pop_front().expect("checked length").1)
                .collect()
        };
        for tx in senders {
            let _ = tx.send(());
        }
    }

    fn complete_segment(&self, target: SegmentId) {
        let tx = {
            let mut state = self.state.lock();
            let pos = state
                .pending
                .iter()
                .position(|(id, _)| *id == target)
                .unwrap_or_else(|| vortex_panic!("segment {target} is not pending"));
            state.pending.remove(pos).expect("valid index").1
        };
        let _ = tx.send(());
    }
}

impl SegmentSource for GatedSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let (tx, rx) = oneshot::channel();
        {
            let mut state = self.state.lock();
            state.pending.push_back((id, tx));
            let in_flight = state.pending.len();
            state.peak = state.peak.max(in_flight);
        }
        async move {
            rx.await
                .map_err(|_| vortex_err!("gated segment {id} sender dropped"))?;
            Ok(BufferHandle::new_host(ByteBuffer::empty()))
        }
        .boxed()
    }
}

fn gated_chunked_builder(
    handle: Handle,
    segments: Arc<GatedSegments>,
    chunk_lens: &[usize],
) -> VortexResult<ScanBuilder> {
    let children: Vec<_> = chunk_lens
        .iter()
        .enumerate()
        .map(|(i, &len)| {
            RowSource::plan(
                len as u64,
                Vec::new(),
                Some(SegmentId::from(u32::try_from(i).expect("chunk fits u32"))),
                false,
            )
        })
        .collect();
    let dtype = children[0].dtype().clone();
    let plan = ConcatPlan::try_new(dtype.clone(), children)?.into_plan();
    let locations: Arc<[SegmentLocation]> = (0..chunk_lens.len())
        .map(|i| SegmentLocation {
            offset: (i as u64) * 16,
            length: 16,
            alignment: Alignment::new(8),
        })
        .collect();
    Ok(ScanBuilder::new(
        VortexSession::empty().with_handle(handle),
        ScanInput {
            dtype,
            root: LazyPlanRef::ready(plan),
            locations,
            segments,
            cache: None,
        },
    ))
}

/// The concurrency bounds the chunks in flight in total, not per worker thread.
#[rstest]
fn concurrency_bounds_chunks_in_flight_in_total(
    #[values(1, 2)] concurrency: usize,
) -> VortexResult<()> {
    let runtime = Arc::new(ControlledRuntime::default());
    let mut pool = LocalPool::new();
    let segments = Arc::new(GatedSegments::default());

    let chunks = 4;
    let builder = gated_chunked_builder(
        runtime.handle(),
        Arc::clone(&segments),
        &vec![MAX_ROWS_PER_CHUNK; chunks],
    )?;
    let driver = ScanDriver::new(runtime.handle(), concurrency, true);
    let prepared = builder.prepare()?;
    let scan = pool
        .spawner()
        .spawn_local_with_handle(driver.run_prepared(prepared, Ok).try_collect::<Vec<_>>())
        .map_err(|err| vortex_err!("spawn scan: {err}"))?;

    for _ in 0..(chunks / concurrency) {
        runtime.run_until_stalled(&mut pool);
        assert_eq!(segments.pending_count(), concurrency);
        assert_eq!(segments.peak(), concurrency);
        segments.complete(concurrency);
    }

    runtime.run_until_stalled(&mut pool);
    let batches = scan
        .now_or_never()
        .ok_or_else(|| vortex_err!("scan did not finish after all reads completed"))??;
    assert_eq!(batches.len(), chunks);
    assert_eq!(segments.peak(), concurrency);
    assert_eq!(segments.pending_count(), 0);
    Ok(())
}

fn chunked_builder(chunks: usize) -> ScanBuilder {
    let plan = RowSource::plan(
        (chunks * MAX_ROWS_PER_CHUNK) as u64,
        Vec::new(),
        None,
        false,
    );
    ScanBuilder::new(
        VortexSession::empty().with_tokio(),
        ScanInput {
            dtype: plan.dtype().clone(),
            root: LazyPlanRef::ready(plan),
            locations: Arc::new([]),
            segments: Arc::new(NoSegments),
            cache: None,
        },
    )
}

/// Drivers sharing a budget run at most its chunks plus one of their own each, all together.
#[test]
fn shared_budget_bounds_chunks_in_flight_across_drivers() -> VortexResult<()> {
    let runtime = Arc::new(ControlledRuntime::default());
    let mut pool = LocalPool::new();
    let segments = Arc::new(GatedSegments::default());

    let (budget, drivers, chunks) = (ChunkBudget::new(1), 2, 4);
    let limit = 1 + drivers;
    let mut prepared_scans = Vec::with_capacity(drivers);
    for _ in 0..drivers {
        let builder = gated_chunked_builder(
            runtime.handle(),
            Arc::clone(&segments),
            &vec![MAX_ROWS_PER_CHUNK; chunks],
        )?;
        let driver = ScanDriver::with_budget(runtime.handle(), budget.clone(), true);
        prepared_scans.push((driver, builder.prepare()?));
    }

    let scans = pool
        .spawner()
        .spawn_local_with_handle(futures::future::try_join_all(
            prepared_scans.into_iter().map(|(driver, prepared)| {
                driver.run_prepared(prepared, Ok).try_collect::<Vec<_>>()
            }),
        ))
        .map_err(|err| vortex_err!("spawn scans: {err}"))?;

    // Two full waves of 3 saturate both drivers' own slots plus the shared slot, followed by
    // the remaining 2 chunks.
    for wave in [limit, limit, drivers * chunks - 2 * limit] {
        runtime.run_until_stalled(&mut pool);
        assert_eq!(segments.pending_count(), wave);
        assert_eq!(segments.peak(), limit);
        segments.complete(wave);
    }

    runtime.run_until_stalled(&mut pool);
    let batches = scans.now_or_never().ok_or_else(|| {
        vortex_err!("shared-budget scans did not finish after all reads completed")
    })??;
    assert!(batches.iter().all(|batches| batches.len() == chunks));
    assert_eq!(segments.peak(), limit);
    assert_eq!(segments.pending_count(), 0);
    Ok(())
}

/// A scan whose output nobody reads keeps its share of the budget, but the others still finish
/// on slots of their own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shared_budget_lets_scans_finish_while_another_stalls() -> VortexResult<()> {
    let budget = ChunkBudget::new(1);
    let stalled_builder = chunked_builder(8);
    let mut stalled =
        ScanDriver::with_budget(stalled_builder.session().handle(), budget.clone(), false)
            .run_prepared(stalled_builder.prepare()?, Ok);
    stalled.next().await.transpose()?;

    let builder = chunked_builder(4);
    let driver = ScanDriver::with_budget(builder.session().handle(), budget, false);
    let batches = tokio::time::timeout(
        TEST_TIMEOUT,
        driver
            .run_prepared(builder.prepare()?, Ok)
            .try_collect::<Vec<_>>(),
    )
    .await
    .map_err(|_| vortex_err!("the scan waited on the stalled one"))??;

    assert_eq!(batches.len(), 4);
    drop(stalled);
    Ok(())
}

/// Ordered collection sorts batches by starting row even when later chunks finish first,
/// both directly via [`order_batches`] and end-to-end through [`ScanDriver::run_prepared`].
#[test]
fn ordered_batches_are_sorted_by_start_row() -> VortexResult<()> {
    let mut pool = LocalPool::new();
    let out_of_order = stream::iter([
        Ok((MAX_ROWS_PER_CHUNK as u64, "second")),
        Ok((0_u64, "first")),
    ]);
    let ordered: Vec<_> = pool.run_until(order_batches(out_of_order).try_collect())?;
    assert_eq!(ordered, vec!["first", "second"]);

    let runtime = Arc::new(ControlledRuntime::default());
    let segments = Arc::new(GatedSegments::default());
    let second_chunk_len = MAX_ROWS_PER_CHUNK / 2;
    let builder = gated_chunked_builder(
        runtime.handle(),
        Arc::clone(&segments),
        &[MAX_ROWS_PER_CHUNK, second_chunk_len],
    )?;
    let mapped_lens = Arc::new(Mutex::new(Vec::new()));
    let map_fn = {
        let mapped_lens = Arc::clone(&mapped_lens);
        move |batch: ArrayRef| {
            mapped_lens.lock().push(batch.len());
            Ok(batch)
        }
    };
    let scan = pool
        .spawner()
        .spawn_local_with_handle(
            ScanDriver::new(runtime.handle(), 2, true)
                .run_prepared(builder.prepare()?, map_fn)
                .try_collect::<Vec<_>>(),
        )
        .map_err(|err| vortex_err!("spawn ordered scan: {err}"))?;

    runtime.run_until_stalled(&mut pool);
    assert_eq!(segments.pending_count(), 2);

    // Complete chunk 1's I/O and drain all ready work so chunk 1 finishes before chunk 0 unblocks.
    segments.complete_segment(SegmentId::from(1));
    runtime.run_until_stalled(&mut pool);
    assert_eq!(segments.pending_count(), 1);
    assert_eq!(*mapped_lens.lock(), vec![second_chunk_len]);

    segments.complete_segment(SegmentId::from(0));
    runtime.run_until_stalled(&mut pool);
    assert_eq!(
        *mapped_lens.lock(),
        vec![second_chunk_len, MAX_ROWS_PER_CHUNK]
    );
    let batches = scan
        .now_or_never()
        .ok_or_else(|| vortex_err!("ordered scan did not finish after all reads completed"))??;
    let lens: Vec<usize> = batches.iter().map(ArrayRef::len).collect();
    assert_eq!(lens, vec![MAX_ROWS_PER_CHUNK, second_chunk_len]);
    Ok(())
}

/// Unordered scans yield chunk 1 to the caller while chunk 0 is still blocked on I/O.
#[test]
fn unordered_scan_yields_ready_chunk_while_earlier_chunk_blocks_on_io() -> VortexResult<()> {
    let runtime = Arc::new(ControlledRuntime::default());
    let mut pool = LocalPool::new();
    let segments = Arc::new(GatedSegments::default());

    let second_chunk_len = MAX_ROWS_PER_CHUNK / 2;
    let builder = gated_chunked_builder(
        runtime.handle(),
        Arc::clone(&segments),
        &[MAX_ROWS_PER_CHUNK, second_chunk_len],
    )?;
    let mut stream =
        ScanDriver::new(runtime.handle(), 2, false).run_prepared(builder.prepare()?, Ok);

    let first_fut = pool
        .spawner()
        .spawn_local_with_handle(async move {
            let first = stream
                .next()
                .await
                .transpose()?
                .ok_or_else(|| vortex_err!("expected first batch"))?;
            Ok::<_, vortex_error::VortexError>((first, stream))
        })
        .map_err(|err| vortex_err!("spawn first: {err}"))?;

    runtime.run_until_stalled(&mut pool);
    assert_eq!(segments.pending_count(), 2);

    // Complete chunk 1's I/O while chunk 0 stays pending; unordered mode yields chunk 1 now.
    segments.complete_segment(SegmentId::from(1));
    runtime.run_until_stalled(&mut pool);
    let (first, mut stream) = first_fut.now_or_never().ok_or_else(|| {
        vortex_err!("unordered scan did not yield chunk 1 while chunk 0 blocked")
    })??;
    assert_eq!(first.len(), second_chunk_len);
    assert_eq!(segments.pending_count(), 1);

    // Completing chunk 0's I/O yields chunk 0 and ends the stream.
    let rest_fut = pool
        .spawner()
        .spawn_local_with_handle(async move {
            let second = stream
                .next()
                .await
                .transpose()?
                .ok_or_else(|| vortex_err!("expected second batch"))?;
            let end = stream.next().await;
            Ok::<_, vortex_error::VortexError>((second, end))
        })
        .map_err(|err| vortex_err!("spawn rest: {err}"))?;

    segments.complete_segment(SegmentId::from(0));
    runtime.run_until_stalled(&mut pool);
    let (second, end) = rest_fut.now_or_never().ok_or_else(|| {
        vortex_err!("unordered scan did not yield chunk 0 after its read completed")
    })??;
    assert_eq!(second.len(), MAX_ROWS_PER_CHUNK);
    assert!(end.is_none());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_fn_error_propagates_through_driver() -> VortexResult<()> {
    for (concurrency, ordered) in [(0, true), (0, false), (2, true), (2, false)] {
        let builder = chunked_builder(2);
        let err = ScanDriver::new(builder.session().handle(), concurrency, ordered)
            .run_prepared(builder.prepare()?, |_| -> VortexResult<ArrayRef> {
                vortex_bail!("mapping failed on chunk")
            })
            .try_collect::<Vec<_>>()
            .await
            .err()
            .ok_or_else(|| vortex_err!("map_fn error unexpectedly succeeded"))?;
        assert!(
            err.to_string().contains("mapping failed on chunk"),
            "unexpected error for concurrency={concurrency}, ordered={ordered}: {err}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn run_reports_preparation_error_as_first_stream_item() -> VortexResult<()> {
    let session = VortexSession::empty().with_tokio();
    let builder = ScanBuilder::new(
        session.clone(),
        ScanInput {
            dtype: vortex_array::dtype::DType::Primitive(
                vortex_array::dtype::PType::U64,
                vortex_array::dtype::Nullability::NonNullable,
            ),
            root: LazyPlanRef::lazy(|| Err(vortex_err!("prepare boom"))),
            locations: Arc::new([]),
            segments: Arc::new(NoSegments),
            cache: None,
        },
    );
    let mut stream = ScanDriver::new(session.handle(), 2, true).run(builder);
    let err = stream
        .next()
        .await
        .ok_or_else(|| vortex_err!("expected error item"))?
        .err()
        .ok_or_else(|| vortex_err!("expected Err"))?;
    assert!(err.to_string().contains("prepare boom"), "{err}");
    assert!(stream.next().await.is_none());
    Ok(())
}
