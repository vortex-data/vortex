// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use parking_lot::Condvar;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_expr::checked_add;
use vortex_expr::gt;
use vortex_expr::lit;
use vortex_expr::root;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::ChunkBudget;
use super::ScanDriver;
use crate::execution::cache::ScanCache;
use crate::execution::chunk::MAX_ROWS_PER_CHUNK;
use crate::execution::pipeline::synthetic::RowSource;
use crate::plan::LazyPlanRef;
use crate::prepare::ScanBuilder;
use crate::source::ScanInput;
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

struct GateState {
    active: usize,
    peak: usize,
    entered: usize,
    released: usize,
    release_all: bool,
}

/// Deterministic latch for chunk worker tasks: each worker signals entry, tracks active/peak
/// concurrency, and blocks until the test explicitly releases its ticket.
struct ConcurrencyGate {
    state: Mutex<GateState>,
    condvar: Condvar,
    entered_tx: mpsc::UnboundedSender<()>,
}

impl ConcurrencyGate {
    fn new() -> (Arc<Self>, mpsc::UnboundedReceiver<()>) {
        let (entered_tx, entered_rx) = mpsc::unbounded();
        let gate = Arc::new(Self {
            state: Mutex::new(GateState {
                active: 0,
                peak: 0,
                entered: 0,
                released: 0,
                release_all: false,
            }),
            condvar: Condvar::new(),
            entered_tx,
        });
        (gate, entered_rx)
    }

    fn wait_turn(&self) {
        let mut state = self.state.lock();
        state.active += 1;
        state.peak = state.peak.max(state.active);
        let ticket = state.entered;
        state.entered += 1;
        let _ = self.entered_tx.unbounded_send(());
        while !state.release_all && ticket >= state.released {
            self.condvar.wait(&mut state);
        }
        state.active -= 1;
    }

    fn release(&self, count: usize) {
        let mut state = self.state.lock();
        state.released += count;
        self.condvar.notify_all();
    }

    fn release_all(&self) {
        let mut state = self.state.lock();
        state.release_all = true;
        self.condvar.notify_all();
    }

    fn active(&self) -> usize {
        self.state.lock().active
    }

    fn peak(&self) -> usize {
        self.state.lock().peak
    }
}

/// Unblocks any waiting workers if a test exits early on failure.
struct ReleaseOnDrop(Arc<ConcurrencyGate>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release_all();
    }
}

async fn wait_entries(
    entered_rx: &mut mpsc::UnboundedReceiver<()>,
    count: usize,
) -> VortexResult<()> {
    tokio::time::timeout(TEST_TIMEOUT, async {
        for _ in 0..count {
            assert!(
                entered_rx.next().await.is_some(),
                "gate channel closed before expected entries arrived"
            );
        }
    })
    .await
    .map_err(|_| vortex_err!("timed out waiting for {count} chunks to enter"))
}

async fn assert_no_extra_entry(entered_rx: &mut mpsc::UnboundedReceiver<()>) {
    for _ in 0..16 {
        tokio::task::yield_now().await;
    }
    assert!(
        entered_rx.try_recv().is_err(),
        "an extra chunk entered beyond the concurrency bound"
    );
}

fn first_u64(batch: &ArrayRef) -> VortexResult<u64> {
    let mut ctx = VortexSession::empty().create_execution_ctx();
    let prim = batch.clone().execute::<PrimitiveArray>(&mut ctx)?;
    Ok(prim.as_slice::<u64>()[0])
}

/// The concurrency bounds the chunks in flight in total, not per worker thread.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrency_bounds_chunks_in_flight_in_total(
    #[values(1, 2)] concurrency: usize,
) -> VortexResult<()> {
    let (gate, mut entered_rx) = ConcurrencyGate::new();
    let _guard = ReleaseOnDrop(Arc::clone(&gate));
    let map_fn = {
        let gate = Arc::clone(&gate);
        move |batch: ArrayRef| {
            gate.wait_turn();
            Ok(batch)
        }
    };

    let chunks = 4;
    let builder = chunked_builder(chunks);
    let driver = ScanDriver::new(builder.session().handle(), concurrency, true);
    let prepared = builder.prepare()?;
    let scan = tokio::spawn(async move {
        driver
            .run_prepared(prepared, map_fn)
            .try_collect::<Vec<_>>()
            .await
    });

    for _ in 0..(chunks / concurrency) {
        wait_entries(&mut entered_rx, concurrency).await?;
        assert_no_extra_entry(&mut entered_rx).await;
        assert_eq!(gate.active(), concurrency);
        assert_eq!(gate.peak(), concurrency);
        gate.release(concurrency);
    }

    let batches = tokio::time::timeout(TEST_TIMEOUT, scan)
        .await
        .map_err(|_| vortex_err!("timed out waiting for scan completion"))?
        .map_err(|err| vortex_err!("scan task panicked: {err}"))??;

    assert_eq!(batches.len(), chunks);
    assert_eq!(gate.peak(), concurrency);
    assert_eq!(gate.active(), 0);
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
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn shared_budget_bounds_chunks_in_flight_across_drivers() -> VortexResult<()> {
    let (gate, mut entered_rx) = ConcurrencyGate::new();
    let _guard = ReleaseOnDrop(Arc::clone(&gate));
    let map_fn = {
        let gate = Arc::clone(&gate);
        move |batch: ArrayRef| {
            gate.wait_turn();
            Ok(batch)
        }
    };

    let (budget, drivers, chunks) = (ChunkBudget::new(1), 2, 4);
    let limit = 1 + drivers;
    let scans = tokio::spawn(async move {
        let futures = (0..drivers).map(|_| {
            let builder = chunked_builder(chunks);
            let driver = ScanDriver::with_budget(builder.session().handle(), budget.clone(), true);
            let map_fn = map_fn.clone();
            async move {
                driver
                    .run_prepared(builder.prepare()?, map_fn)
                    .try_collect::<Vec<_>>()
                    .await
            }
        });
        futures::future::try_join_all(futures).await
    });

    // Two full waves of 3 saturate both drivers' own slots plus the shared slot; the remaining
    // 2 chunks finish once released.
    for _ in 0..2 {
        wait_entries(&mut entered_rx, limit).await?;
        assert_no_extra_entry(&mut entered_rx).await;
        assert_eq!(gate.active(), limit);
        assert_eq!(gate.peak(), limit);
        gate.release(limit);
    }
    gate.release_all();

    let batches = tokio::time::timeout(TEST_TIMEOUT, scans)
        .await
        .map_err(|_| vortex_err!("timed out waiting for shared-budget scans"))?
        .map_err(|err| vortex_err!("scans task panicked: {err}"))??;

    assert!(batches.iter().all(|batches| batches.len() == chunks));
    assert_eq!(gate.peak(), limit);
    assert_eq!(gate.active(), 0);
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

/// Ordered scans still yield chunk 0 first when chunk 0 is held until chunk 1 has mapped;
/// unordered scans yield chunk 1 to the caller while chunk 0 is still blocked.
#[rstest]
#[case::ordered(true)]
#[case::unordered(false)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chunk_ordering_with_out_of_order_completion(#[case] ordered: bool) -> VortexResult<()> {
    struct BlockZero {
        released: Mutex<bool>,
        condvar: Condvar,
    }

    impl BlockZero {
        fn wait(&self) {
            let mut released = self.released.lock();
            while !*released {
                self.condvar.wait(&mut released);
            }
        }

        fn release(&self) {
            *self.released.lock() = true;
            self.condvar.notify_all();
        }
    }

    struct ReleaseZeroOnDrop(Arc<BlockZero>);
    impl Drop for ReleaseZeroOnDrop {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    let block_zero = Arc::new(BlockZero {
        released: Mutex::new(false),
        condvar: Condvar::new(),
    });
    let _guard = ReleaseZeroOnDrop(Arc::clone(&block_zero));

    let map_fn = {
        let block_zero = Arc::clone(&block_zero);
        move |batch: ArrayRef| {
            let start = first_u64(&batch)?;
            if start == 0 {
                block_zero.wait();
            } else if ordered {
                // Chunk 1 has finished mapping; unblock chunk 0 so the ordered scan can collect both.
                block_zero.release();
            }
            Ok(batch)
        }
    };

    let builder = chunked_builder(2);
    let mut stream = ScanDriver::new(builder.session().handle(), 2, ordered)
        .run_prepared(builder.prepare()?, map_fn);

    let first = tokio::time::timeout(TEST_TIMEOUT, stream.next())
        .await
        .map_err(|_| vortex_err!("timed out waiting for first batch"))?
        .transpose()?
        .ok_or_else(|| vortex_err!("expected first batch"))?;
    if !ordered {
        // Unordered mode yielded chunk 1 while chunk 0 is still held; release chunk 0 now.
        assert_eq!(first_u64(&first)?, MAX_ROWS_PER_CHUNK as u64);
        block_zero.release();
    } else {
        assert_eq!(first_u64(&first)?, 0);
    }

    let second = tokio::time::timeout(TEST_TIMEOUT, stream.next())
        .await
        .map_err(|_| vortex_err!("timed out waiting for second batch"))?
        .transpose()?
        .ok_or_else(|| vortex_err!("expected second batch"))?;
    assert_eq!(
        first_u64(&second)?,
        if ordered {
            MAX_ROWS_PER_CHUNK as u64
        } else {
            0
        }
    );
    assert!(stream.next().await.is_none());
    Ok(())
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_fn_error_propagates_through_driver(
    #[values(0, 2)] concurrency: usize,
    #[values(true, false)] ordered: bool,
) -> VortexResult<()> {
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
        "unexpected error: {err}"
    );
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
