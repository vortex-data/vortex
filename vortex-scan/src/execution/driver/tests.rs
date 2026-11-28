// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

use futures::StreamExt;
use futures::TryStreamExt;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::assert_arrays_eq;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
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
use crate::source::SegmentFuture;
use crate::source::SegmentId;
use crate::source::SegmentSource;

struct NoSegments;

impl SegmentSource for NoSegments {
    fn request(&self, _id: SegmentId) -> SegmentFuture {
        Box::pin(async { vortex_bail!("an in-memory source must not request bytes") })
    }
}

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

/// The concurrency bounds the chunks in flight in total, not per worker thread.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrency_bounds_chunks_in_flight_in_total(
    #[values(1, 2)] concurrency: usize,
) -> VortexResult<()> {
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let map_fn = {
        let (active, peak) = (Arc::clone(&active), Arc::clone(&peak));
        move |batch: ArrayRef| {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(20));
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(batch)
        }
    };

    let chunks = 4;
    let plan = RowSource::plan(
        (chunks * MAX_ROWS_PER_CHUNK) as u64,
        Vec::new(),
        None,
        false,
    );
    let builder = ScanBuilder::new(
        VortexSession::empty().with_tokio(),
        ScanInput {
            dtype: plan.dtype().clone(),
            root: LazyPlanRef::ready(plan),
            locations: Arc::new([]),
            segments: Arc::new(NoSegments),
            cache: None,
        },
    );
    let driver = ScanDriver::new(builder.session().handle(), concurrency, true);
    let batches: Vec<ArrayRef> = driver
        .run_prepared(builder.prepare()?, map_fn)
        .try_collect()
        .await?;

    assert_eq!(batches.len(), chunks);
    assert!(peak.load(Ordering::SeqCst) <= concurrency);
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
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let map_fn = {
        let (active, peak) = (Arc::clone(&active), Arc::clone(&peak));
        move |batch: ArrayRef| {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(20));
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(batch)
        }
    };

    let (budget, drivers, chunks) = (ChunkBudget::new(1), 2, 4);
    let scans = (0..drivers).map(|_| {
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
    let batches = futures::future::try_join_all(scans).await?;

    assert!(batches.iter().all(|batches| batches.len() == chunks));
    assert!(peak.load(Ordering::SeqCst) <= 1 + drivers);
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
        Duration::from_secs(10),
        driver
            .run_prepared(builder.prepare()?, Ok)
            .try_collect::<Vec<_>>(),
    )
    .await
    .map_err(|_| vortex_error::vortex_err!("the scan waited on the stalled one"))??;

    assert_eq!(batches.len(), 4);
    drop(stalled);
    Ok(())
}
