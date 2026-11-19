// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::thread;

use futures::TryStreamExt;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::assert_arrays_eq;
use vortex_array::expr::checked_add;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::ScanDriver;
use crate::execution::cache::ScanCache;
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
