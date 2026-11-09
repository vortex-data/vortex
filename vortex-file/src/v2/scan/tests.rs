// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use futures::TryStreamExt;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::session::RuntimeSession;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use super::ScanDriver;
use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    crate::register_default_encodings(&session);
    crate::enable_all_registered_array_encodings(&session);
    session
});

async fn file() -> VortexResult<VortexFile> {
    let chunks = 8;
    let numbers = ChunkedArray::from_iter(
        (0..chunks).map(|chunk| buffer![chunk, chunk + 1, chunk + 2, chunk + 3].into_array()),
    )
    .into_array();
    let strings = ChunkedArray::from_iter(
        (0..chunks).map(|_| VarBinArray::from(vec!["a", "bb", "ccc", "dddd"]).into_array()),
    )
    .into_array();
    let array = StructArray::from_fields(&[("numbers", numbers), ("strings", strings)])?;
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.into_array().to_array_stream())
        .await?;
    SESSION.open_options().open_buffer(buf)
}

fn driver() -> ScanDriver {
    ScanDriver::new(TokioRuntime::current(), 2, true)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_batches_match_arrays() -> VortexResult<()> {
    let file = file().await?;
    let scan = || ScanBuilder::new(SESSION.clone(), file.scan_file());
    let schema = Arc::new(SESSION.arrow().to_arrow_schema(scan().dtype())?);
    let field = Field::new_struct("", schema.fields().clone(), false);

    let arrays: Vec<ArrayRef> = driver().run(scan()).try_collect().await?;
    let mut ctx = SESSION.create_execution_ctx();
    let expected = arrays
        .into_iter()
        .map(|array| {
            let arrow = SESSION
                .arrow()
                .execute_arrow(array, Some(&field), &mut ctx)?;
            Ok(RecordBatch::from(arrow.as_struct().clone()))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let derived: Vec<RecordBatch> = driver()
        .run_record_batches(scan(), None)
        .try_collect()
        .await?;
    let given: Vec<RecordBatch> = driver()
        .run_record_batches(scan(), Some(schema))
        .try_collect()
        .await?;

    assert_eq!(derived, expected);
    assert_eq!(given, expected);
    Ok(())
}

/// Per-batch work, such as executing to Arrow, must run on the driver's tasks: on the thread
/// polling the stream it is serialised, which is what slowed the decompress benchmark down.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batches_are_mapped_off_the_polling_thread() -> VortexResult<()> {
    let file = file().await?;
    let polling = thread::current().id();
    let mapped_on = Arc::new(Mutex::new(Vec::new()));
    let map_fn = {
        let mapped_on = Arc::clone(&mapped_on);
        Arc::new(move |batch: ArrayRef| {
            mapped_on.lock().push(thread::current().id());
            Ok(batch)
        })
    };

    let batches: Vec<ArrayRef> = driver()
        .drive(ScanBuilder::new(SESSION.clone(), file.scan_file()), map_fn)
        .try_collect()
        .await?;

    let mapped_on = mapped_on.lock();
    assert!(!batches.is_empty());
    assert_eq!(mapped_on.len(), batches.len());
    assert!(mapped_on.iter().all(|thread| *thread != polling));
    Ok(())
}

/// A file of one `u32` column counting up from zero, spread over many scan chunks.
async fn counting_file(rows: u32) -> VortexResult<VortexFile> {
    let numbers = ChunkedArray::from_iter(
        (0..rows)
            .step_by(10_000)
            .map(|start| Buffer::from_iter(start..(start + 10_000).min(rows)).into_array()),
    )
    .into_array();
    let array = StructArray::from_fields(&[("numbers", numbers)])?;
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.into_array().to_array_stream())
        .await?;
    SESSION.open_options().open_buffer(buf)
}

/// The `numbers` of `batches`, in the order the batches hold them.
fn numbers(batches: Vec<ArrayRef>) -> VortexResult<Vec<u32>> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut numbers = Vec::new();
    for batch in batches {
        let batch = batch.execute::<StructArray>(&mut ctx)?;
        let column = batch
            .unmasked_field_by_name("numbers")?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        numbers.extend_from_slice(column.as_slice::<u32>());
    }
    Ok(numbers)
}

#[rstest]
#[case::inline(1)]
#[case::spawned(4)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ordered_scans_stream_rows_in_order(
    #[case] concurrency: usize,
    #[values(false, true)] filtered: bool,
) -> VortexResult<()> {
    let rows = 400_000u32;
    let file = counting_file(rows).await?;
    let mut scan = ScanBuilder::new(SESSION.clone(), file.scan_file());
    if filtered {
        let filter = gt(get_item("numbers", root()), lit(1_000u32)).bind(file.dtype())?;
        scan = scan.with_some_filter(Some(filter));
    }

    let batches: Vec<ArrayRef> = ScanDriver::new(TokioRuntime::current(), concurrency, true)
        .run(scan)
        .try_collect()
        .await?;

    let start = if filtered { 1_001 } else { 0 };
    assert!(batches.len() > 1, "the scan should cover several chunks");
    assert_eq!(numbers(batches)?, (start..rows).collect::<Vec<_>>());
    Ok(())
}

#[rstest]
#[case::within_a_chunk(10)]
#[case::across_chunks(70_000)]
#[case::past_the_end(1_000_000)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn limits_end_the_scan(#[case] limit: u64) -> VortexResult<()> {
    let rows = 200_000u32;
    let file = counting_file(rows).await?;
    let expected = u32::try_from(limit.min(u64::from(rows)))?;
    let scan = || ScanBuilder::new(SESSION.clone(), file.scan_file());

    let ordered: Vec<ArrayRef> = ScanDriver::new(TokioRuntime::current(), 2, true)
        .with_limit(Some(limit))
        .run(scan())
        .try_collect()
        .await?;
    assert_eq!(numbers(ordered)?, (0..expected).collect::<Vec<_>>());

    let unordered: Vec<ArrayRef> = ScanDriver::new(TokioRuntime::current(), 2, false)
        .with_limit(Some(limit))
        .run(scan())
        .try_collect()
        .await?;
    let mut unordered = numbers(unordered)?;
    unordered.sort_unstable();
    unordered.dedup();
    assert_eq!(unordered.len(), expected as usize);

    let schema = Arc::new(SESSION.arrow().to_arrow_schema(scan().dtype())?);
    let record_batches: Vec<RecordBatch> = ScanDriver::new(TokioRuntime::current(), 2, true)
        .with_limit(Some(limit))
        .run_record_batches(scan(), Some(schema))
        .try_collect()
        .await?;
    let total: usize = record_batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(total, expected as usize);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_scans_run_like_builders() -> VortexResult<()> {
    let file = counting_file(150_000).await?;
    let filter = gt(get_item("numbers", root()), lit(70_000u32)).bind(file.dtype())?;
    let scan = || {
        ScanBuilder::new(SESSION.clone(), file.scan_file()).with_some_filter(Some(filter.clone()))
    };

    let built: Vec<ArrayRef> = driver().run(scan()).try_collect().await?;
    let prepared: Vec<ArrayRef> = driver()
        .run_prepared(Arc::new(scan().prepare()?))
        .try_collect()
        .await?;

    assert_eq!(numbers(prepared)?, numbers(built)?);
    Ok(())
}

/// Counts the segments a file's scans request from its source.
struct CountingSegments {
    inner: Arc<dyn SegmentSource>,
    requests: AtomicUsize,
}

impl SegmentSource for CountingSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.inner.request(id)
    }
}

/// The segment requests of two V2 scans of one cached file, opened with `retained` bytes.
async fn requests_of_two_scans(retained: Option<u64>) -> VortexResult<(usize, usize)> {
    let mut buf = ByteBufferMut::empty();
    let array =
        StructArray::from_fields(&[("numbers", Buffer::from_iter(0..100_000u32).into_array())])?;
    SESSION
        .write_options()
        .write(&mut buf, array.into_array().to_array_stream())
        .await?;
    let mut options = SESSION.open_options().with_layout_reader_cache();
    if let Some(bytes) = retained {
        options = options.with_retained_segment_bytes(bytes);
    }
    let file = options.open_buffer(buf)?;
    let counting = Arc::new(CountingSegments {
        inner: file.segment_source(),
        requests: AtomicUsize::new(0),
    });
    let file = file.with_segment_source(Arc::clone(&counting) as Arc<dyn SegmentSource>);

    let mut requests = Vec::new();
    for _ in 0..2 {
        let before = counting.requests.load(Ordering::Relaxed);
        let batches: Vec<ArrayRef> = driver()
            .run(ScanBuilder::new(SESSION.clone(), file.scan_file()))
            .try_collect()
            .await?;
        assert!(!batches.is_empty());
        requests.push(counting.requests.load(Ordering::Relaxed) - before);
    }
    Ok((requests[0], requests[1]))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retained_segment_bytes_bound_what_outlives_a_scan() -> VortexResult<()> {
    let (first, second) = requests_of_two_scans(None).await?;
    assert!(first > 0);
    assert!(
        second < first,
        "the default cache retains the first scan's segments"
    );

    let (first, second) = requests_of_two_scans(Some(0)).await?;
    assert!(first > 0);
    assert_eq!(
        second, first,
        "a cache retaining nothing reads every segment again"
    );
    Ok(())
}
