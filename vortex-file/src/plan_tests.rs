// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan data sources read through the Scan API's `split` and `open`, over real files whose
//! segments are read through the session's I/O.

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::dynamic;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSession;
use vortex_layout::scan::scan_builder::ScanBuilder;
use vortex_layout::session::LayoutSession;
use vortex_scan::DataSourceRef;
use vortex_scan::ScanRequest;
use vortex_scan::source::PlanDataSource;
use vortex_scan::stream::ScanStream;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;
use crate::multi::MultiFileSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with::<MultiFileSession>();
    crate::register_default_encodings(&session);
    crate::enable_all_registered_array_encodings(&session);
    session
});

/// The rows of a one-file source: several parts of the scan, filtered or not.
const ROWS: u32 = 600_000;

/// The `m` value of row `n`: scrambled, so the column does not compress away and its segments
/// are read during the scan rather than with the footer.
fn scramble(n: u32) -> u32 {
    n.wrapping_mul(2_654_435_761)
}

const THRESHOLD: u32 = u32::MAX / 2;

/// The filter of filtered cases, which keeps about half of the rows and, with the `n`
/// projection, makes the scan read two columns.
fn m_filter() -> Expression {
    gt(get_item("m", root()), lit(THRESHOLD))
}

/// The values of `n` in `rows` that a scan returns, with [`m_filter`] when `filtered`.
fn expected(rows: Range<u32>, filtered: bool) -> Vec<u32> {
    rows.filter(|&n| !filtered || scramble(n) > THRESHOLD)
        .collect()
}

/// A file of an `n` column holding `values` and an `m` column of their scrambles.
async fn file_bytes(values: Range<u32>) -> VortexResult<ByteBuffer> {
    let m = Buffer::from_iter(values.clone().map(scramble)).into_array();
    let array =
        StructArray::from_fields(&[("n", Buffer::from_iter(values).into_array()), ("m", m)])?
            .into_array();
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await?;
    Ok(buf.freeze())
}

/// A reader of in-memory bytes whose reads pend once before they complete, as storage reads do.
#[derive(Clone)]
struct YieldingRead(ByteBuffer);

impl VortexReadAt for YieldingRead {
    fn concurrency(&self) -> usize {
        self.0.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.0.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let read = self.0.read_at(offset, length, alignment);
        async move {
            tokio::task::yield_now().await;
            read.await
        }
        .boxed()
    }
}

async fn open_one_file() -> VortexResult<VortexFile> {
    SESSION
        .open_options()
        .open_read(YieldingRead(file_bytes(0..ROWS).await?))
        .await
}

async fn one_file() -> VortexResult<DataSourceRef> {
    Ok(open_one_file().await?.plan_data_source())
}

fn n() -> Expression {
    get_item("n", root())
}

fn request(ordered: bool) -> ScanRequest {
    ScanRequest {
        projection: n(),
        ordered,
        ..Default::default()
    }
}

async fn values(stream: ScanStream) -> VortexResult<Vec<u32>> {
    let arrays: Vec<ArrayRef> = stream.try_collect().await?;
    let mut ctx = SESSION.create_execution_ctx();
    let mut values = Vec::new();
    for array in arrays {
        values.extend_from_slice(array.execute::<PrimitiveArray>(&mut ctx)?.as_slice::<u32>());
    }
    Ok(values)
}

/// Reads `stream` with `workers` clones, each polled on a task of its own.
async fn read_stream(stream: ScanStream, workers: usize) -> VortexResult<Vec<Vec<u32>>> {
    let tasks: Vec<_> = (0..workers)
        .map(|_| tokio::spawn(values(stream.clone())))
        .collect();
    let mut out = Vec::with_capacity(workers);
    for task in tasks {
        out.push(
            task.await
                .map_err(|err| vortex_err!("worker task failed: {err}"))??,
        );
    }
    Ok(out)
}

fn sorted(workers: Vec<Vec<u32>>) -> Vec<u32> {
    let mut all: Vec<u32> = workers.into_iter().flatten().collect();
    all.sort_unstable();
    all
}

/// The clones of an unordered scan's stream together read every selected row once.
#[rstest]
#[case::projection(false)]
#[case::filter(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_reads_every_row_once(#[case] filtered: bool) -> VortexResult<()> {
    let source = one_file().await?;
    let scan = source
        .scan(ScanRequest {
            filter: filtered.then(m_filter),
            ..request(false)
        })
        .await?;
    let workers = read_stream(scan.open()?, 4).await?;
    assert_eq!(sorted(workers), expected(0..ROWS, filtered));
    Ok(())
}

/// A child's splits are its parts, each of several chunks, and the workers read every row of them once.
#[rstest]
#[case::unordered(false, false)]
#[case::unordered_filter(false, true)]
#[case::ordered(true, false)]
#[case::ordered_filter(true, true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_cuts_parts_on_file_splits(
    #[case] ordered: bool,
    #[case] filtered: bool,
) -> VortexResult<()> {
    let file = open_one_file().await?;
    let splits: Arc<[u64]> = Arc::from([0, 150_000, 400_000, u64::from(ROWS)]);
    let source: DataSourceRef = Arc::new(PlanDataSource::from_child(
        SESSION.clone(),
        file.plan_child().with_splits(splits),
    ));
    let scan = source
        .scan(ScanRequest {
            filter: filtered.then(m_filter),
            ..request(ordered)
        })
        .await?;
    let workers = read_stream(scan.open()?, 3).await?;
    if ordered {
        let reader = workers.iter().find(|values| !values.is_empty());
        assert_eq!(reader, Some(&expected(0..ROWS, filtered)));
    } else {
        // Each worker returns its parts' rows in order within each part.
        assert_eq!(sorted(workers), expected(0..ROWS, filtered));
    }
    Ok(())
}

/// A file's parts are its layout splits over the fields the scan reads, as a V1 scan of the same
/// fields cuts it: no batch of an unordered scan crosses one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_cuts_parts_on_layout_splits() -> VortexResult<()> {
    let file = open_one_file().await?;
    let splits = ScanBuilder::new(SESSION.clone(), file.layout_reader()?)
        .with_projection(n().bind(file.dtype())?)
        .full_file_splits()?;
    assert!(splits.len() > 2, "the file has several layout splits");

    // One worker takes the parts in row order and finishes each before the next.
    let scan = file.plan_data_source().scan(request(false)).await?;
    let arrays: Vec<ArrayRef> = scan.open()?.try_collect().await?;
    let mut boundaries = vec![0u64];
    for array in &arrays {
        let end = boundaries.last().copied().unwrap_or(0) + array.len() as u64;
        boundaries.push(end);
    }
    for split in splits {
        assert!(
            boundaries.contains(&split),
            "a batch crosses layout split {split}"
        );
    }
    Ok(())
}

/// A dynamic filter, such as a top-N's bound, prunes a file as a whole before its parts are cut,
/// as queued work rather than in `open`. The rows follow the bound as it stands when the workers
/// read, not as it stood when the scan was opened.
#[rstest]
#[case::unordered(false)]
#[case::ordered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_prunes_the_file_first_for_a_dynamic_filter(
    #[case] ordered: bool,
) -> VortexResult<()> {
    let bound = Arc::new(AtomicU32::new(0));
    let value = Arc::clone(&bound);
    let filter = dynamic(
        CompareOperator::Gt,
        move || Some(value.load(Ordering::Relaxed).into()),
        DType::Primitive(PType::U32, Nullability::NonNullable),
        true,
        n(),
    );
    let scan = one_file()
        .await?
        .scan(ScanRequest {
            filter: Some(filter),
            ..request(ordered)
        })
        .await?;
    let stream = scan.open()?;
    bound.store(ROWS - 10, Ordering::Relaxed);
    assert_eq!(
        sorted(read_stream(stream, 3).await?),
        (ROWS - 9..ROWS).collect::<Vec<_>>()
    );
    Ok(())
}

/// An ordered scan gives all of its rows, in order, to the first clone polled.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn ordered_open_reads_in_order_on_one_clone() -> VortexResult<()> {
    let source = one_file().await?;
    let mut workers = read_stream(source.scan(request(true)).await?.open()?, 3).await?;
    workers.sort_by_key(|values| std::cmp::Reverse(values.len()));
    assert_eq!(workers[0], (0..ROWS).collect::<Vec<_>>());
    assert!(workers[1..].iter().all(Vec::is_empty));
    Ok(())
}

/// A scan's limit holds across the clones of its stream; an ordered scan returns the first rows.
#[rstest]
#[case::unordered(false, false)]
#[case::unordered_filter(false, true)]
#[case::ordered(true, false)]
#[case::ordered_filter(true, true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_applies_the_limit(#[case] ordered: bool, #[case] filtered: bool) -> VortexResult<()> {
    const LIMIT: usize = 150_000;
    let rows = expected(0..ROWS, filtered);
    let source = one_file().await?;
    let scan = source
        .scan(ScanRequest {
            filter: filtered.then(m_filter),
            limit: Some(LIMIT as u64),
            ..request(ordered)
        })
        .await?;
    // A limit holds across the whole scan, so a limited scan is not split.
    let scan = {
        let mut pieces = scan.split(NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN))?;
        assert_eq!(pieces.len(), 1);
        pieces.remove(0)
    };
    let workers = read_stream(scan.open()?, 3).await?;
    if ordered {
        let reader = workers.iter().find(|values| !values.is_empty());
        assert_eq!(reader.map(Vec::as_slice), Some(&rows[..LIMIT]));
    }
    let all = sorted(workers);
    assert_eq!(all.len(), LIMIT);
    all.windows(2).for_each(|w| assert!(w[0] < w[1]));
    assert!(all.iter().all(|value| rows.binary_search(value).is_ok()));
    Ok(())
}
