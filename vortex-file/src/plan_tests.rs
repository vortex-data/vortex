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
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::Expression;
use vortex_expr::get_item;
use vortex_expr::gt;
use vortex_expr::lit;
use vortex_expr::root;
use vortex_functions::expr::dynamic;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_scan::DataSourceRef;
use vortex_scan::ScanRequest;
use vortex_scan::stream::ScanExec;
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

fn request() -> ScanRequest {
    ScanRequest {
        projection: n(),
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

/// Reads `exec` with `workers` streams, each polled on a task of its own.
async fn read_stream(exec: ScanExec, workers: usize) -> VortexResult<Vec<Vec<u32>>> {
    let tasks: Vec<_> = (0..workers)
        .map(|_| tokio::spawn(values(exec.stream())))
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

/// The streams of an unordered scan together read every selected row once.
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
            ..request()
        })
        .await?;
    let workers = read_stream(scan.open()?, 4).await?;
    assert_eq!(sorted(workers), expected(0..ROWS, filtered));
    Ok(())
}

/// The pieces of a split scan together read every selected row once, whether the scan has
/// parts enough for every piece or cuts its parts on grid blocks to make them.
#[rstest]
#[case::projection_few_pieces(false, 4)]
#[case::filter_few_pieces(true, 4)]
#[case::projection_many_pieces(false, 32)]
#[case::filter_many_pieces(true, 32)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn split_pieces_read_every_row_once(
    #[case] filtered: bool,
    #[case] n: usize,
) -> VortexResult<()> {
    let scan = one_file()
        .await?
        .scan(ScanRequest {
            filter: filtered.then(m_filter),
            ..request()
        })
        .await?;
    let pieces = scan.split(NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN))?;
    // The file has the blocks for every piece, so a host gets one per thread it asked for.
    assert_eq!(pieces.len(), n);
    let mut workers = Vec::new();
    for piece in pieces {
        workers.extend(read_stream(piece.open()?, 2).await?);
    }
    assert_eq!(sorted(workers), expected(0..ROWS, filtered));
    Ok(())
}

/// A scan's parts are its chunks: no batch of an unordered scan crosses the chunk grid.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_cuts_parts_on_the_chunk_grid() -> VortexResult<()> {
    // A projection of one of the file's two fields plans chunks of 64Ki rows.
    const CHUNK_ROWS: u64 = 65_536;
    // One worker takes the parts in row order and finishes each before the next.
    let scan = one_file().await?.scan(request()).await?;
    let arrays: Vec<ArrayRef> = scan.open()?.stream().try_collect().await?;
    let mut boundaries = vec![0u64];
    for array in &arrays {
        let end = boundaries.last().copied().unwrap_or(0) + array.len() as u64;
        boundaries.push(end);
    }
    assert_eq!(boundaries.last().copied(), Some(u64::from(ROWS)));
    for grid in (CHUNK_ROWS..u64::from(ROWS)).step_by(CHUNK_ROWS as usize) {
        assert!(
            boundaries.contains(&grid),
            "a batch crosses the chunk boundary at row {grid}"
        );
    }
    Ok(())
}

/// A dynamic filter, such as a top-N's bound, prunes each part as it stands when a stream takes
/// the part, whatever the part's size. The rows follow the bound as it stands when the streams
/// read, not as it stood when the scan was made.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn parts_prune_on_the_dynamic_filter_as_it_stands() -> VortexResult<()> {
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
            ..request()
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

/// A scan's limit holds across its streams.
#[rstest]
#[case::projection(false)]
#[case::filter(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_applies_the_limit(#[case] filtered: bool) -> VortexResult<()> {
    const LIMIT: usize = 150_000;
    let rows = expected(0..ROWS, filtered);
    let source = one_file().await?;
    let scan = source
        .scan(ScanRequest {
            filter: filtered.then(m_filter),
            limit: Some(LIMIT as u64),
            ..request()
        })
        .await?;
    // A limit holds across the whole scan, so a limited scan is not split.
    let scan = {
        let mut pieces = scan.split(NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN))?;
        assert_eq!(pieces.len(), 1);
        pieces.remove(0)
    };
    let workers = read_stream(scan.open()?, 3).await?;
    let all = sorted(workers);
    assert_eq!(all.len(), LIMIT);
    all.windows(2).for_each(|w| assert!(w[0] < w[1]));
    assert!(all.iter().all(|value| rows.binary_search(value).is_ok()));
    Ok(())
}
