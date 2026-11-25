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
use vortex_array::arrays::Chunked;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::chunked::ChunkedArrayExt as _;
use vortex_array::arrays::struct_::StructArrayExt as _;
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
use vortex_layout::scan::options::LayoutScanOptions;
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::session::LayoutSession;
use vortex_scan::DataSourceRef;
use vortex_scan::ScanRequest;
use vortex_scan::prepare::ScanBuilder as PreparedScanBuilder;
use vortex_scan::split::MAX_PART_ROWS;
use vortex_scan::split::MIN_BATCH_ROWS;
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

/// No batch of an unordered scan crosses from one of its parts to the next.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_returns_batches_within_parts() -> VortexResult<()> {
    let file = open_one_file().await?;
    let (_, parts) = part_points(&file, &n(), None, None, 1).await?;
    // One worker takes the parts in row order and finishes each before the next.
    let scan = file.plan_data_source().scan(request()).await?;
    let arrays: Vec<ArrayRef> = scan.open()?.stream().try_collect().await?;
    let mut boundaries = vec![0u64];
    for array in &arrays {
        let end = boundaries.last().copied().unwrap_or(0) + array.len() as u64;
        boundaries.push(end);
    }
    assert_eq!(boundaries.last().copied(), Some(u64::from(ROWS)));
    for cut in parts {
        assert!(
            boundaries.contains(&cut),
            "a batch crosses the part boundary at row {cut}"
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

/// The rows of [`chunked_strings`].
const STRING_ROWS: u64 = 400_000;

/// A file of an `s` column of strings, whose many distinct values the writer cuts into chunks off
/// the 8192-row grid, beside an `n` column of row numbers.
async fn chunked_strings() -> VortexResult<VortexFile> {
    let s = VarBinViewArray::from_iter_str(
        (0..STRING_ROWS).map(|row| format!("{:0>40}", (row / 30_000) * 5000 + row % 5000)),
    )
    .into_array();
    let n = Buffer::from_iter(0..STRING_ROWS).into_array();
    let array = StructArray::from_fields(&[("s", s), ("n", n)])?.into_array();
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await?;
    SESSION.open_options().open_buffer(buf.freeze())
}

/// The points at which each executor cuts a scan of `file` into at least `min_parts` parts, each
/// measuring the parts by the rows its own zone pruning of the filter's static conjuncts keeps.
async fn part_points(
    file: &VortexFile,
    projection: &Expression,
    filter: Option<&Expression>,
    rows: Option<Range<u64>>,
    min_parts: usize,
) -> VortexResult<(Vec<u64>, Vec<u64>)> {
    let dtype = file.dtype();
    let projection = projection.bind(dtype)?.optimize_recursive()?;
    let filter = filter
        .map(|filter| filter.bind(dtype)?.optimize_recursive())
        .transpose()?;
    let v1 = file
        .scan()?
        .with_projection(projection.clone())
        .with_some_filter(filter.clone())
        .with_split_by(SplitBy::Parts { min_parts });
    let v2 = PreparedScanBuilder::new(SESSION.clone(), file.scan_file())
        .with_projection(projection)
        .with_some_filter(filter);
    let (v1, v2) = match rows {
        Some(rows) => (v1.with_row_range(rows.clone()), v2.with_row_range(rows)),
        None => (v1, v2),
    };
    let v1 = v1.with_static_pruning().await?;
    let v2 = Arc::new(v2.prepare()?);
    let kept = v2.static_kept().await?;
    Ok((
        v1.split_points()?,
        v2.part_points(min_parts, kept.as_ref())?,
    ))
}

/// Both executors cut a scan into the same parts, so comparing them compares their execution.
/// A scan whose zones prune few rows has at least as many parts as its floor asks for, where its
/// rows hold the blocks; [`parts_share_a_selective_filters_rows`] covers the selective filter's.
#[rstest]
#[case::projection(get_item("s", root()), None, true)]
#[case::identity(root(), None, true)]
#[case::filter_on_two_fields(
    get_item("n", root()),
    Some(gt(get_item("s", root()), lit(format!("{:0>40}", 2500)))),
    true
)]
#[case::filter_on_one_narrow_field(
    get_item("n", root()),
    Some(gt(get_item("n", root()), lit(10u64))),
    true
)]
#[case::selective_filter(
    get_item("s", root()),
    Some(gt(get_item("n", root()), lit(SELECTED_FROM))),
    false
)]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn executors_cut_the_same_parts(
    #[case] projection: Expression,
    #[case] filter: Option<Expression>,
    #[case] keeps_most_rows: bool,
    #[values(None, Some(1000..350_123))] rows: Option<Range<u64>>,
    #[values(1, 18, 64)] min_parts: usize,
) -> VortexResult<()> {
    let file = chunked_strings().await?;
    let (v1, v2) =
        part_points(&file, &projection, filter.as_ref(), rows.clone(), min_parts).await?;
    assert_eq!(v1, v2);
    let rows = rows.unwrap_or(0..STRING_ROWS);
    assert_eq!(v2.first(), Some(&rows.start));
    assert_eq!(v2.last(), Some(&rows.end));
    if keeps_most_rows {
        let blocks = usize::try_from((rows.end - rows.start).div_ceil(8192))?;
        assert!(v2.len() > min_parts.min(blocks));
    }
    Ok(())
}

/// The first row a selective filter on [`chunked_strings`] keeps: its zones prune all rows before
/// it, about nine tenths of the file.
const SELECTED_FROM: u64 = 360_000;

/// A scan of fewer rows than [`MAX_PART_ROWS`] with no floor on its parts is one part: its chunks
/// are joined rather than cut on the row grid.
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn small_scans_are_one_part() -> VortexResult<()> {
    let file = chunked_strings().await?;
    const { assert!(STRING_ROWS < MAX_PART_ROWS) };
    for projection in [get_item("s", root()), root()] {
        let (v1, v2) = part_points(&file, &projection, None, None, 1).await?;
        assert_eq!(v1, vec![0, STRING_ROWS]);
        assert_eq!(v2, vec![0, STRING_ROWS]);
    }
    Ok(())
}

/// A selective filter's parts hold about equal surviving rows, so the rows its zones keep are
/// spread over a host's threads rather than left in the last part, and the rows its zones prune
/// are not cut at all.
#[rstest]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn parts_share_a_selective_filters_rows(
    #[values(4, 8)] min_parts: usize,
) -> VortexResult<()> {
    let file = chunked_strings().await?;
    let filter = gt(get_item("n", root()), lit(SELECTED_FROM));
    let (_, points) = part_points(
        &file,
        &get_item("s", root()),
        Some(&filter),
        None,
        min_parts,
    )
    .await?;
    // The zones of `n` hold 8192 rows, so the first kept zone is the one that holds the first
    // selected row.
    let kept_from = SELECTED_FROM / 8192 * 8192;
    let kept_blocks = usize::try_from((STRING_ROWS - kept_from).div_ceil(8192))?;
    assert!(
        points[1..points.len() - 1]
            .iter()
            .all(|&point| point > kept_from),
        "{points:?}"
    );
    assert!(points.len() > min_parts.min(kept_blocks), "{points:?}");
    let surviving: Vec<u64> = points
        .windows(2)
        .map(|pair| pair[1] - pair[0].max(kept_from))
        .collect();
    let (least, most) = (surviving.iter().min(), surviving.iter().max());
    assert!(
        most.zip(least)
            .is_some_and(|(most, least)| most - least <= 2 * 8192),
        "{points:?}"
    );
    Ok(())
}

/// The chunk boundaries of the fields of a struct `batch` that lie inside it.
fn field_bounds(batch: &ArrayRef) -> Vec<usize> {
    batch
        .as_opt::<Struct>()
        .into_iter()
        .flat_map(|array| {
            array
                .iter_unmasked_fields()
                .filter_map(|field| field.as_opt::<Chunked>())
                .flat_map(|field| field.chunk_offset_values().to_vec())
                .collect::<Vec<_>>()
        })
        .filter(|&bound| bound > 0 && bound < batch.len())
        .collect()
}

/// Each executor, splitting a scan into the same parts, cuts its batches at the chunks of their
/// fields: a boundary is left inside a batch only within [`MIN_BATCH_ROWS`] of its start or end,
/// where cutting it would leave a narrower batch.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn batches_are_cut_at_field_chunks(
    #[values(false, true)] v2: bool,
    #[values(false, true)] filtered: bool,
) -> VortexResult<()> {
    let file = chunked_strings().await?;
    let source = if v2 {
        file.plan_data_source()
    } else {
        file.layout_reader_data_source(
            LayoutScanOptions::default().with_split_by(SplitBy::Parts { min_parts: 1 }),
        )?
    };
    // Keeps every row but the first ten, so the batches have the same chunks as unfiltered.
    let filter = filtered.then(|| gt(get_item("n", root()), lit(9u64)));
    let scan = source
        .scan(ScanRequest {
            filter,
            ..Default::default()
        })
        .await?;
    let batches: Vec<ArrayRef> = scan.open()?.stream().try_collect().await?;
    assert_eq!(
        batches.iter().map(|batch| batch.len() as u64).sum::<u64>(),
        STRING_ROWS - if filtered { 10 } else { 0 }
    );
    for batch in &batches {
        assert!(!batch.is::<Chunked>(), "a batch is chunked");
        for bound in field_bounds(batch) {
            assert!(
                bound < MIN_BATCH_ROWS || batch.len() - bound < MIN_BATCH_ROWS,
                "a batch of {} rows holds a chunk boundary at {bound}",
                batch.len()
            );
        }
    }
    Ok(())
}

/// A V1 scan split into parts gives a host one piece per thread it asks for where the file has
/// the blocks, and the pieces together read every selected row once.
#[rstest]
#[case::projection_few_pieces(false, 4)]
#[case::filter_many_pieces(true, 32)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn v1_parts_pieces_read_every_row_once(
    #[case] filtered: bool,
    #[case] n: usize,
) -> VortexResult<()> {
    let source = open_one_file().await?.layout_reader_data_source(
        LayoutScanOptions::default().with_split_by(SplitBy::Parts { min_parts: 1 }),
    )?;
    let scan = source
        .scan(ScanRequest {
            filter: filtered.then(m_filter),
            ..request()
        })
        .await?;
    let pieces = scan.split(NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN))?;
    assert_eq!(pieces.len(), n);
    let mut workers = Vec::new();
    for piece in pieces {
        workers.extend(read_stream(piece.open()?, 2).await?);
    }
    assert_eq!(sorted(workers), expected(0..ROWS, filtered));
    Ok(())
}
