// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan data sources read through the Scan API's `split` and `open`, over real files whose
//! segments are read through the session's I/O.

use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::stats::Precision;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::VortexReadAt;
use vortex_io::filesystem::FileListing;
use vortex_io::filesystem::FileSystem;
use vortex_io::filesystem::FileSystemRef;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_scan::DataSourceRef;
use vortex_scan::DataSourceScanRef;
use vortex_scan::PartitionRef;
use vortex_scan::ScanRequest;
use vortex_scan::pool::ScanPoolRef;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::WriteOptionsSessionExt;
use crate::multi::MultiFileDataSource;
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

/// The rows of a one-file source: several chunks of the scan's chunk grid.
const ROWS: u32 = 200_000;

/// A file of one `n` column holding `values`, with file statistics.
async fn file_bytes(values: Range<u32>) -> VortexResult<ByteBuffer> {
    let array =
        StructArray::from_fields(&[("n", Buffer::from_iter(values).into_array())])?.into_array();
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await?;
    Ok(buf.freeze())
}

async fn one_file() -> VortexResult<DataSourceRef> {
    let file = SESSION
        .open_options()
        .open_buffer(file_bytes(0..ROWS).await?)?;
    Ok(file.plan_data_source())
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

async fn values(stream: SendableArrayStream) -> VortexResult<Vec<u32>> {
    let arrays: Vec<ArrayRef> = stream.try_collect().await?;
    let mut ctx = SESSION.create_execution_ctx();
    let mut values = Vec::new();
    for array in arrays {
        values.extend_from_slice(array.execute::<PrimitiveArray>(&mut ctx)?.as_slice::<u32>());
    }
    Ok(values)
}

/// Reads `pool` with `workers` workers, each polled on a task of its own, in the order the
/// workers were taken.
async fn read_pool(pool: ScanPoolRef, workers: usize) -> VortexResult<Vec<Vec<u32>>> {
    let tasks: Vec<_> = (0..workers)
        .map(|_| tokio::spawn(values(Arc::clone(&pool).worker())))
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

/// Reads a scan with one worker of its pool.
async fn read_one(scan: DataSourceScanRef) -> VortexResult<Vec<u32>> {
    values(scan.open()?.worker()).await
}

fn sorted(workers: Vec<Vec<u32>>) -> Vec<u32> {
    let mut all: Vec<u32> = workers.into_iter().flatten().collect();
    all.sort_unstable();
    all
}

/// Split pieces are contiguous, never empty, and read every row in order.
#[rstest]
#[case::one(1)]
#[case::two(2)]
#[case::seven(7)]
#[case::many(1000)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn split_pieces_cover_all_rows_in_order(#[case] n: usize) -> VortexResult<()> {
    let source = one_file().await?;
    let scan = source.scan(request(true)).await?;
    assert_eq!(scan.row_count(), Precision::exact(u64::from(ROWS)));
    let pieces = scan.split(NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN))?;
    assert!(!pieces.is_empty() && pieces.len() <= n);
    if n > 1 {
        assert!(pieces.len() > 1);
    }
    let mut counted = 0;
    let mut read = Vec::new();
    for piece in pieces {
        let rows = piece.row_count();
        let rows = rows
            .as_exact()
            .ok_or_else(|| vortex_err!("inexact {rows:?}"))?;
        assert!(rows > 0);
        counted += rows;
        let values = read_one(piece).await?;
        assert_eq!(values.len() as u64, rows);
        read.extend(values);
    }
    assert_eq!(counted, u64::from(ROWS));
    assert_eq!(read, (0..ROWS).collect::<Vec<_>>());
    Ok(())
}

/// The workers of an unordered pool together read every selected row once.
#[rstest]
#[case::projection(None)]
#[case::filter(Some(gt(n(), lit(1_000u32))))]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_reads_every_row_once(#[case] filter: Option<Expression>) -> VortexResult<()> {
    let first = if filter.is_some() { 1_001 } else { 0 };
    let source = one_file().await?;
    let scan = source
        .scan(ScanRequest {
            filter,
            split_concurrency: NonZeroUsize::new(2),
            ..request(false)
        })
        .await?;
    let workers = read_pool(scan.open()?, 4).await?;
    assert_eq!(sorted(workers), (first..ROWS).collect::<Vec<_>>());
    Ok(())
}

/// An ordered pool gives all of its rows, in order, to its first worker.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn ordered_open_reads_in_order_on_the_first_worker() -> VortexResult<()> {
    let source = one_file().await?;
    let workers = read_pool(source.scan(request(true)).await?.open()?, 3).await?;
    assert_eq!(workers[0], (0..ROWS).collect::<Vec<_>>());
    assert!(workers[1..].iter().all(Vec::is_empty));
    Ok(())
}

/// A pool's limit holds across its workers; an ordered pool returns the first rows.
#[rstest]
#[case::unordered(false, None)]
#[case::unordered_filter(false, Some(gt(n(), lit(1_000u32))))]
#[case::ordered(true, None)]
#[case::ordered_filter(true, Some(gt(n(), lit(1_000u32))))]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_applies_the_limit(
    #[case] ordered: bool,
    #[case] filter: Option<Expression>,
) -> VortexResult<()> {
    const LIMIT: u32 = 70_000;
    let first = if filter.is_some() { 1_001 } else { 0 };
    let source = one_file().await?;
    let scan = source
        .scan(ScanRequest {
            filter,
            limit: Some(u64::from(LIMIT)),
            ..request(ordered)
        })
        .await?;
    // A limit holds across the whole scan, so a limited scan is not split.
    let scan = {
        let mut pieces = scan.split(NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN))?;
        assert_eq!(pieces.len(), 1);
        pieces.remove(0)
    };
    let workers = read_pool(scan.open()?, 3).await?;
    if ordered {
        assert_eq!(workers[0], (first..first + LIMIT).collect::<Vec<_>>());
    }
    let all = sorted(workers);
    assert_eq!(all.len(), LIMIT as usize);
    all.windows(2).for_each(|w| assert!(w[0] < w[1]));
    assert!(all.iter().all(|value| (first..ROWS).contains(value)));
    Ok(())
}

/// An in-memory file system over named files.
struct MemoryFileSystem {
    files: Vec<(String, ByteBuffer)>,
}

impl fmt::Debug for MemoryFileSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryFileSystem").finish()
    }
}

impl MemoryFileSystem {
    fn file(&self, path: &str) -> Option<&ByteBuffer> {
        self.files
            .iter()
            .find(|(name, _)| name == path)
            .map(|(_, bytes)| bytes)
    }
}

#[async_trait]
impl FileSystem for MemoryFileSystem {
    fn list(&self, _prefix: &str) -> stream::BoxStream<'_, VortexResult<FileListing>> {
        stream::empty().boxed()
    }

    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
        Ok(self.file(path).map(|bytes| FileListing {
            path: path.to_string(),
            size: Some(bytes.len() as u64),
        }))
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        let bytes = self
            .file(path)
            .ok_or_else(|| vortex_err!("no file {path}"))?
            .clone();
        Ok(Arc::new(bytes))
    }

    async fn delete(&self, _path: &str) -> VortexResult<()> {
        Ok(())
    }
}

/// The values of the files of [`three_files`].
const FILES: [Range<u32>; 3] = [0..70_000, 70_000..140_000, 140_000..210_000];

/// A V2 source over three files, of which all but the first are opened during scans.
async fn three_files() -> VortexResult<DataSourceRef> {
    let mut files = Vec::new();
    for (i, values) in FILES.into_iter().enumerate() {
        files.push((format!("{i}.vortex"), file_bytes(values).await?));
    }
    let fs: FileSystemRef = Arc::new(MemoryFileSystem { files });
    let mut builder = MultiFileDataSource::new(SESSION.clone());
    for i in 0..FILES.len() {
        builder = builder.with_glob(format!("{i}.vortex"), Some(Arc::clone(&fs)));
    }
    builder.build_plan().await
}

/// Deferred files are opened by the workers that take them, and every row of every file is read
/// once.
#[rstest]
#[case::all(None, 0)]
#[case::filter(Some(gt(n(), lit(100_000u32))), 100_001)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_reads_deferred_files(
    #[case] filter: Option<Expression>,
    #[case] first: u32,
) -> VortexResult<()> {
    let source = three_files().await?;
    assert_eq!(source.row_count(), Precision::inexact(210_000u64));
    let request = ScanRequest {
        filter,
        ..request(false)
    };
    let workers = read_pool(source.scan(request.clone()).await?.open()?, 4).await?;
    assert_eq!(sorted(workers), (first..210_000).collect::<Vec<_>>());

    // Ordered, the files are read in order.
    let ordered = ScanRequest {
        ordered: true,
        ..request
    };
    let workers = read_pool(source.scan(ordered).await?.open()?, 2).await?;
    assert_eq!(workers[0], (first..210_000).collect::<Vec<_>>());
    Ok(())
}

/// A multi-file scan splits on files that are not open yet and on the rows of the open one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn split_keeps_deferred_files_whole() -> VortexResult<()> {
    let source = three_files().await?;
    let scan = source.scan(request(true)).await?;
    let pieces = scan.split(NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN))?;
    assert!(pieces.len() > 1 && pieces.len() <= 4);
    let mut read = Vec::new();
    for piece in pieces {
        let values = read_one(piece).await?;
        assert!(!values.is_empty());
        read.extend(values);
    }
    assert_eq!(read, (0..210_000).collect::<Vec<_>>());
    Ok(())
}

/// An unordered multi-file limit holds across files and workers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn open_limits_across_deferred_files() -> VortexResult<()> {
    let source = three_files().await?;
    let scan = source
        .scan(ScanRequest {
            limit: Some(100_000),
            ..request(false)
        })
        .await?;
    let all = sorted(read_pool(scan.open()?, 4).await?);
    assert_eq!(all.len(), 100_000);
    all.windows(2).for_each(|w| assert!(w[0] < w[1]));
    Ok(())
}

/// The partitions of a multi-file scan are its files, deferred ones included.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(miri, ignore)]
async fn partitions_are_files() -> VortexResult<()> {
    let source = three_files().await?;
    let partitions: Vec<PartitionRef> = source
        .scan(request(true))
        .await?
        .partitions()
        .try_collect()
        .await?;
    assert_eq!(
        partitions.iter().map(|p| p.index()).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let mut read = Vec::new();
    for partition in partitions {
        read.extend(values(partition.execute()?).await?);
    }
    assert_eq!(read, (0..210_000).collect::<Vec<_>>());
    Ok(())
}
