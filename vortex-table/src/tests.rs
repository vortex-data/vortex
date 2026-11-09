// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use futures::TryStreamExt;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::expr::col;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::VortexReadAt;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_scan::planning::drive::ParallelDriver;
use vortex_scan::planning::drive::PlanDriver;
use vortex_scan::planning::drive::ScopedBatch;
use vortex_scan::planning::drive::SequentialDriver;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::FileCache;
use crate::Manifest;
use crate::ManifestBuilder;
use crate::MemoryStorage;
use crate::TableScan;
use crate::TableStorage;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

/// Counts the reads of each file, to show which files a scan touched.
struct CountingStorage {
    inner: Arc<MemoryStorage>,
    reads: Arc<Mutex<HashMap<String, usize>>>,
    opens: Mutex<usize>,
}

impl CountingStorage {
    fn new(inner: Arc<MemoryStorage>) -> Self {
        Self {
            inner,
            reads: Arc::default(),
            opens: Mutex::new(0),
        }
    }

    fn reads(&self, path: &str) -> usize {
        self.reads.lock().get(path).copied().unwrap_or(0)
    }
}

impl TableStorage for CountingStorage {
    fn open(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        *self.opens.lock() += 1;
        Ok(Arc::new(CountingRead {
            inner: self.inner.open(path)?,
            path: path.to_string(),
            reads: Arc::clone(&self.reads),
        }))
    }
}

struct CountingRead {
    inner: Arc<dyn VortexReadAt>,
    path: String,
    reads: Arc<Mutex<HashMap<String, usize>>>,
}

impl VortexReadAt for CountingRead {
    fn concurrency(&self) -> usize {
        self.inner.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.inner.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        *self.reads.lock().entry(self.path.clone()).or_default() += 1;
        self.inner.read_at(offset, length, alignment)
    }
}

const FILES: u64 = 4;
const ROWS_PER_FILE: u64 = 1000;

/// File `f` holds ids `f * 1000 .. (f + 1) * 1000`, and a name per id.
async fn data_file(file: u64) -> VortexResult<ByteBuffer> {
    let ids = file * ROWS_PER_FILE..(file + 1) * ROWS_PER_FILE;
    let names: Vec<String> = ids.clone().map(|id| format!("row-{id}")).collect();
    let array = StructArray::from_fields(&[
        ("id", PrimitiveArray::from_iter(ids).into_array()),
        ("name", VarBinArray::from(names).into_array()),
    ])?;
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.into_array().to_array_stream())
        .await?;
    Ok(buf.freeze())
}

/// A table of [`FILES`] files in memory, its manifest round-tripped through a Vortex file.
async fn table(inline_footers: bool) -> VortexResult<(Arc<MemoryStorage>, Arc<Manifest>)> {
    let storage = Arc::new(MemoryStorage::new());
    let mut builder = ManifestBuilder::new(SESSION.clone()).with_inline_footers(inline_footers);
    for file in 0..FILES {
        let path = format!("data/part-{file}.vortex");
        let bytes = data_file(file).await?;
        builder.add(path.clone(), &bytes).await?;
        storage.put(path, bytes);
    }
    let manifest = builder.finish();
    let mut buf = ByteBufferMut::empty();
    manifest.write(&SESSION, &mut buf).await?;
    let read = Manifest::read(&SESSION, Arc::new(buf.freeze())).await?;
    assert_eq!(read, manifest);
    Ok((storage, Arc::new(read)))
}

fn drivers() -> Vec<Box<dyn PlanDriver>> {
    vec![
        Box::new(SequentialDriver::new()),
        Box::new(ParallelDriver::new(TokioRuntime::current()).with_workers(3)),
    ]
}

/// The sorted ids of every batch.
fn ids(batches: Vec<ScopedBatch>) -> VortexResult<Vec<u64>> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut ids = Vec::new();
    for batch in batches {
        let array = batch.batch.execute::<StructArray>(&mut ctx)?;
        let column = array
            .unmasked_field_by_name("id")?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        ids.extend_from_slice(column.as_slice::<u64>());
    }
    ids.sort_unstable();
    Ok(ids)
}

#[rstest]
#[case::inline_footers(true)]
#[case::read_footers(false)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scans_every_file(#[case] inline: bool) -> VortexResult<()> {
    let (storage, manifest) = table(inline).await?;
    assert_eq!(manifest.row_count(), FILES * ROWS_PER_FILE);
    for driver in drivers() {
        let scan = TableScan::new(
            SESSION.clone(),
            Arc::clone(&storage) as _,
            Arc::clone(&manifest),
        );
        let batches: Vec<ScopedBatch> = scan.execute(driver.as_ref()).try_collect().await?;
        assert_eq!(
            ids(batches)?,
            (0..FILES * ROWS_PER_FILE).collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[rstest]
#[case::inline_footers(true)]
#[case::read_footers(false)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filters_and_prunes_files(#[case] inline: bool) -> VortexResult<()> {
    let (storage, manifest) = table(inline).await?;
    for driver in drivers() {
        let counting = Arc::new(CountingStorage::new(Arc::clone(&storage)));
        let batches: Vec<ScopedBatch> = TableScan::new(
            SESSION.clone(),
            Arc::clone(&counting) as _,
            Arc::clone(&manifest),
        )
        .with_filter(gt_eq(col("id"), lit(2500u64)))
        .execute(driver.as_ref())
        .try_collect()
        .await?;
        assert!(batches.iter().all(|batch| batch.scope.file_ordinal >= 2));
        assert_eq!(
            ids(batches)?,
            (2500..FILES * ROWS_PER_FILE).collect::<Vec<_>>()
        );
        // Files 0 and 1 are pruned on their footer statistics: with inlined footers they are
        // never read, and otherwise only their footer region is.
        let footer_reads = usize::from(!inline);
        let paths: Vec<&str> = manifest.entries().iter().map(|e| e.path.as_str()).collect();
        assert_eq!(counting.reads(paths[0]), footer_reads);
        assert_eq!(counting.reads(paths[1]), footer_reads);
        assert!(counting.reads(paths[2]) > footer_reads);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reads_only_selected_files_and_rows() -> VortexResult<()> {
    let (storage, manifest) = table(false).await?;
    let selections = HashMap::from_iter([
        (
            1,
            Selection::IncludeByIndex(StrictSortedBuffer::try_new(buffer![0u64, 5, 999])?),
        ),
        (3, Selection::All),
    ]);
    for driver in drivers() {
        let batches: Vec<ScopedBatch> = TableScan::new(
            SESSION.clone(),
            Arc::clone(&storage) as _,
            Arc::clone(&manifest),
        )
        .with_file_selections(selections.clone())
        .execute(driver.as_ref())
        .try_collect()
        .await?;
        let mut expected = vec![1000, 1005, 1999];
        expected.extend(3000..4000);
        assert_eq!(ids(batches)?, expected);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rewritten_file_fails_the_scan() -> VortexResult<()> {
    let (storage, manifest) = table(false).await?;
    // Same path, different bytes: the footer no longer hashes to what the manifest recorded.
    let path = &manifest.entries()[2].path;
    storage.put(path.clone(), data_file(1).await?);
    let result: VortexResult<Vec<ScopedBatch>> = TableScan::new(
        SESSION.clone(),
        Arc::clone(&storage) as Arc<dyn TableStorage>,
        manifest,
    )
    .execute(&SequentialDriver::new())
    .try_collect()
    .await;
    assert!(result.is_err());
    Ok(())
}

#[rstest]
#[case::inline_footers(true)]
#[case::read_footers(false)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_file_cache_skips_opening_files_again(#[case] inline: bool) -> VortexResult<()> {
    let (storage, manifest) = table(inline).await?;
    let counting = Arc::new(CountingStorage::new(storage));
    let cache = Arc::new(FileCache::new());
    let scan = || {
        TableScan::new(
            SESSION.clone(),
            Arc::clone(&counting) as _,
            Arc::clone(&manifest),
        )
        .with_file_cache(Arc::clone(&cache))
    };
    let first: Vec<ScopedBatch> = scan()
        .execute(&SequentialDriver::new())
        .try_collect()
        .await?;
    assert_eq!(*counting.opens.lock(), usize::try_from(FILES)?);
    let second: Vec<ScopedBatch> = scan()
        .execute(&SequentialDriver::new())
        .try_collect()
        .await?;
    // No file is opened, or its footer read, a second time.
    assert_eq!(*counting.opens.lock(), usize::try_from(FILES)?);
    assert_eq!(ids(first)?, ids(second)?);
    Ok(())
}
