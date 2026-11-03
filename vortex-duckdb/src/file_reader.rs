// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::chunked::ChunkedArrayExt as _;
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex::dtype::FieldPath;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
use vortex::file::Footer;
use vortex::file::multi::MultiFileSession;
use vortex::file::multi::open_cached;
use vortex::file::multi::parse_uri_or_path;
use vortex::file::v2::FileStatsLayoutReader;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::io::std_file::StdFileSystem;
use vortex::layout::LayoutReaderRef;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::v2::ScanBuilder as V2ScanBuilder;
use vortex::layout::scan::v2::ScanFile;
use vortex::mask::Mask;
use vortex::session::SessionExt as _;

use crate::REGISTRY;
use crate::RUNTIME;
use crate::SESSION;
use crate::column_statistics::ColumnStatistics;
use crate::column_statistics::ColumnStatisticsAggregate;
use crate::duckdb::BindResultRef;
use crate::duckdb::DataChunkRef;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::projection::Filter;
use crate::projection::extract_schema_from_dtype;
use crate::scan_v2::FileScan;
use crate::table_function::BindState;
use crate::table_function::GlobalState;
use crate::table_function::LocalState;
use crate::table_function::Split;
use crate::table_function::convert_result;

// See src/table_function.rs for definition of table function state machine.
// Duckdb's inner state machine, called by table function state machine, is
// file reader state machine. It opens Vortex files, reads their contents
// and populates output data chunks with data from these files.
//
// Definitions:
// "global lock" - lock over all threads. Only one thread may access some
//   section.
// "file-local lock" - multiple threads may access different File's in
//   parallel, but only one thread can access a single File at a time.
//
// First there's bind phase in planning, called by one thread on first
// file expanded from glob, to get scan schema and estimate overall scan
// cardinality.
//
// `reader_open` -> `reader_bind` -> `reader_get_statistics`
//
// After this, still in planning phase, one thread calls the following chain on
// all files to try and replace scanning with metadata answer:
//
// `can_get_partition_stats` -> `footer_get_cached` -> `footer_get_statistics`.
//
// Then there's query runtime phase, called for all files in scan:
//
// `reader_open` -> `reader_initialize` ->
// `reader_try_initialize_scan` -> `reader_scan`
//
// `reader_get_progress_in_file` is called during `reader_scan` calls from a
// separate thread.

fn resolve_filesystem(url: &Url) -> VortexResult<(FileSystemRef, String)> {
    // Compat makes us use tokio which is very bad for local reads on
    // high-core machines because reads go into blocking pool
    if url.scheme() == "file" {
        let path = url
            // path() is fine on Linux but produces /C:/ (lead slash) on Windows
            .to_file_path()
            .map_err(|()| vortex_err!("invalid file url {url}"))?;
        return Ok((
            Arc::new(StdFileSystem::new(RUNTIME.handle())),
            path.to_string_lossy().into_owned(),
        ));
    }

    let (object_store, path) = REGISTRY.resolve(url)?;

    Ok((
        Arc::new(ObjectStoreFileSystem::new(
            Arc::new(Compat::new(object_store)),
            RUNTIME.handle(),
        )),
        path.to_string(),
    ))
}

pub struct OpenFileReader {
    /// The layout reader and conversion cache of the file. `None` after the file is skipped or
    /// all its splits are taken.
    ///
    /// DuckDB releases its last reference to a file while it holds the global lock of the
    /// multi-file scan, and dropping this state takes long. So every local state that scans the
    /// file holds a reference too, and the file gives its own reference to a local state before
    /// DuckDB releases the file. The last local state drops the state outside the lock.
    scan_state: Option<Arc<FileScanState>>,
    /// File splits stored in inverse order
    pub splits: Vec<Split>,
    total_splits: usize,
}

/// The state of an open file that the threads scanning it share.
pub struct FileScanState {
    reader: LayoutReaderRef,
    /// The file as the V2 executor reads it.
    scan_file: ScanFile,
    /// The file's V2 scan, which every thread scanning the file drives. Set when the file is
    /// initialized for a V2 scan.
    v2: OnceLock<FileScan>,
    cache: ConversionCache,
}

impl OpenFileReader {
    async fn open(path: String) -> VortexResult<Self> {
        let (fs, fs_path) = resolve_filesystem(&parse_uri_or_path(&path)?)?;
        let source = fs.open_read(&fs_path).await?;
        let file = open_cached(&SESSION, Some(&path), source, None, &|options| options).await?;
        let mut scan_file = file.scan_file();
        scan_file.cache.get_or_insert_default();
        Ok(OpenFileReader {
            scan_state: Some(Arc::new(FileScanState {
                reader: file.layout_reader()?,
                scan_file,
                v2: OnceLock::new(),
                cache: ConversionCache::default(),
            })),
            splits: vec![],
            total_splits: 0,
        })
    }

    fn scan_state(&self) -> &FileScanState {
        self.scan_state
            .as_ref()
            .vortex_expect("file used after its scan state was released")
    }

    fn reader(&self) -> &LayoutReaderRef {
        &self.scan_state().reader
    }

    fn can_skip(&self, filter: &Filter) -> VortexResult<bool> {
        let Some(filter) = &filter.filter else {
            return Ok(false);
        };
        let row_count = self.reader().row_count();
        let row_range = 0..row_count;
        let mask = Mask::new_true(usize::try_from(row_count).unwrap_or(usize::MAX));
        let evaluation = self.reader().pruning_evaluation(&row_range, filter, mask)?;
        match evaluation.now_or_never() {
            Some(mask) => mask.map(|mask| mask.all_false()),
            None => Ok(false),
        }
    }
}

/// Called once per file while initializing the scan under file-local lock.
/// Files are opened lazily.
pub fn reader_open(file_path: &str) -> VortexResult<OpenFileReader> {
    RUNTIME.block_on(OpenFileReader::open(file_path.to_owned()))
}

/// Called once per scan with first file without locks. Populates "result"
/// with first file schema which is the scan schema. Unlike Parquet, we don't
/// support schema evolution, so if any file schema doesn't match first schema,
/// we break.
pub fn reader_bind(file: &OpenFileReader, result: &mut BindResultRef) -> VortexResult<BindState> {
    let dtype = file.reader().dtype().clone();
    let columns = extract_schema_from_dtype(&dtype)?;

    for column in &columns {
        result.add_result_column(&column.name, &column.logical_type);
    }

    Ok(BindState {
        dtype,
        first_file_row_count: file.reader().row_count(),
        filters: vec![],
        columns,
        has_non_optional_filter: AtomicBool::new(false),
        aggregates: vec![],
        no_footer_caches: false,
    })
}

/// Called once per file by one thread under file-local lock. Determines
/// whether the opened file should be skipped. If this function returns false,
/// duckdb closes the file and doesn't call reader_try_initialize_scan on it.
pub fn reader_initialize(file: &mut OpenFileReader, global: &GlobalState) -> VortexResult<bool> {
    if file.can_skip(&global.filter)? {
        // DuckDB releases a skipped file under the global lock. Here only the
        // file-local lock is held, so drop the scan state now.
        file.scan_state = None;
        return Ok(true);
    }

    // Getting splits, or planning a V2 scan, is non-trivial work so we prefer
    // doing it here under file lock and not in reader_try_initialize_scan under
    // global lock.
    if global.scan_v2 {
        let state = file.scan_state();
        let splits = ScanBuilder::new(SESSION.clone(), Arc::clone(&state.reader))
            .with_projection(global.projection.clone())
            .with_some_filter(global.filter.filter.clone())
            .full_file_splits()?;
        let dynamic = global
            .filter
            .filter
            .as_ref()
            .is_some_and(|filter| DynamicExprUpdates::new(filter).is_some());
        let scan = FileScan::try_new(v2_builder(state, global), &splits, dynamic)?;
        if state.v2.set(scan).is_err() {
            vortex_panic!("file initialized twice");
        }
        return Ok(false);
    }

    let reader = Arc::clone(file.reader());
    let filter = &global.filter;
    let builder = ScanBuilder::new(SESSION.clone(), reader)
        .with_projection(global.projection.clone())
        .with_some_filter(filter.filter.clone())
        .with_selection(filter.row_selection.clone());
    let scan = builder.prepare()?;
    let mut splits: Vec<Split> = scan
        .execute(filter.row_range.clone())?
        .into_iter()
        .map(Split::V1)
        .collect();

    // threads take last element of file.splits so we need to reverse
    splits.reverse();
    file.total_splits = splits.len();
    file.splits = splits;
    Ok(false)
}

/// The V2 scan of the file's rows that the scan's row range and selection keep.
fn v2_builder(state: &FileScanState, global: &GlobalState) -> V2ScanBuilder {
    let filter = &global.filter;
    let row_count = state.reader.row_count();
    let rows = filter.row_range.as_ref().map_or(0..row_count, |range| {
        range.start.min(row_count)..range.end.min(row_count)
    });
    V2ScanBuilder::new(SESSION.clone(), state.scan_file.clone())
        .with_projection(global.projection.clone())
        .with_some_filter(filter.filter.clone())
        .with_selection(filter.row_selection.clone())
        .with_row_range(rows)
        // DuckDB reads a batch in vectors of its own, so a part is exported as one batch, as a
        // V1 split is.
        .with_max_batch_rows(usize::MAX)
}

/// What a thread's split yields next.
enum SplitItem {
    /// The split has no more rows.
    Done,
    /// The split produced no rows this time but may produce more.
    Empty,
    Array(ArrayRef),
}

/// Takes the next array of the thread's split, keeping the split while it has more to yield.
fn next_split_item(local: &mut LocalState) -> VortexResult<SplitItem> {
    match local.split.take() {
        None => Ok(SplitItem::Done),
        Some(Split::V1(task)) => Ok(RUNTIME
            .block_on(task)?
            .map_or(SplitItem::Empty, SplitItem::Array)),
        Some(Split::V2) => {
            let file = local.file.as_ref().vortex_expect("split without a file");
            let scan = file
                .v2
                .get()
                .vortex_expect("V2 split of a file without a V2 scan");
            let array = match local.pending.pop() {
                Some(array) => array,
                None => {
                    let Some(array) = scan.next_array(&mut local.run)? else {
                        return Ok(SplitItem::Done);
                    };
                    // A V2 batch can join chunks of several layout chunks. Canonicalizing it
                    // would decode their dictionaries, so export each chunk on its own, as a V1
                    // scan yields them.
                    match array.as_opt::<Chunked>() {
                        Some(chunked) => {
                            local.pending.extend(chunked.chunks().into_iter().rev());
                            local.pending.pop().unwrap_or(array)
                        }
                        None => array,
                    }
                }
            };
            local.split = Some(Split::V2);
            Ok(if array.is_empty() {
                SplitItem::Empty
            } else {
                SplitItem::Array(array)
            })
        }
    }
}

/// Called by all threads under global lock. If this function returns true,
/// thread calls reader_scan on this file. If this function returns false,
/// duckdb thinks file is exhausted, closes the file, and the first thread to
/// get "false" switches to next file.
///
/// This function must not drop a file's scan state, because the caller holds
/// the global lock. It only moves references between the file and the local
/// state, and reader_scan drops the retired ones.
pub fn reader_try_initialize_scan(file: &mut OpenFileReader, local: &mut LocalState) -> bool {
    // A V2 scan's threads take chunks from the file's scan until it has spawned
    // them all, so the file has work for another thread until then.
    let v2 = file
        .scan_state
        .as_ref()
        .and_then(|state| state.v2.get())
        .map(|scan| (!scan.is_exhausted()).then_some(Split::V2));
    let Some(split) = v2.unwrap_or_else(|| file.splits.pop()) else {
        // DuckDB releases the file after this call, under the global lock.
        // Give the reference of the file to this thread so it drops outside.
        local.retired_files.extend(file.scan_state.take());
        return false;
    };

    let scan_state = file
        .scan_state
        .as_ref()
        .vortex_expect("file has splits but no scan state");
    if !local
        .file
        .as_ref()
        .is_some_and(|current| Arc::ptr_eq(current, scan_state))
    {
        let previous = local.file.replace(Arc::clone(scan_state));
        local.retired_files.extend(previous);
    }

    local.split = Some(split);
    true
}

/// Called by all threads operating on a file without locks. If this function
/// returns false, duckdb closes the file, and first thread to get "false"
/// switches to next file.
pub fn reader_scan(
    global: &GlobalState,
    local: &mut LocalState,
    chunk: &mut DataChunkRef,
) -> VortexResult<bool> {
    // No lock is held here, so drop the scan states of files this thread left.
    local.retired_files.clear();

    if !global.aggregates.is_empty() {
        return reader_scan_aggregate(global, local);
    }

    if local.exporter.is_none() {
        let array = match next_split_item(local)? {
            SplitItem::Done => return Ok(false),
            // split is filtered
            SplitItem::Empty => return Ok(true),
            SplitItem::Array(array) => array,
        };
        let mut ctx = SESSION.create_execution_ctx();
        let array = convert_result(array, &mut ctx)?;
        let file = local.file.as_ref().vortex_expect("split without a file");
        local.exporter = Some(ArrayExporter::try_new(&array, &file.cache, ctx)?);
    }
    let exporter = local.exporter.as_mut().vortex_expect("no exporter");

    let has_more_data = exporter.export(chunk, global.file_row_number_column_pos)?;
    if !has_more_data {
        local.exporter = None;
    }
    Ok(true)
}

fn reader_scan_aggregate(global: &GlobalState, local: &mut LocalState) -> VortexResult<bool> {
    let array = match next_split_item(local)? {
        SplitItem::Done => return Ok(false),
        // split is filtered
        SplitItem::Empty => return Ok(true),
        SplitItem::Array(array) => array,
    };

    let mut ctx = SESSION.create_execution_ctx();
    let array = convert_result(array, &mut ctx)?;

    for (position, partial) in local.partials.iter_mut() {
        partial.accumulate(array.unmasked_field(*position), &mut ctx)?;
    }

    if global.has_count_star {
        let len = array.len() as u64;
        global.row_count.fetch_add(len, Ordering::Relaxed);
    }

    Ok(true)
}

/// Called by one thread in plan phase without locks only on the first file
/// after calling reader_open and reader_bind on it.
pub fn reader_get_statistics(
    file: &OpenFileReader,
    bind: &BindState,
    column: &str,
) -> Option<ColumnStatistics> {
    if !bind.aggregates.is_empty() {
        return None;
    }

    let reader = file
        .reader()
        .as_any()
        .downcast_ref::<FileStatsLayoutReader>()?;

    let (stats, dtype) = reader
        .file_stats()
        .get_by_path(&FieldPath::from_name(column))?;

    let stats = ColumnStatisticsAggregate::new(stats);
    match ColumnStatistics::try_from(stats, dtype.clone()) {
        Ok(stats) => Some(stats),
        Err(e) => vortex_panic!(e),
    }
}

/// Called from a separate thread (not related to threads for Vortex
/// table function) under global lock.
pub fn reader_get_progress_in_file(file: &OpenFileReader) -> f64 {
    if let Some(scan) = file.scan_state.as_ref().and_then(|state| state.v2.get()) {
        return scan.progress();
    }
    let total = file.total_splits;
    let left = file.splits.len();
    let denom = total + (total == 0) as usize;
    100.0 * (total - left) as f64 / denom as f64
}

/// Called by one thread in planning phase
pub fn can_get_partition_stats(bind: &BindState) -> bool {
    // Re-reading footers on every request is costly so we mimic DuckDB's
    // TryLoadCaches for Parquet and load them once per file
    !bind.no_footer_caches
    // This function is called during planning where we haven't read data yet.
    // If there's a filter, we need to read the data to evaluate stats.
    // Even if we could do this fast, we report some filters here as not pushed
    // (see table_function.rs) and stats would be incorrect.
        && bind.filters.is_empty()
}

/// Called by one thread for every file. When we open files, we cache footers
/// in MultiFileSession so this function retrieves the cached data if present.
/// If any footer is not present, it sets a flag in BindState so we won't try
/// again.
pub fn footer_get_cached(bind: &mut BindState, path: &str) -> VortexResult<Option<Footer>> {
    let footer = SESSION
        .get_opt::<MultiFileSession>()
        .vortex_expect("MultiFileSession not found")
        .get_footer(path);
    bind.no_footer_caches |= footer.is_none();
    Ok(footer)
}

/// Called by one thread for every footer in planning phase
pub fn footer_get_statistics(footer: &Footer, column: &str) -> Option<ColumnStatistics> {
    let stats = footer.statistics()?;
    let (stats, dtype) = stats.get_by_path(&FieldPath::from_name(column))?;
    let stats = ColumnStatisticsAggregate::new(stats);
    match ColumnStatistics::try_from(stats, dtype.clone()) {
        Ok(stats) => Some(stats),
        Err(e) => vortex_panic!(e),
    }
}
