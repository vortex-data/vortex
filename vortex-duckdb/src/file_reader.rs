// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::StreamExt;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::dtype::FieldPath;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
use vortex::file::Footer;
use vortex::file::multi::MultiFileSession;
use vortex::file::multi::open_cached;
use vortex::file::multi::parse_uri_or_path;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::io::std_file::StdFileSystem;
use vortex::layout::scan::options::LayoutScanOptions;
use vortex::layout::scan::split_by::SplitBy;
use vortex::scan::DataSourceRef;
use vortex::scan::ScanRequest;
use vortex::scan::stream::Claims;
use vortex::scan::stream::ScanExec;
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
use crate::projection::extract_schema_from_dtype;
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
    /// The source, scan and conversion cache of the file. `None` after the file is skipped
    /// or no thread can join it any more.
    ///
    /// DuckDB releases its last reference to a file while it holds the global lock of the
    /// multi-file scan, and dropping this state takes long. So every local state that scans the
    /// file holds a reference too, and the file gives its own reference to a local state before
    /// DuckDB releases the file. The last local state drops the state outside the lock.
    scan_state: Option<Arc<FileScanState>>,
}

/// The state of an open file that the threads scanning it share.
pub struct FileScanState {
    /// The file as a Scan API source on the executor the scan runs on. After opening the file the
    /// reader uses nothing else of it.
    source: DataSourceRef,
    /// Whether the file's scan is split for DuckDB's threads.
    split: bool,
    /// The opened pieces of the file's scan, each of which threads scanning the file read streams
    /// of. Set when the file is initialized.
    pieces: OnceLock<Vec<ScanExec>>,
    /// The piece the next thread to join the file starts looking for work at.
    next_piece: AtomicUsize,
    cache: ConversionCache,
}

impl FileScanState {
    /// The scan work a thread joining the file would take, or `None` once the file has none.
    fn next_split(&self) -> Option<Split> {
        // A thread joins the file until its last unit of work, a V1 split or a V2 part, is taken,
        // then moves on to the next file while the threads holding the last work finish it: no
        // thread waits for the rest of a file to drain. Threads joining the file start at
        // successive pieces, so they spread over its rows. The thread creates its stream on its
        // first scan call, outside DuckDB's lock.
        let pieces = self.pieces.get()?;
        let start = self.next_piece.fetch_add(1, Ordering::Relaxed);
        (0..pieces.len())
            .map(|offset| (start + offset) % pieces.len())
            .find(|&piece| !pieces[piece].is_exhausted())
            .map(|piece| Split {
                piece,
                stream: None,
            })
    }
}

impl OpenFileReader {
    async fn open(path: String) -> VortexResult<Self> {
        let (fs, fs_path) = resolve_filesystem(&parse_uri_or_path(&path)?)?;
        let source = fs.open_read(&fs_path).await?;
        let file = open_cached(&SESSION, Some(&path), source, None, &|options| options).await?;
        // The engine is chosen here, once per file. Everything after reads the file through the
        // Scan API, the same way for either executor.
        let (source, split) = if crate::scan_v2() {
            (file.plan_data_source(), true)
        } else {
            // DuckDB runs one thread per core, so its threads are the scan's parallelism: each
            // runs one split at a time, polling it itself. A stream with room for more would claim
            // splits other threads could have run, leaving them nothing on a file with few splits.
            let options = LayoutScanOptions::default().with_concurrency(NonZeroUsize::MIN);
            let parts = crate::v1_split_parts();
            let options = match parts {
                true => options.with_split_by(SplitBy::Parts { min_parts: 1 }),
                false => options,
            };
            (file.layout_reader_data_source(options)?, parts)
        };
        Ok(OpenFileReader {
            scan_state: Some(Arc::new(FileScanState {
                source,
                split,
                pieces: OnceLock::new(),
                next_piece: AtomicUsize::new(0),
                cache: ConversionCache::default(),
            })),
        })
    }

    fn scan_state(&self) -> &FileScanState {
        self.scan_state
            .as_ref()
            .vortex_expect("file used after its scan state was released")
    }

    fn source(&self) -> &DataSourceRef {
        &self.scan_state().source
    }

    /// The file's rows, which a Vortex file's source knows exactly.
    pub(crate) fn row_count(&self) -> VortexResult<u64> {
        self.source()
            .row_count()
            .as_exact()
            .ok_or_else(|| vortex_err!("a Vortex file source must know its exact row count"))
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
    let dtype = file.source().dtype().clone();
    let columns = extract_schema_from_dtype(&dtype)?;

    for column in &columns {
        result.add_result_column(&column.name, &column.logical_type);
    }

    Ok(BindState {
        dtype,
        first_file_row_count: file.row_count()?,
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
    // Opening the file's scan is non-trivial work so we prefer doing it here under file lock and
    // not in reader_try_initialize_scan under global lock.
    let state = file.scan_state();
    let filter = &global.filter;
    let request = ScanRequest {
        projection: global.projection_expr.clone(),
        filter: filter.expr.clone(),
        row_range: filter.row_range.clone(),
        selection: filter.row_selection.clone(),
        ordered: false,
        bind_cache: Some(global.bind_cache.clone()),
        ..ScanRequest::default()
    };
    // An unordered scan without a limit opens onto a queue of the file's work, whose streams each
    // take the next once they have finished their last. A scan cut into parts, as every V2 scan
    // is, is first split into its share of DuckDB's threads, so that a scan of fewer parts than
    // threads cuts its files into smaller parts rather than leaving threads idle. A V1 scan's work
    // is otherwise its layout splits, which splitting would only regroup. A scan, or a piece, the
    // file's statistics reject opens exhausted.
    let scan = RUNTIME.block_on(state.source.scan(request))?;
    let pieces = if state.split {
        scan.split(global.file_pieces)?
    } else {
        vec![scan]
    };
    let pieces = pieces
        .into_iter()
        .map(|piece| piece.open())
        .filter(|exec| !exec.as_ref().is_ok_and(ScanExec::is_exhausted))
        .collect::<VortexResult<Vec<_>>>()?;
    if pieces.is_empty() {
        // DuckDB releases a skipped file under the global lock. Here only the file-local lock is
        // held, so drop the scan state now.
        file.scan_state = None;
        return Ok(true);
    }
    if state.pieces.set(pieces).is_err() {
        vortex_panic!("file initialized twice");
    }
    Ok(false)
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
    let Some(Split { piece, stream }) = local.split.take() else {
        return Ok(SplitItem::Done);
    };
    let file = local.file.as_ref().vortex_expect("split without a file");
    let exec = &file
        .pieces
        .get()
        .vortex_expect("split of a file without an opened scan")[piece];
    let mut stream = stream.unwrap_or_else(|| exec.stream());
    let Some(array) = RUNTIME.block_on(stream.next()).transpose()? else {
        // Dropping the last stream drops the piece's queue of work; do that outside DuckDB's
        // locks.
        local.retired_splits.push(Split {
            piece,
            stream: Some(stream),
        });
        return Ok(SplitItem::Done);
    };
    local.split = Some(Split {
        piece,
        stream: Some(stream),
    });
    Ok(if array.is_empty() {
        SplitItem::Empty
    } else {
        SplitItem::Array(array)
    })
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
    // A thread joins a file while its scan has work left to take. The check is
    // one load of a flag that is written once, so it is cheap under the global
    // lock.
    let split = file
        .scan_state
        .as_ref()
        .and_then(|state| state.next_split());
    let Some(split) = split else {
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

    local.retired_splits.extend(local.split.replace(split));
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
    // No lock is held here, so drop the splits and scan states this thread left.
    local.retired_splits.clear();
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

    let source = file.source();
    let dtype = source.dtype().as_struct_fields_opt()?.field(column)?;
    // A file source reads its statistics from the footer it has already read, so this does no I/O.
    let stats = RUNTIME
        .block_on(source.field_statistics(&FieldPath::from_name(column)))
        .ok()
        .filter(|stats| !stats.is_empty())?;

    let stats = ColumnStatisticsAggregate::new(&stats);
    match ColumnStatistics::try_from(stats, dtype) {
        Ok(stats) => Some(stats),
        Err(e) => vortex_panic!(e),
    }
}

/// Called from a separate thread (not related to threads for Vortex
/// table function) under global lock.
pub fn reader_get_progress_in_file(file: &OpenFileReader) -> f64 {
    // A file without a scan state was skipped or has every split taken.
    let Some(state) = file.scan_state.as_ref() else {
        return 100.0;
    };
    let Some(pieces) = state.pieces.get() else {
        return 0.0;
    };
    let (claimed, total) = pieces
        .iter()
        .filter_map(ScanExec::progress)
        .fold((0, 0), |(claimed, total), progress: Claims| {
            (claimed + progress.claimed, total + progress.total)
        });
    if total == 0 {
        100.0
    } else {
        100.0 * claimed as f64 / total as f64
    }
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
    let stats = ColumnStatisticsAggregate::new(&stats.to_stats_set());
    match ColumnStatistics::try_from(stats, dtype.clone()) {
        Ok(stats) => Some(stats),
        Err(e) => vortex_panic!(e),
    }
}
