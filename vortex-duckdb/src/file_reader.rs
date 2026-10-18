// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::SystemTime;

use futures::FutureExt;
use object_store::registry::ObjectStoreRegistry;
use parking_lot::Mutex;
use url::Url;
use vortex::aggregate_fn::AggregateFnVTableExt as _;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::min::Min;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::dtype::FieldPath;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_panic;
use vortex::expr::BoundExpression;
use vortex::expr::bound_and_collect;
use vortex::file::Footer;
use vortex::file::VortexFile;
use vortex::file::multi::MultiFileSession;
use vortex::file::multi::open_cached;
use vortex::file::multi::parse_uri_or_path;
use vortex::file::v2::FileStatsLayoutReader;
use vortex::io::VortexReadAt;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::io::std_file::MmapReadAt;
use vortex::io::std_file::StdFileSystem;
use vortex::layout::LayoutReader as _;
use vortex::layout::LayoutReaderRef;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::v2::ScanBuilder as V2ScanBuilder;
use vortex::layout::scan::v2::ScanFile;
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::session::SessionExt as _;
use vortex_utils::aliases::hash_map::HashMap;

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

/// Whether to read local files through memory maps of them, see [`enable_mmap_local_files`].
static MMAP_LOCAL_FILES: AtomicBool = AtomicBool::new(false);

/// Reads local files opened from now on through memory maps of them, which saves copying every
/// read out of the page cache. Local files are read with copies unless this is called.
///
/// # Safety
///
/// No local file scanned by this process may be truncated or modified during a scan. Accessing
/// the unmapped pages of a truncated file crashes the process with `SIGBUS`. Maps are reused by
/// later scans while a file keeps its length and modification time, so a file must also not be
/// modified between scans without changing either.
pub unsafe fn enable_mmap_local_files() {
    MMAP_LOCAL_FILES.store(true, Ordering::Relaxed);
}

/// A memory map of a local file, with the length and modification time it had when mapped, and
/// the layout reader and scan file of the file once a scan has opened it.
struct MappedFile {
    len: u64,
    modified: SystemTime,
    source: Arc<dyn VortexReadAt>,
    opened: OnceLock<(LayoutReaderRef, ScanFile)>,
}

/// The maps of the local files scanned so far, by path.
///
/// Each query opens the files it scans again. Reusing their maps saves unmapping a file after
/// every query and faulting its pages into a fresh map in the next one, and reusing their layout
/// readers saves decoding the footer and the zone maps of the file again. The maps and readers of
/// every file scanned stay cached for the life of the process.
static MAPPED_FILES: LazyLock<Mutex<HashMap<String, Arc<MappedFile>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// A memory map of the local file at `path`, reusing the map of an earlier scan while the file
/// keeps its length and modification time.
fn map_local_file(path: &str) -> VortexResult<Arc<MappedFile>> {
    let metadata = std::fs::metadata(path)?;
    let len = metadata.len();
    let modified = metadata.modified()?;
    let mut files = MAPPED_FILES.lock();
    if let Some(file) = files.get(path)
        && file.len == len
        && file.modified == modified
    {
        return Ok(Arc::clone(file));
    }
    // SAFETY: `enable_mmap_local_files` requires that scanned files do not change during a scan.
    let source: Arc<dyn VortexReadAt> = Arc::new(unsafe { MmapReadAt::open(path) }?);
    let file = Arc::new(MappedFile {
        len,
        modified,
        source,
        opened: OnceLock::new(),
    });
    files.insert(path.to_owned(), Arc::clone(&file));
    Ok(file)
}

fn resolve_filesystem(url: &Url) -> VortexResult<(FileSystemRef, String)> {
    // Compat makes us use tokio which is very bad for local reads on
    // high-core machines because reads go into blocking pool
    if url.scheme() == "file" {
        let fs = StdFileSystem::new(RUNTIME.handle());
        // SAFETY: `enable_mmap_local_files` requires that scanned files do not change during a
        // scan.
        let fs = if MMAP_LOCAL_FILES.load(Ordering::Relaxed) {
            unsafe { fs.with_mmap() }
        } else {
            fs
        };
        return Ok((Arc::new(fs), url.path().to_string()));
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
    /// Whether the threads scanning the file build its splits, see [`Split::Lazy`].
    lazy_splits: bool,
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
        let url = parse_uri_or_path(&path)?;
        // Reads of mapped files are cheap to issue late, so their splits are built by the
        // threads that scan them instead of all up front during reader initialization.
        let lazy_splits = url.scheme() == "file" && MMAP_LOCAL_FILES.load(Ordering::Relaxed);
        let open = |file: VortexFile| -> VortexResult<(LayoutReaderRef, ScanFile)> {
            let mut scan_file = file.scan_file();
            scan_file.cache.get_or_insert_default();
            Ok((file.layout_reader()?, scan_file))
        };
        let (reader, scan_file) = if lazy_splits {
            // Mapping a local file takes a few system calls, so it is done here instead of
            // handing it to a blocking thread, which takes longer than the calls themselves.
            let mapped = map_local_file(url.path())?;
            match mapped.opened.get() {
                Some(opened) => opened.clone(),
                None => {
                    let source = Arc::clone(&mapped.source);
                    let file = open_cached(&SESSION, Some(&path), source, None, &|options| options)
                        .await?;
                    let opened = open(file)?;
                    mapped.opened.get_or_init(|| opened).clone()
                }
            }
        } else {
            let (fs, fs_path) = resolve_filesystem(&url)?;
            let source = fs.open_read(&fs_path).await?;
            open(open_cached(&SESSION, Some(&path), source, None, &|options| options).await?)?
        };
        Ok(OpenFileReader {
            scan_state: Some(Arc::new(FileScanState {
                reader,
                scan_file,
                v2: OnceLock::new(),
                cache: ConversionCache::default(),
            })),
            splits: vec![],
            total_splits: 0,
            lazy_splits,
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

    /// The parts of `filter` that the file statistics do not prove redundant.
    ///
    /// DuckDB pushes the range of a hash join's build keys into the probe side scan as optional
    /// filters, which often span every value of a foreign key column. A conjunct the statistics
    /// prove true for every row keeps every row, and an optional one need only hold for rows that
    /// are not null, so such conjuncts are dropped instead of being evaluated row by row.
    fn unsatisfied_filter(&self, filter: &Filter) -> VortexResult<Option<BoundExpression>> {
        let Some(file_stats) = self
            .reader()
            .as_any()
            .downcast_ref::<FileStatsLayoutReader>()
        else {
            return Ok(filter.filter.clone());
        };
        let mut unsatisfied = Vec::new();
        for (part, optional) in [(&filter.required, false), (&filter.optional, true)] {
            let mut conjuncts = Vec::new();
            if let Some(part) = part {
                collect_conjuncts(part, &mut conjuncts);
            }
            let bounds: Vec<Option<ColumnBound>> = conjuncts
                .iter()
                .map(|conjunct| {
                    optional
                        .then(|| ColumnBound::try_new(file_stats, conjunct))
                        .flatten()
                })
                .collect();
            for (conjunct, bound) in conjuncts.into_iter().zip(&bounds) {
                let satisfied = if optional {
                    bound
                        .as_ref()
                        .is_some_and(|bound| keeps_most_values(bound, &bounds))
                        || file_stats.is_satisfied_by_non_null(&conjunct)?
                } else {
                    file_stats.is_satisfied(&conjunct)?
                };
                if !satisfied {
                    unsatisfied.push(conjunct);
                }
            }
        }
        Ok(bound_and_collect(unsatisfied))
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

/// An optional comparison that keeps at least this share of a column's range of values is not
/// worth evaluating row by row. DuckDB stops evaluating the range filters of a hash join's build
/// keys on its own tables when they keep this share of rows.
const MIN_KEPT_SHARE_TO_SKIP_OPTIONAL: f64 = 0.75;

/// A bound on a column from comparing it with a constant, with the column's range in the file.
struct ColumnBound {
    field: usize,
    lower: Option<f64>,
    upper: Option<f64>,
    min: f64,
    max: f64,
}

impl ColumnBound {
    /// The bound `conjunct` puts on a column, if it compares a column with a constant and the
    /// file has the column's minimum and maximum.
    fn try_new(file_stats: &FileStatsLayoutReader, conjunct: &BoundExpression) -> Option<Self> {
        let operator = conjunct.as_opt::<Binary>()?;
        let is_column = |expr: &BoundExpression| expr.is::<GetItem>() && expr.child(0).is_root();
        let (column, constant, operator) = if is_column(conjunct.child(0)) {
            (
                conjunct.child(0),
                conjunct.child(1).as_opt::<Literal>()?,
                *operator,
            )
        } else if is_column(conjunct.child(1)) {
            let flipped = match operator {
                Operator::Gt => Operator::Lt,
                Operator::Gte => Operator::Lte,
                Operator::Lt => Operator::Gt,
                Operator::Lte => Operator::Gte,
                _ => return None,
            };
            (
                conjunct.child(1),
                conjunct.child(0).as_opt::<Literal>()?,
                flipped,
            )
        } else {
            return None;
        };
        let fields = file_stats.dtype().as_struct_fields_opt()?;
        let name = column.as_::<GetItem>();
        let field = fields.find(name)?;
        let (stats, _) = file_stats
            .file_stats()
            .get_by_path(&FieldPath::from_name(name.clone()))?;
        let as_f64 = |value: Option<Scalar>| value?.as_primitive_opt()?.as_::<f64>();
        let min = as_f64(
            stats
                .get(&Min.bind(NumericalAggregateOpts::skip_nans()))
                .as_exact(),
        )?;
        let max = as_f64(
            stats
                .get(&Max.bind(NumericalAggregateOpts::skip_nans()))
                .as_exact(),
        )?;
        let value = constant.as_primitive_opt()?.as_::<f64>()?;
        let (lower, upper) = match operator {
            Operator::Gt | Operator::Gte => (Some(value), None),
            Operator::Lt | Operator::Lte => (None, Some(value)),
            _ => return None,
        };
        Some(Self {
            field,
            lower,
            upper,
            min,
            max,
        })
    }
}

/// Whether all the bounds on the column of `bound` together keep at least
/// [`MIN_KEPT_SHARE_TO_SKIP_OPTIONAL`] of its range. The bounds of one column are judged
/// together, as a narrow range of a sorted column is worth its pruning even when each bound alone
/// keeps most values.
fn keeps_most_values(bound: &ColumnBound, bounds: &[Option<ColumnBound>]) -> bool {
    let column_bounds = bounds
        .iter()
        .flatten()
        .filter(|other| other.field == bound.field);
    let lower = column_bounds
        .clone()
        .filter_map(|other| other.lower)
        .fold(bound.min, f64::max);
    let upper = column_bounds
        .filter_map(|other| other.upper)
        .fold(bound.max, f64::min);
    kept_share(lower, upper, bound.min, bound.max)
        .is_some_and(|kept| kept >= MIN_KEPT_SHARE_TO_SKIP_OPTIONAL)
}

/// The share of the range from `min` to `max` that lies between `lower` and `upper`.
fn kept_share(lower: f64, upper: f64, min: f64, max: f64) -> Option<f64> {
    (max > min).then(|| ((upper.min(max) - lower.max(min)) / (max - min)).clamp(0.0, 1.0))
}

/// Appends the operands of the AND expressions at the top of `expr` to `out`.
fn collect_conjuncts(expr: &BoundExpression, out: &mut Vec<BoundExpression>) {
    if expr
        .as_opt::<Binary>()
        .is_some_and(|operator| *operator == Operator::And)
    {
        for child in expr.children() {
            collect_conjuncts(child, out);
        }
    } else {
        out.push(expr.clone());
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
        let scan = FileScan::try_new(v2_builder(state, global))?;
        if state.v2.set(scan).is_err() {
            vortex_panic!("file initialized twice");
        }
        return Ok(false);
    }

    let reader = Arc::clone(file.reader());
    let filter = &global.filter;
    let builder = ScanBuilder::new(SESSION.clone(), reader)
        .with_projection(global.projection.clone())
        .with_some_filter(file.unsatisfied_filter(filter)?)
        .with_selection(filter.row_selection.clone());
    let scan = builder.prepare()?;
    let mut splits: Vec<Split> = if file.lazy_splits {
        scan.execute_lazy(filter.row_range.clone())?
            .into_iter()
            .map(Split::Lazy)
            .collect()
    } else {
        scan.execute(filter.row_range.clone())?
            .into_iter()
            .map(Split::Started)
            .collect()
    };

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
        Some(Split::Started(task)) => Ok(RUNTIME
            .block_on(task)?
            .map_or(SplitItem::Empty, SplitItem::Array)),
        Some(Split::Lazy(split)) => Ok(RUNTIME
            .block_on(split.start()?)?
            .map_or(SplitItem::Empty, SplitItem::Array)),
        Some(Split::V2) => {
            let file = local.file.as_ref().vortex_expect("split without a file");
            let scan = file
                .v2
                .get()
                .vortex_expect("V2 split of a file without a V2 scan");
            let Some(array) = scan.next_array(&mut local.run)? else {
                return Ok(SplitItem::Done);
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

    let path = FieldPath::from_name(column);
    let (stats, dtype) = reader.file_stats().get_by_path(&path)?;

    let stats = ColumnStatisticsAggregate::new(
        stats,
        reader.file_stats().distinct_count_by_path(&path),
        reader.file_stats().max_length_by_path(&path),
    );
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
    let file_stats = footer.statistics()?;
    let path = FieldPath::from_name(column);
    let (stats, dtype) = file_stats.get_by_path(&path)?;
    let stats = ColumnStatisticsAggregate::new(
        stats,
        file_stats.distinct_count_by_path(&path),
        file_stats.max_length_by_path(&path),
    );
    match ColumnStatistics::try_from(stats, dtype.clone()) {
        Ok(stats) => Some(stats),
        Err(e) => vortex_panic!(e),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::kept_share;

    #[rstest]
    #[case(37.0, 2_399_788.0, Some(1.0))]
    #[case(1_172_551.0, 1_319_947.0, Some(0.061))]
    #[case(1_200_000.0, f64::INFINITY, Some(0.5))]
    #[case(f64::NEG_INFINITY, -5.0, Some(0.0))]
    fn share_of_range_kept(#[case] lower: f64, #[case] upper: f64, #[case] expected: Option<f64>) {
        let kept = kept_share(lower, upper, 0.0, 2_400_000.0);
        match (kept, expected) {
            (Some(kept), Some(expected)) => assert!((kept - expected).abs() < 1e-3, "{kept}"),
            (kept, expected) => assert_eq!(kept, expected),
        }
    }

    #[test]
    fn share_of_empty_range_is_unknown() {
        assert_eq!(kept_share(0.0, 1.0, 5.0, 5.0), None);
    }
}
