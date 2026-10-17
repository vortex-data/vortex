// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_panic;
use vortex::expr::BoundExpression;
use vortex::expr::bound_and_collect;
use vortex::expr::stats::Stat;
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
use vortex::layout::LayoutReader as _;
use vortex::layout::LayoutReaderRef;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
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
// `can_get_partition_stats` -> `footer_open` -> `footer_get_statistics`.
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
/// the unmapped pages of a truncated file crashes the process with `SIGBUS`.
pub unsafe fn enable_mmap_local_files() {
    MMAP_LOCAL_FILES.store(true, Ordering::Relaxed);
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
}

/// The state of an open file that the threads scanning it share.
pub struct FileScanState {
    reader: LayoutReaderRef,
    cache: ConversionCache,
}

impl OpenFileReader {
    async fn open(path: String) -> VortexResult<Self> {
        let (fs, fs_path) = resolve_filesystem(&parse_uri_or_path(&path)?)?;
        let source = fs.open_read(&fs_path).await?;
        let file = open_cached(&SESSION, Some(&path), source, None, &|options| options).await?;
        Ok(OpenFileReader {
            scan_state: Some(Arc::new(FileScanState {
                reader: file.layout_reader()?,
                cache: ConversionCache::default(),
            })),
            splits: vec![],
            total_splits: 0,
        })
    }

    fn reader(&self) -> &LayoutReaderRef {
        &self
            .scan_state
            .as_ref()
            .vortex_expect("file used after its scan state was released")
            .reader
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
            for conjunct in conjuncts {
                let satisfied = if optional {
                    (keeps_most_values(file_stats, &conjunct) && !self.prunes_zones(&conjunct)?)
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

    /// Whether the zone maps of the file let `conjunct` skip a noticeable share of its rows, so
    /// that it is worth keeping for pruning even when evaluating it on rows is not.
    fn prunes_zones(&self, conjunct: &BoundExpression) -> VortexResult<bool> {
        let row_count = self.reader().row_count();
        let len = usize::try_from(row_count).unwrap_or(usize::MAX);
        let evaluation =
            self.reader()
                .pruning_evaluation(&(0..row_count), conjunct, Mask::new_true(len))?;
        let mask = RUNTIME.block_on(evaluation)?;
        Ok(mask.true_count() < len - len / MIN_PRUNED_DIVISOR)
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

/// Whether `conjunct` compares a column with a constant such that, judging by the column's
/// minimum and maximum in the file, it keeps most of the column's values.
fn keeps_most_values(file_stats: &FileStatsLayoutReader, conjunct: &BoundExpression) -> bool {
    let Some(operator) = conjunct.as_opt::<Binary>() else {
        return false;
    };
    let is_column = |expr: &BoundExpression| expr.is::<GetItem>() && expr.child(0).is_root();
    let (column, constant, operator) = if is_column(conjunct.child(0))
        && let Some(constant) = conjunct.child(1).as_opt::<Literal>()
    {
        (conjunct.child(0), constant, *operator)
    } else if is_column(conjunct.child(1))
        && let Some(constant) = conjunct.child(0).as_opt::<Literal>()
    {
        let flipped = match operator {
            Operator::Gt => Operator::Lt,
            Operator::Gte => Operator::Lte,
            Operator::Lt => Operator::Gt,
            Operator::Lte => Operator::Gte,
            _ => return false,
        };
        (conjunct.child(1), constant, flipped)
    } else {
        return false;
    };
    let field = column.as_::<GetItem>();
    let Some(fields) = file_stats.dtype().as_struct_fields_opt() else {
        return false;
    };
    let Some(index) = fields.find(field) else {
        return false;
    };
    let (Some(stats), Some(dtype)) = (
        file_stats.file_stats().stats_sets().get(index),
        fields.field_by_index(index),
    ) else {
        return false;
    };
    let as_f64 = |value: Option<ScalarValue>| {
        Scalar::try_new(dtype.clone(), value)
            .ok()?
            .as_primitive_opt()?
            .as_::<f64>()
    };
    let (Some(min), Some(max), Some(value)) = (
        as_f64(stats.get(Stat::Min).as_exact()),
        as_f64(stats.get(Stat::Max).as_exact()),
        constant
            .as_primitive_opt()
            .and_then(|value| value.as_::<f64>()),
    ) else {
        return false;
    };
    kept_share(operator, value, min, max)
        .is_some_and(|kept| kept >= MIN_KEPT_SHARE_TO_SKIP_OPTIONAL)
}

/// The share of the range from `min` to `max` that `operator` against `value` keeps.
fn kept_share(operator: Operator, value: f64, min: f64, max: f64) -> Option<f64> {
    if max <= min {
        return None;
    }
    let kept = match operator {
        Operator::Gt | Operator::Gte => (max - value) / (max - min),
        Operator::Lt | Operator::Lte => (value - min) / (max - min),
        _ => return None,
    };
    Some(kept.clamp(0.0, 1.0))
}

/// An optional comparison whose zone maps skip at least 1/20 of a file's rows is kept for
/// pruning.
const MIN_PRUNED_DIVISOR: usize = 20;

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

    // Getting splits is non-trivial work so we prefer doing it here under file
    // lock and not in reader_try_initialize_scan under global lock.
    let reader = Arc::clone(file.reader());
    let filter = &global.filter;
    let builder = ScanBuilder::new(SESSION.clone(), reader)
        .with_projection(global.projection.clone())
        .with_some_filter(file.unsatisfied_filter(filter)?)
        .with_selection(filter.row_selection.clone());
    let scan = builder.prepare()?;
    let mut splits = scan.execute(filter.row_range.clone())?;

    // threads take last element of file.splits so we need to reverse
    splits.reverse();
    file.total_splits = splits.len();
    file.splits = splits;
    Ok(false)
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
    let Some(split) = file.splits.pop() else {
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
        let Some(split) = local.split.take() else {
            return Ok(false);
        };
        let Some(array) = RUNTIME.block_on(split)? else {
            // split is filtered
            return Ok(true);
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
    let Some(split) = local.split.take() else {
        return Ok(false);
    };
    let Some(array) = RUNTIME.block_on(split)? else {
        // split is filtered
        return Ok(true);
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

    let DType::Struct(fields, _) = &file.reader().dtype() else {
        return None;
    };
    let index = fields.find(column)?;
    let stats_sets = reader.file_stats().stats_sets();

    let dtype = fields.field_by_index(index)?;

    let stats = ColumnStatisticsAggregate::new(
        stats_sets.get(index)?,
        reader.file_stats().distinct_count(index),
    );
    match ColumnStatistics::try_from(stats, dtype) {
        Ok(stats) => Some(stats),
        Err(e) => vortex_panic!(e),
    }
}

/// Called from a separate thread (not related to threads for Vortex
/// table function) under global lock.
pub fn reader_get_progress_in_file(file: &OpenFileReader) -> f64 {
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
pub fn footer_get_statistics(footer: &Footer, index: usize) -> Option<ColumnStatistics> {
    let DType::Struct(fields, _) = footer.dtype() else {
        return None;
    };
    let file_stats = footer.statistics()?;
    let dtype = fields.field_by_index(index)?;
    let stats = ColumnStatisticsAggregate::new(
        file_stats.stats_sets().get(index)?,
        file_stats.distinct_count(index),
    );
    match ColumnStatistics::try_from(stats, dtype) {
        Ok(stats) => Some(stats),
        Err(e) => vortex_panic!(e),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::scalar_fn::fns::operators::Operator;

    use super::kept_share;

    #[rstest]
    #[case(Operator::Gte, 37.0, Some(1.0))]
    #[case(Operator::Lte, 2_399_788.0, Some(0.999_9))]
    #[case(Operator::Gte, 1_200_000.0, Some(0.5))]
    #[case(Operator::Lt, -5.0, Some(0.0))]
    #[case(Operator::Eq, 5.0, None)]
    fn share_of_range_kept(
        #[case] operator: Operator,
        #[case] value: f64,
        #[case] expected: Option<f64>,
    ) {
        let kept = kept_share(operator, value, 0.0, 2_400_000.0);
        match (kept, expected) {
            (Some(kept), Some(expected)) => assert!((kept - expected).abs() < 1e-3, "{kept}"),
            (kept, expected) => assert_eq!(kept, expected),
        }
    }
}
