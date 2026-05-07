// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex implementation of [`MultiFileFunction`].
//!
//! Plugs Vortex into DuckDB's `MultiFileFunction<OP>` template: cross-file
//! orchestration (globbing, parallelism, virtual columns, hive partitioning)
//! is handled by DuckDB. This module supplies the per-file reader using
//! [`VortexFile`] directly so file-level statistics, dtype, and pruning are
//! available without going through `MultiLayoutDataSource`.

use std::collections::VecDeque;
use std::ffi::CStr;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use itertools::Itertools;
use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ScalarFn;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::VortexExprExt;
use vortex::expr::and_collect;
use vortex::expr::cast;
use vortex::expr::col;
use vortex::expr::merge;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::Footer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::layout::layouts::row_idx::row_idx;
use vortex::layout::scan::split_by::SplitBy;
use vortex::scalar_fn::fns::pack::Pack;
use vortex::scan::selection::Selection;

use crate::RUNTIME;
use crate::SESSION;
use crate::convert::try_from_bound_expression;
use crate::convert::try_from_table_filter;
use crate::convert::try_from_virtual_column_filter;
use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb::BaseFileReader;
use crate::duckdb::Cardinality;
use crate::duckdb::ClientContextRef;
use crate::duckdb::ColumnStatistics;
use crate::duckdb::DataChunkRef;
use crate::duckdb::DuckdbStringMapRef;
use crate::duckdb::ExpressionClass;
use crate::duckdb::ExpressionRef;
use crate::duckdb::ExtractedValue;
use crate::duckdb::LogicalType;
use crate::duckdb::MultiFileFunction;
use crate::duckdb::PartitionStats;
use crate::duckdb::ProjectedColumn;
use crate::duckdb::SchemaBuilder;
use crate::duckdb::TableFilterSetRef;
use crate::duckdb::duckdb_vector_size;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::filesystem::resolve_filesystem;
use crate::multi_file::parse_glob_url;

type ScanTask = Task<VortexResult<Option<ArrayRef>>>;

const VORTEX_METADATA_CACHE_SETTING: &CStr = c"vortex_metadata_cache";
const VORTEX_FOOTER_CACHE_TYPE: &CStr = c"vortex_footer";
const DEFAULT_FOOTER_CACHE_BYTES: usize = 10 * 1024;
const FILE_ROW_NUMBER_COLUMN_ID: u64 = 9223372036854775809;
const FILE_INDEX_COLUMN_ID: u64 = 9223372036854775810;
// DuckDB drains one multi-file reader at a time, so raw layout-level splits can
// create thousands of tiny scan assignments for ClickBench-sized shards.
const MULTI_FILE_SCAN_MIN_SPLIT_ROWS: usize = 131_072;
const MULTI_FILE_SCAN_MAX_SPLIT_ROWS: usize = MULTI_FILE_SCAN_MIN_SPLIT_ROWS * 2;

/// Open a [`VortexFile`] using whichever filesystem the user has configured
/// via the `vortex_filesystem` extension option. DuckDB has already expanded
/// any glob and chosen this exact path; we only need the right reader for
/// the URL scheme. Routing through the filesystem also lets HTTP/S3/etc.
/// transparently use DuckDB's `httpfs` when the user picks `'duckdb'`.
fn open_vortex_file(ctx: &ClientContextRef, path: &str) -> VortexResult<VortexFile> {
    let metadata_cache_enabled = vortex_metadata_cache_enabled(ctx);
    let cached_footer = if metadata_cache_enabled {
        // SAFETY: this module is the only writer for `vortex_footer` entries,
        // and it stores exactly `Footer` values for this object type.
        unsafe { ctx.object_cache_get_cloned::<Footer>(path, VORTEX_FOOTER_CACHE_TYPE) }?
    } else {
        None
    };
    let cache_miss = metadata_cache_enabled && cached_footer.is_none();

    let url = parse_glob_url(path)?;
    let mut base_url = url.clone();
    base_url.set_path("");
    let fs = resolve_filesystem(&base_url, ctx)?;
    let mut options = SESSION.open_options();
    if let Some(footer) = cached_footer {
        options = options.with_footer(footer);
    }

    let file = RUNTIME.block_on(async move {
        let reader = fs.open_read(url.path()).await?;
        options.open(reader).await
    })?;

    if cache_miss {
        ctx.object_cache_put(
            path,
            VORTEX_FOOTER_CACHE_TYPE,
            footer_cache_memory(file.footer()),
            file.footer().clone(),
        )?;
    }

    Ok(file)
}

/// Multi-file Vortex scan registered via `MultiFileFunction<OP>`.
///
/// Compared to [`crate::multi_file::VortexMultiFileScan`] (the table-function
/// path), this delegates file globbing, virtual columns, and hive partitioning
/// to DuckDB's native machinery, and reads each file via [`VortexFile`].
#[derive(Debug)]
pub struct VortexMultiFileFunction;

#[derive(Default)]
pub struct VortexReaderOptions;

/// Bind-time data shared across all per-file readers in a query.
#[derive(Clone, Default)]
pub struct VortexBindData {
    /// Metadata and open handle for the file DuckDB selected for binding.
    first_file: Option<BoundFirstFile>,
    /// Exact complex filters pushed at optimizer time. These are copied into
    /// every per-file reader before scan planning.
    complex_filter_exprs: Vec<Expression>,
}

#[derive(Clone)]
struct BoundFirstFile {
    path: String,
    file: VortexFile,
    column_dtypes: Vec<(String, DType)>,
}

#[derive(Debug)]
pub struct VortexGlobal;

#[derive(Default)]
pub struct VortexLocal {
    /// Row range claimed under DuckDB's multi-file scheduling lock.
    row_range: Option<Range<u64>>,
    /// Remaining metadata-only rows for zero-projection, no-filter scans.
    remaining_rows: u64,
    /// Split task claimed under DuckDB's multi-file scheduling lock.
    task: Option<ScanTask>,
    /// Export batches being drained for this local scan assignment.
    exporters: VecDeque<ArrayExporter>,
}

/// Per-file scan state. Holds the open [`VortexFile`] plus immutable scan
/// configuration shared by DuckDB workers. The scan is built once after
/// projection/filter preparation into a queue of split tasks; workers then
/// claim those tasks into [`VortexLocal`] under DuckDB's scheduling lock.
pub struct VortexFileReader {
    file: VortexFile,
    file_idx: usize,
    cache: ConversionCache,
    /// Projection set by [`Self::prepare_reader`]. `None` means prepare was
    /// never called (defensive — scan all columns). `Some(empty)` is the
    /// explicit zero-projection case (e.g. `SELECT count(*)`); the scan
    /// produces struct arrays with no fields, and `ArrayExporter` short-
    /// circuits on the empty fields list.
    projection: Option<Vec<FieldName>>,
    /// Positions in DuckDB's intermediate scan chunk for fields materialized by
    /// the Vortex projection.
    field_positions: Vec<usize>,
    /// Number of columns DuckDB allocated in the intermediate scan chunk.
    scan_column_count: usize,
    /// Position of DuckDB's file_index virtual column in the scan chunk, if
    /// DuckDB asks the reader to materialize it instead of filling it as a
    /// per-file constant.
    file_index_column_pos: Option<usize>,
    /// Position of DuckDB's file_row_number virtual column in the scan chunk.
    file_row_number_column_pos: Option<usize>,
    /// File-relative row indices selected by a pushed file_row_number filter.
    row_selection: Selection,
    /// File-relative row range selected by a pushed file_row_number filter.
    row_range: Option<Range<u64>>,
    /// Filter expression set by [`Self::prepare_reader`]. None when no filters
    /// were pushed down or when conversion failed.
    filter: Option<Expression>,
    /// Complex filters accepted at bind/optimizer time and applied by every
    /// per-file scan.
    complex_filter_exprs: Vec<Expression>,
    /// Set when a filter has been pushed down and file-level statistics prove
    /// the file can be skipped. Causes [`Self::try_initialize_scan`] to return
    /// false without opening a scan iterator.
    file_pruned: bool,
    /// Split tasks prepared from one scan builder. DuckDB serializes
    /// TryInitializeScan with its global lock, but the reader itself is shared,
    /// so the queue still needs interior mutability on the Rust side.
    tasks: Mutex<VecDeque<ScanTask>>,
    /// Whether the single metadata-only assignment has been claimed.
    metadata_only_claimed: AtomicBool,
    /// Total rows in the file, cached for [`Self::progress_in_file`].
    total_rows: u64,
    /// Rows produced so far. Bumped after each chunk in [`Self::scan`].
    rows_scanned: AtomicU64,
}

impl Debug for VortexFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexFileReader")
            .field("file_idx", &self.file_idx)
            .field("row_count", &self.file.row_count())
            .field("remaining_tasks", &self.tasks.lock().len())
            .finish_non_exhaustive()
    }
}

impl MultiFileFunction for VortexMultiFileFunction {
    type ReaderOptions = VortexReaderOptions;
    type BindData = VortexBindData;
    type GlobalState = VortexGlobal;
    type LocalState = VortexLocal;
    type Reader = VortexFileReader;

    const FILTER_PUSHDOWN: bool = true;
    const FILTER_PRUNE: bool = true;

    fn create_options(_ctx: &ClientContextRef) -> VortexResult<Self::ReaderOptions> {
        Ok(VortexReaderOptions)
    }

    fn initialize_bind_data(_options: Self::ReaderOptions) -> VortexResult<Self::BindData> {
        Ok(VortexBindData::default())
    }

    fn pushdown_complex_filter(
        bind_data: &mut Self::BindData,
        expr: &ExpressionRef,
    ) -> VortexResult<bool> {
        if !contains_string_filter(expr) {
            return Ok(false);
        }
        let Some(expr) = try_from_bound_expression(expr)? else {
            return Ok(false);
        };
        bind_data.complex_filter_exprs.push(expr);
        Ok(true)
    }

    fn bind_reader(
        ctx: &ClientContextRef,
        bind_data: &mut Self::BindData,
        first_file: &str,
        schema: &mut SchemaBuilder,
    ) -> VortexResult<()> {
        // Open the first file (using whichever filesystem the user picked via
        // the `vortex_filesystem` extension option) to discover the schema.
        let file = open_vortex_file(ctx, first_file)?;
        let dtype = file.dtype();
        let fields = dtype.as_struct_fields_opt().ok_or_else(|| {
            vortex_err!("Vortex file must contain a struct array at the top level")
        })?;
        for (name, field_dtype) in fields.names().iter().zip(fields.fields()) {
            let logical_type = LogicalType::try_from(&field_dtype)?;
            schema.add_column(name.as_ref(), &logical_type);
        }
        bind_data.first_file = Some(BoundFirstFile {
            path: first_file.to_string(),
            column_dtypes: column_dtypes(&file),
            file,
        });
        Ok(())
    }

    fn init_global(
        _ctx: &ClientContextRef,
        _bind_data: &Self::BindData,
    ) -> VortexResult<Self::GlobalState> {
        Ok(VortexGlobal)
    }

    fn init_local(_global: &Self::GlobalState) -> Self::LocalState {
        VortexLocal::default()
    }

    fn create_reader(
        ctx: &ClientContextRef,
        _global: &Self::GlobalState,
        bind_data: &Self::BindData,
        file_path: &str,
        file_idx: usize,
    ) -> VortexResult<Self::Reader> {
        let (file, _column_dtypes) = open_or_reuse_vortex_file(ctx, bind_data, file_path)?;

        let total_rows = file.row_count();
        Ok(VortexFileReader {
            file,
            file_idx,
            cache: ConversionCache {
                file_index: file_idx,
                ..Default::default()
            },
            projection: None,
            field_positions: vec![],
            scan_column_count: 0,
            file_index_column_pos: None,
            file_row_number_column_pos: None,
            row_selection: Selection::All,
            row_range: None,
            filter: None,
            complex_filter_exprs: bind_data.complex_filter_exprs.clone(),
            file_pruned: false,
            tasks: Mutex::new(VecDeque::new()),
            metadata_only_claimed: AtomicBool::new(false),
            total_rows,
            rows_scanned: AtomicU64::new(0),
        })
    }

    fn cardinality(bind_data: &Self::BindData, file_count: usize) -> Cardinality {
        if !bind_data.complex_filter_exprs.is_empty() {
            return Cardinality::Unknown;
        }
        let first_file_row_count = bind_data
            .first_file
            .as_ref()
            .map(|first| first.file.row_count());
        match (file_count, first_file_row_count) {
            (0, _) | (_, None) => Cardinality::Unknown,
            (1, Some(row_count)) => Cardinality::Maximum(row_count),
            (file_count, Some(rows_per_file)) => {
                Cardinality::Estimate(rows_per_file.saturating_mul(file_count as u64))
            }
        }
    }

    fn partition_stats(
        ctx: &ClientContextRef,
        bind_data: &Self::BindData,
        file_path: &str,
    ) -> VortexResult<Option<PartitionStats>> {
        if !bind_data.complex_filter_exprs.is_empty() {
            return Ok(None);
        }
        if let Some(first_file) = bind_data
            .first_file
            .as_ref()
            .filter(|first| first.path == file_path)
        {
            return Ok(Some(PartitionStats {
                row_count: first_file.file.row_count(),
            }));
        }
        if !vortex_metadata_cache_enabled(ctx) {
            return Ok(None);
        }

        // SAFETY: this module is the only writer for `vortex_footer` entries,
        // and it stores exactly `Footer` values for this object type.
        let Some(footer) =
            unsafe { ctx.object_cache_get_cloned::<Footer>(file_path, VORTEX_FOOTER_CACHE_TYPE) }?
        else {
            return Ok(None);
        };
        Ok(Some(PartitionStats {
            row_count: footer.row_count(),
        }))
    }

    fn to_string(bind_data: &Self::BindData, map: &mut DuckdbStringMapRef) {
        map.push("Function", "Vortex Multi-File Scan");
        if !bind_data.complex_filter_exprs.is_empty() {
            let mut filters = bind_data
                .complex_filter_exprs
                .iter()
                .map(|f| format!("{}", f));
            map.push("Filters", &filters.join(" /\\\n"));
        }
    }
}

impl BaseFileReader<VortexGlobal, VortexLocal> for VortexFileReader {
    fn prepare_reader(
        &mut self,
        projection: &[ProjectedColumn<'_>],
        filters: Option<&TableFilterSetRef>,
    ) -> VortexResult<()> {
        // Capture the physical projection in scan chunk order, excluding
        // DuckDB virtual columns that Vortex must synthesize separately.
        // `Some(empty)` is the explicit zero-projection case (e.g. SELECT
        // count(*)) and is handled when the scan starts.
        let mut proj = Vec::new();
        let mut physical_field_positions = Vec::new();
        let mut physical_by_projection = Vec::with_capacity(projection.len());
        let mut file_index_column_pos = None;
        let mut file_row_number_column_pos = None;
        for (column_pos, column) in projection.iter().enumerate() {
            if column.is_virtual {
                if column.is_projected {
                    match column.column_id {
                        FILE_INDEX_COLUMN_ID => file_index_column_pos = Some(column_pos),
                        FILE_ROW_NUMBER_COLUMN_ID => file_row_number_column_pos = Some(column_pos),
                        _ => {}
                    }
                }
                physical_by_projection.push(None);
            } else {
                let name = FieldName::from(column.name);
                physical_by_projection.push(Some(name.clone()));
                if column.is_projected {
                    physical_field_positions.push(column_pos);
                    proj.push(name);
                }
            }
        }
        let mut field_positions = Vec::with_capacity(
            physical_field_positions.len() + file_row_number_column_pos.is_some() as usize,
        );
        if let Some(pos) = file_row_number_column_pos {
            field_positions.push(pos);
        }
        field_positions.extend(physical_field_positions);
        // Build a Vortex filter expression from DuckDB's table filters and
        // complex filters. DuckDB
        // keys multi-file filters by position in BaseFileReader::column_ids;
        // the C++ adapter passes that same ordered list here as `projection`.
        let mut pieces = self.complex_filter_exprs.clone();
        let mut row_selection = Selection::All;
        let mut row_range = None;
        if let Some(filters) = filters {
            let dtype = self.file.dtype();
            for (idx, filter) in filters.into_iter() {
                let idx = usize::try_from(idx)
                    .map_err(|_| vortex_err!("filter column index does not fit usize"))?;
                let Some(column) = projection.get(idx) else {
                    continue;
                };
                if column.is_virtual {
                    match column.column_id {
                        FILE_ROW_NUMBER_COLUMN_ID => {
                            let (selection, range) = try_from_virtual_column_filter(filter)?;
                            row_selection = selection;
                            row_range = range;
                        }
                        FILE_INDEX_COLUMN_ID => {
                            if !file_filter_matches(filter, self.file_idx)? {
                                self.file_pruned = true;
                            }
                        }
                        _ => {}
                    }
                    continue;
                }
                let Some(name) = physical_by_projection.get(idx).and_then(Option::as_ref) else {
                    continue;
                };
                if let Some(expr) = try_from_table_filter(filter, &col(name.as_ref()), dtype)? {
                    pieces.push(expr);
                }
            }
        }
        normalize_row_filter(&mut row_selection, &mut row_range);
        let filter = and_collect(pieces);
        if proj.is_empty()
            && file_row_number_column_pos.is_none()
            && let Some(filter) = &filter
        {
            proj = filter_field_projection(filter, self.file.dtype());
        }

        self.file_index_column_pos = file_index_column_pos;
        self.file_row_number_column_pos = file_row_number_column_pos;
        self.field_positions = field_positions;
        self.scan_column_count = projection.len();
        self.row_selection = row_selection;
        self.row_range = row_range;
        self.filter = filter;
        self.projection = Some(proj);

        // File-level pruning: if the filter combined with the file's stored
        // statistics proves no row can match, skip this file entirely.
        if !self.file_pruned
            && let Some(filter) = &self.filter
            && self.file.can_prune(filter)?
        {
            self.file_pruned = true;
        }
        self.metadata_only_claimed.store(false, Ordering::Release);
        let tasks = if self.file_pruned || self.metadata_only_count() {
            VecDeque::new()
        } else {
            self.build_scan_tasks()?
        };
        if !self.file_pruned && !self.metadata_only_count() && tasks.is_empty() {
            self.file_pruned = true;
        }
        self.tasks = Mutex::new(tasks);
        Ok(())
    }

    fn try_initialize_scan(
        &self,
        _global: &VortexGlobal,
        local: &mut VortexLocal,
    ) -> VortexResult<bool> {
        if self.file_pruned {
            return Ok(false);
        }
        local.remaining_rows = 0;
        local.task = None;
        local.exporters.clear();

        if self.metadata_only_count() {
            if self.metadata_only_claimed.swap(true, Ordering::AcqRel) {
                return Ok(false);
            }
            local.row_range = Some(0..self.total_rows);
            return Ok(true);
        }

        let Some(task) = self.tasks.lock().pop_front() else {
            return Ok(false);
        };

        local.row_range = None;
        local.task = Some(task);
        Ok(true)
    }

    fn prepare_scan(&self, _global: &VortexGlobal, local: &mut VortexLocal) -> VortexResult<()> {
        if self.metadata_only_count() {
            return Ok(());
        }
        let Some(task) = local.task.take() else {
            return Ok(());
        };

        let array = RUNTIME.block_on(task)?;
        if self.row_count_only() {
            local.remaining_rows = array.map(|array| array.len() as u64).unwrap_or(0);
            return Ok(());
        }

        local.exporters = array
            .map(|array| {
                make_exporters(
                    array,
                    &self.cache,
                    self.field_positions.clone(),
                    self.scan_column_count,
                )
            })
            .transpose()?
            .unwrap_or_default();
        Ok(())
    }

    fn scan(
        &self,
        _global: &VortexGlobal,
        local: &mut VortexLocal,
        chunk: &mut DataChunkRef,
    ) -> VortexResult<()> {
        if self.metadata_only_count() {
            return self.scan_metadata_only(local, chunk);
        }
        if self.row_count_only() {
            return self.scan_remaining_rows(local, chunk);
        }

        // Drain the in-flight split arrays if we have any.
        while let Some(exporter) = local.exporters.front_mut() {
            let has_more = exporter.export(
                chunk,
                self.file_index_column_pos,
                self.file_row_number_column_pos,
            )?;
            if has_more {
                if let Some(pos) = self.file_index_column_pos {
                    chunk
                        .get_vector_mut(pos)
                        .reference_value(&crate::duckdb::Value::from(self.file_idx as u64));
                }
                self.rows_scanned.fetch_add(chunk.len(), Ordering::Relaxed);
                return Ok(());
            }
            local.exporters.pop_front();
        }

        chunk.set_len(0);
        Ok(())
    }

    fn get_statistics(&self, name: &str) -> Option<ColumnStatistics> {
        let stats = self.file.file_stats()?;
        let (stats_set, dtype) = stats.get_by_name(self.file.dtype(), name)?;
        Some(make_column_statistics(stats_set, dtype))
    }

    fn progress_in_file(&self) -> f64 {
        if self.total_rows == 0 {
            return 100.0;
        }
        let rows_scanned = self.rows_scanned.load(Ordering::Relaxed);
        let pct = (rows_scanned as f64 / self.total_rows as f64) * 100.0;
        pct.clamp(0.0, 100.0)
    }
}

fn contains_string_filter(expr: &ExpressionRef) -> bool {
    match expr.as_class() {
        Some(ExpressionClass::BoundFunction(func)) => func.scalar_function.name() == "contains",
        Some(ExpressionClass::BoundOperator(op))
            if op.op == DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT =>
        {
            op.children().any(contains_string_filter)
        }
        Some(ExpressionClass::BoundConjunction(conj)) => {
            conj.children().any(contains_string_filter)
        }
        _ => false,
    }
}

fn file_filter_matches(
    filter: &crate::duckdb::TableFilterRef,
    file_idx: usize,
) -> VortexResult<bool> {
    let file_idx =
        u64::try_from(file_idx).map_err(|_| vortex_err!("file index does not fit u64"))?;
    let (selection, range) = try_from_virtual_column_filter(filter)?;
    let selection_matches = match selection {
        Selection::All => true,
        Selection::IncludeByIndex(indices) => indices.as_slice().binary_search(&file_idx).is_ok(),
        Selection::ExcludeByIndex(indices) => indices.as_slice().binary_search(&file_idx).is_err(),
        Selection::IncludeRoaring(indices) => indices.contains(file_idx),
        Selection::ExcludeRoaring(indices) => !indices.contains(file_idx),
    };
    let range_matches = range.as_ref().is_none_or(|range| range.contains(&file_idx));
    Ok(selection_matches && range_matches)
}

fn normalize_row_filter(selection: &mut Selection, range: &mut Option<Range<u64>>) {
    let Some(active_range) = range.clone() else {
        return;
    };
    let Selection::IncludeByIndex(indices) = selection else {
        return;
    };
    let filtered = indices
        .iter()
        .copied()
        .filter(|idx| active_range.contains(idx))
        .collect::<Buffer<u64>>();
    *selection = Selection::IncludeByIndex(filtered);
    *range = None;
}

fn filter_field_projection(filter: &Expression, dtype: &DType) -> Vec<FieldName> {
    let referenced = filter.field_references();
    dtype
        .as_struct_fields_opt()
        .map(|fields| {
            fields
                .names()
                .iter()
                .filter(|name| referenced.contains(*name))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

impl VortexFileReader {
    fn metadata_only_count(&self) -> bool {
        self.projection.as_ref().is_some_and(Vec::is_empty)
            && self.file_row_number_column_pos.is_none()
            && self.filter.is_none()
    }

    fn row_count_only(&self) -> bool {
        self.scan_column_count == 0
            && self.file_row_number_column_pos.is_none()
            && self.field_positions.is_empty()
            && self.filter.is_some()
            && self
                .projection
                .as_ref()
                .is_some_and(|proj| !proj.is_empty())
    }

    fn scan_remaining_rows(
        &self,
        local: &mut VortexLocal,
        chunk: &mut DataChunkRef,
    ) -> VortexResult<()> {
        if local.remaining_rows == 0 {
            chunk.set_len(0);
            return Ok(());
        }

        let chunk_len =
            duckdb_vector_size().min(usize::try_from(local.remaining_rows).unwrap_or(usize::MAX));
        chunk.reset();
        chunk.set_len(chunk_len);
        local.remaining_rows -= chunk_len as u64;
        self.rows_scanned
            .fetch_add(chunk_len as u64, Ordering::Relaxed);
        Ok(())
    }

    fn scan_metadata_only(
        &self,
        local: &mut VortexLocal,
        chunk: &mut DataChunkRef,
    ) -> VortexResult<()> {
        if local.remaining_rows == 0 {
            let Some(row_range) = local.row_range.take() else {
                chunk.set_len(0);
                return Ok(());
            };
            local.remaining_rows = row_range.end.saturating_sub(row_range.start);
        }

        if local.remaining_rows == 0 {
            chunk.set_len(0);
            return Ok(());
        }

        let chunk_len =
            duckdb_vector_size().min(usize::try_from(local.remaining_rows).unwrap_or(usize::MAX));
        chunk.reset();
        chunk.set_len(chunk_len);
        local.remaining_rows -= chunk_len as u64;
        self.rows_scanned
            .fetch_add(chunk_len as u64, Ordering::Relaxed);
        if let Some(pos) = self.file_index_column_pos {
            chunk
                .get_vector_mut(pos)
                .reference_value(&crate::duckdb::Value::from(self.file_idx as u64));
        }
        Ok(())
    }

    fn build_scan_tasks(&self) -> VortexResult<VecDeque<ScanTask>> {
        let mut builder = self
            .file
            .scan()?
            .with_split_by(SplitBy::layout_with_row_limits(
                Some(MULTI_FILE_SCAN_MIN_SPLIT_ROWS),
                Some(MULTI_FILE_SCAN_MAX_SPLIT_ROWS),
            ));
        // Apply projection. `None` (prepare not called) defaults to all
        // columns; `Some` (including the empty case for SELECT count(*))
        // applies an explicit `select` so the resulting struct arrays contain
        // exactly the columns DuckDB expects.
        if let Some(names) = &self.projection {
            let names = FieldNames::from_iter(names.iter().cloned());
            let select = select(names, root());
            let projection = if self.file_row_number_column_pos.is_some() {
                let row_idx = cast(row_idx(), DType::Primitive(PType::I64, false.into()));
                let row_idx_struct = pack([("file_row_number", row_idx)], false.into());
                merge([row_idx_struct, select])
            } else {
                select
            };
            builder = builder.with_projection(projection);
        }
        if let Some(row_range) = self.row_range.clone() {
            builder = builder.with_row_range(row_range);
        }
        builder = builder.with_selection(self.row_selection.clone());
        if let Some(filter) = self.filter.clone() {
            builder = builder.with_filter(filter);
        }
        let handle = RUNTIME.handle();
        Ok(builder
            .build()?
            .into_iter()
            .map(|task| handle.spawn(task))
            .collect())
    }
}

/// Convert the next array off the scan stream into a [`StructArray`] suitable
/// for [`ArrayExporter`].
fn make_exporter(
    array: ArrayRef,
    cache: &ConversionCache,
    field_positions: Vec<usize>,
    scan_column_count: usize,
) -> VortexResult<ArrayExporter> {
    let mut ctx = SESSION.create_execution_ctx();
    let struct_array: StructArray = if let Some(s) = array.as_opt::<Struct>() {
        s.into_owned()
    } else if let Some(array) = array.as_opt::<ScalarFn>()
        && let Some(pack_options) = array.scalar_fn().as_opt::<Pack>()
    {
        StructArray::new(
            pack_options.names.clone(),
            array.children(),
            array.len(),
            pack_options.nullability.into(),
        )
    } else {
        array.execute::<Canonical>(&mut ctx)?.into_struct()
    };
    ArrayExporter::try_new_with_positions(
        &struct_array,
        cache,
        ctx,
        field_positions,
        scan_column_count,
    )
}

fn make_exporters(
    array: ArrayRef,
    cache: &ConversionCache,
    field_positions: Vec<usize>,
    scan_column_count: usize,
) -> VortexResult<VecDeque<ArrayExporter>> {
    array
        .to_array_iterator()
        .map(|array| make_exporter(array?, cache, field_positions.clone(), scan_column_count))
        .collect()
}

fn column_dtypes(file: &VortexFile) -> Vec<(String, DType)> {
    file.dtype()
        .as_struct_fields_opt()
        .map(|fields| {
            fields
                .names()
                .iter()
                .zip(fields.fields())
                .map(|(name, dtype)| (name.to_string(), dtype))
                .collect()
        })
        .unwrap_or_default()
}

fn open_or_reuse_vortex_file(
    ctx: &ClientContextRef,
    bind_data: &VortexBindData,
    file_path: &str,
) -> VortexResult<(VortexFile, Vec<(String, DType)>)> {
    if let Some(first_file) = cached_first_file(bind_data, file_path) {
        return Ok(first_file);
    }

    let file = open_vortex_file(ctx, file_path)?;
    let column_dtypes = column_dtypes(&file);
    Ok((file, column_dtypes))
}

fn cached_first_file(
    bind_data: &VortexBindData,
    file_path: &str,
) -> Option<(VortexFile, Vec<(String, DType)>)> {
    let first_file = bind_data.first_file.as_ref()?;
    (first_file.path == file_path)
        .then(|| (first_file.file.clone(), first_file.column_dtypes.clone()))
}

fn vortex_metadata_cache_enabled(ctx: &ClientContextRef) -> bool {
    ctx.try_get_current_setting(VORTEX_METADATA_CACHE_SETTING)
        .is_some_and(|value| matches!(value.extract(), ExtractedValue::Boolean(true)))
}

fn footer_cache_memory(footer: &Footer) -> usize {
    footer
        .approx_byte_size()
        .unwrap_or(DEFAULT_FOOTER_CACHE_BYTES)
}

/// Build a [`ColumnStatistics`] from a Vortex `StatsSet`. Handles the shared
/// shape (min/max/has_null/max_string_length); same logic as the existing
/// `datasource.rs` path.
fn make_column_statistics(
    stats_set: &vortex::array::stats::StatsSet,
    dtype: &DType,
) -> ColumnStatistics {
    use vortex::expr::stats::Precision;
    use vortex::expr::stats::Stat;
    use vortex::scalar::Scalar;

    use crate::convert::ToDuckDBScalar;

    let min = match stats_set.get(Stat::Min) {
        Some(Precision::Exact(v)) => Scalar::try_new(dtype.clone(), Some(v))
            .ok()
            .and_then(|s| s.try_to_duckdb_scalar().ok()),
        _ => None,
    };
    let max = match stats_set.get(Stat::Max) {
        Some(Precision::Exact(v)) => Scalar::try_new(dtype.clone(), Some(v))
            .ok()
            .and_then(|s| s.try_to_duckdb_scalar().ok()),
        _ => None,
    };
    let max_string_length = match stats_set.get(Stat::UncompressedSizeInBytes) {
        Some(Precision::Exact(v)) => v
            .as_primitive()
            .as_u64()
            .map(|u| (1u64 << 63) | u)
            .unwrap_or(0),
        _ => 0,
    };
    let has_null = match stats_set.get(Stat::NullCount) {
        Some(Precision::Exact(c)) => c.as_primitive().as_u64().map(|u| u > 0).unwrap_or(true),
        _ => true,
    } && dtype.is_nullable();

    ColumnStatistics {
        min,
        max,
        max_string_length,
        has_null,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex::array::IntoArray;
    use vortex::array::arrays::ChunkedArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::stream::ArrayStreamAdapter;
    use vortex::expr::lit;
    use vortex::expr::lt_eq;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
    use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;

    use super::*;
    use crate::cpp::DUCKDB_TYPE;
    use crate::duckdb::DataChunk;

    #[test]
    fn try_initialize_scan_assigns_independent_splits() -> VortexResult<()> {
        let rows_per_chunk = i32::try_from(MULTI_FILE_SCAN_MIN_SPLIT_ROWS)
            .map_err(|_| vortex_err!("split row count does not fit i32"))?;
        let (_temp_file, file) = write_chunked_struct_file(4, rows_per_chunk)?;
        let total_rows = file.row_count();
        let mut reader = VortexFileReader {
            file,
            file_idx: 0,
            cache: ConversionCache::default(),
            projection: None,
            field_positions: vec![],
            scan_column_count: 0,
            file_index_column_pos: None,
            file_row_number_column_pos: None,
            row_selection: Selection::All,
            row_range: None,
            filter: None,
            complex_filter_exprs: vec![],
            file_pruned: false,
            tasks: Mutex::new(VecDeque::new()),
            metadata_only_claimed: AtomicBool::new(false),
            total_rows,
            rows_scanned: AtomicU64::new(0),
        };
        let projection = [ProjectedColumn {
            name: "number",
            column_id: 0,
            is_virtual: false,
            is_projected: true,
        }];
        reader.prepare_reader(&projection, None)?;
        let task_count = reader.tasks.lock().len();
        assert!(
            task_count > 1,
            "test file should expose multiple layout splits"
        );

        let global = VortexGlobal;
        let mut first_local = VortexLocal::default();
        let mut second_local = VortexLocal::default();
        assert!(reader.try_initialize_scan(&global, &mut first_local)?);
        assert!(reader.try_initialize_scan(&global, &mut second_local)?);
        assert!(first_local.task.is_some());
        assert!(second_local.task.is_some());
        assert_eq!(reader.tasks.lock().len(), task_count - 2);
        reader.prepare_scan(&global, &mut first_local)?;
        reader.prepare_scan(&global, &mut second_local)?;
        assert!(first_local.task.is_none());
        assert!(second_local.task.is_none());
        assert!(!first_local.exporters.is_empty());
        assert!(!second_local.exporters.is_empty());

        let mut chunk = DataChunk::new([LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER)]);
        reader.scan(&global, &mut first_local, &mut chunk)?;
        assert!(!chunk.is_empty());
        reader.scan(&global, &mut second_local, &mut chunk)?;
        assert!(!chunk.is_empty());

        Ok(())
    }

    #[test]
    fn prepare_reader_does_not_project_filter_only_columns() -> VortexResult<()> {
        let (_temp_file, file) = write_two_column_struct_file(64)?;
        let total_rows = file.row_count();
        let mut reader = VortexFileReader {
            file,
            file_idx: 0,
            cache: ConversionCache::default(),
            projection: None,
            field_positions: vec![],
            scan_column_count: 0,
            file_index_column_pos: None,
            file_row_number_column_pos: None,
            row_selection: Selection::All,
            row_range: None,
            filter: None,
            complex_filter_exprs: vec![],
            file_pruned: false,
            tasks: Mutex::new(VecDeque::new()),
            metadata_only_claimed: AtomicBool::new(false),
            total_rows,
            rows_scanned: AtomicU64::new(0),
        };
        let projection = [
            ProjectedColumn {
                name: "filter_key",
                column_id: 0,
                is_virtual: false,
                is_projected: false,
            },
            ProjectedColumn {
                name: "payload",
                column_id: 1,
                is_virtual: false,
                is_projected: true,
            },
        ];

        reader.prepare_reader(&projection, None)?;

        let projected = reader
            .projection
            .as_ref()
            .unwrap()
            .iter()
            .map(|name| name.as_ref())
            .collect::<Vec<_>>();
        assert_eq!(projected, ["payload"]);
        assert_eq!(reader.field_positions, [1]);
        assert_eq!(reader.scan_column_count, 2);

        Ok(())
    }

    #[test]
    fn filtered_count_star_projects_filter_fields_for_row_counts() -> VortexResult<()> {
        let (_temp_file, file) = write_two_column_struct_file(64)?;
        let total_rows = file.row_count();
        let mut reader = VortexFileReader {
            file,
            file_idx: 0,
            cache: ConversionCache::default(),
            projection: None,
            field_positions: vec![],
            scan_column_count: 0,
            file_index_column_pos: None,
            file_row_number_column_pos: None,
            row_selection: Selection::All,
            row_range: None,
            filter: None,
            complex_filter_exprs: vec![lt_eq(col("filter_key"), lit(1i32))],
            file_pruned: false,
            tasks: Mutex::new(VecDeque::new()),
            metadata_only_claimed: AtomicBool::new(false),
            total_rows,
            rows_scanned: AtomicU64::new(0),
        };
        let projection: [ProjectedColumn<'_>; 0] = [];

        reader.prepare_reader(&projection, None)?;

        let projected = reader
            .projection
            .as_ref()
            .unwrap()
            .iter()
            .map(|name| name.as_ref())
            .collect::<Vec<_>>();
        assert_eq!(projected, ["filter_key"]);
        assert!(reader.field_positions.is_empty());
        assert_eq!(reader.scan_column_count, 0);

        let global = VortexGlobal;
        let mut rows_seen = 0;
        loop {
            let mut local = VortexLocal::default();
            if !reader.try_initialize_scan(&global, &mut local)? {
                break;
            }
            reader.prepare_scan(&global, &mut local)?;
            let mut chunk = DataChunk::new([]);
            loop {
                reader.scan(&global, &mut local, &mut chunk)?;
                if chunk.is_empty() {
                    break;
                }
                rows_seen += chunk.len();
            }
        }
        assert_eq!(rows_seen, 2);

        Ok(())
    }

    #[test]
    fn count_star_uses_one_metadata_only_assignment() -> VortexResult<()> {
        let (_temp_file, file) = write_chunked_struct_file(4, 16_384)?;
        let total_rows = file.row_count();
        let mut reader = VortexFileReader {
            file,
            file_idx: 0,
            cache: ConversionCache::default(),
            projection: None,
            field_positions: vec![],
            scan_column_count: 0,
            file_index_column_pos: None,
            file_row_number_column_pos: None,
            row_selection: Selection::All,
            row_range: None,
            filter: None,
            complex_filter_exprs: vec![],
            file_pruned: false,
            tasks: Mutex::new(VecDeque::new()),
            metadata_only_claimed: AtomicBool::new(false),
            total_rows,
            rows_scanned: AtomicU64::new(0),
        };
        let projection: [ProjectedColumn<'_>; 0] = [];
        reader.prepare_reader(&projection, None)?;
        assert!(reader.tasks.lock().is_empty());

        let global = VortexGlobal;
        let mut local = VortexLocal::default();
        assert!(reader.try_initialize_scan(&global, &mut local)?);
        reader.prepare_scan(&global, &mut local)?;
        assert!(local.task.is_none());
        assert!(
            !reader.try_initialize_scan(&global, &mut VortexLocal::default())?,
            "COUNT(*) should not claim one assignment per layout split"
        );

        let mut chunk = DataChunk::new([]);
        let mut rows_seen = 0;
        loop {
            reader.scan(&global, &mut local, &mut chunk)?;
            if chunk.is_empty() {
                break;
            }
            rows_seen += chunk.len();
            assert!(local.task.is_none());
        }
        assert_eq!(rows_seen, total_rows);

        Ok(())
    }

    #[test]
    fn cardinality_uses_first_file_row_count_from_bind_data() -> VortexResult<()> {
        let (_temp_file, file) = write_chunked_struct_file(1, 42)?;
        let bind_data = VortexBindData {
            first_file: Some(BoundFirstFile {
                path: "first.vortex".to_string(),
                column_dtypes: column_dtypes(&file),
                file,
            }),
            complex_filter_exprs: vec![],
        };

        let Cardinality::Maximum(42) = VortexMultiFileFunction::cardinality(&bind_data, 1) else {
            panic!("single-file cardinality should be exact maximum");
        };
        let Cardinality::Estimate(126) = VortexMultiFileFunction::cardinality(&bind_data, 3) else {
            panic!("multi-file cardinality should estimate from first file row count");
        };

        let bind_data = VortexBindData::default();
        let Cardinality::Unknown = VortexMultiFileFunction::cardinality(&bind_data, 3) else {
            panic!("cardinality should be unknown before bind_reader records row count");
        };

        Ok(())
    }

    #[test]
    fn cardinality_is_unknown_with_pushed_complex_filters() -> VortexResult<()> {
        let (_temp_file, file) = write_chunked_struct_file(1, 42)?;
        let bind_data = VortexBindData {
            first_file: Some(BoundFirstFile {
                path: "first.vortex".to_string(),
                column_dtypes: column_dtypes(&file),
                file,
            }),
            complex_filter_exprs: vec![root()],
        };

        let Cardinality::Unknown = VortexMultiFileFunction::cardinality(&bind_data, 1) else {
            panic!("cardinality should be unknown after pushing a complex filter into Vortex");
        };

        Ok(())
    }

    #[test]
    fn partition_stats_are_disabled_with_pushed_complex_filters() -> VortexResult<()> {
        let (_temp_file, file) = write_chunked_struct_file(1, 42)?;
        let db = crate::duckdb::Database::open_in_memory()?;
        let conn = db.connect()?;
        let ctx = conn.client_context()?;
        let bind_data = VortexBindData {
            first_file: Some(BoundFirstFile {
                path: "first.vortex".to_string(),
                column_dtypes: column_dtypes(&file),
                file,
            }),
            complex_filter_exprs: vec![root()],
        };

        assert!(
            VortexMultiFileFunction::partition_stats(ctx, &bind_data, "first.vortex")?.is_none(),
            "physical file row counts are pre-filter and cannot be exact scan output stats"
        );

        Ok(())
    }

    #[test]
    fn cached_first_file_matches_only_bind_path() -> VortexResult<()> {
        let (_temp_file, file) = write_chunked_struct_file(1, 16)?;
        let bind_data = VortexBindData {
            first_file: Some(BoundFirstFile {
                path: "first.vortex".to_string(),
                column_dtypes: column_dtypes(&file),
                file,
            }),
            complex_filter_exprs: vec![],
        };

        let Some((cached, cached_dtypes)) = cached_first_file(&bind_data, "first.vortex") else {
            panic!("expected first file to be reused for matching path");
        };
        assert_eq!(cached.row_count(), 16);
        assert_eq!(cached_dtypes.len(), 1);
        assert!(cached_first_file(&bind_data, "other.vortex").is_none());

        Ok(())
    }

    #[test]
    fn make_exporters_unwraps_chunked_struct_batches_before_export() -> VortexResult<()> {
        let first = struct_with_dict_strings(["a", "b"], [0u32, 1, 0])?;
        let second = struct_with_dict_strings(["c", "d"], [1u32, 0, 1])?;
        let dtype = first.dtype().clone();
        let array = ChunkedArray::try_new(vec![first.into_array(), second.into_array()], dtype)?
            .into_array();

        let mut exporters = make_exporters(array, &ConversionCache::default(), vec![0], 1)?;

        assert_eq!(exporters.len(), 2);

        let mut first_exporter = exporters.pop_front().unwrap();
        let mut chunk = DataChunk::new([LogicalType::varchar()]);
        assert!(first_exporter.export(&mut chunk, None, None)?);
        let display = String::try_from(&*chunk)?;

        assert!(
            display.contains("DICTIONARY VARCHAR"),
            "expected dictionary export, got:\n{display}"
        );

        Ok(())
    }

    fn write_chunked_struct_file(
        chunk_count: usize,
        rows_per_chunk: i32,
    ) -> VortexResult<(tempfile::NamedTempFile, VortexFile)> {
        RUNTIME.block_on(async {
            let temp_file = tempfile::Builder::new().suffix(".vortex").tempfile()?;
            let chunks = (0..chunk_count)
                .map(|chunk_idx| {
                    let chunk_idx = i32::try_from(chunk_idx)
                        .map_err(|_| vortex_err!("chunk index does not fit i32"))?;
                    let start = chunk_idx * rows_per_chunk;
                    let numbers = PrimitiveArray::from_iter(start..start + rows_per_chunk);
                    StructArray::from_fields(&[("number", numbers.into_array())])
                        .map(IntoArray::into_array)
                })
                .collect::<VortexResult<Vec<_>>>()?;
            let dtype = chunks[0].dtype().clone();
            let stream = futures::stream::iter(chunks.into_iter().map(Ok));
            let stream = ArrayStreamAdapter::new(dtype, stream);

            let mut writer = async_fs::File::create(&temp_file).await?;
            SESSION
                .write_options()
                .with_strategy(Arc::new(ChunkedLayoutStrategy::new(
                    FlatLayoutStrategy::default(),
                )))
                .write(&mut writer, stream)
                .await?;
            drop(writer);

            let file = SESSION.open_options().open_path(temp_file.path()).await?;
            Ok((temp_file, file))
        })
    }

    fn write_two_column_struct_file(
        rows: i32,
    ) -> VortexResult<(tempfile::NamedTempFile, VortexFile)> {
        RUNTIME.block_on(async {
            let temp_file = tempfile::Builder::new().suffix(".vortex").tempfile()?;
            let filter_key = PrimitiveArray::from_iter(0..rows);
            let payload = PrimitiveArray::from_iter((0..rows).map(|value| value * 10));
            let chunk = StructArray::from_fields(&[
                ("filter_key", filter_key.into_array()),
                ("payload", payload.into_array()),
            ])?
            .into_array();
            let dtype = chunk.dtype().clone();
            let stream = futures::stream::iter([Ok(chunk)]);
            let stream = ArrayStreamAdapter::new(dtype, stream);

            let mut writer = async_fs::File::create(&temp_file).await?;
            SESSION
                .write_options()
                .with_strategy(Arc::new(ChunkedLayoutStrategy::new(
                    FlatLayoutStrategy::default(),
                )))
                .write(&mut writer, stream)
                .await?;
            drop(writer);

            let file = SESSION.open_options().open_path(temp_file.path()).await?;
            Ok((temp_file, file))
        })
    }

    fn struct_with_dict_strings<const N: usize>(
        values: [&str; 2],
        codes: [u32; N],
    ) -> VortexResult<StructArray> {
        let values = VarBinViewArray::from_iter_str(values).into_array();
        let codes = PrimitiveArray::from_iter(codes).into_array();
        let strings = DictArray::new(codes, values).into_array();
        StructArray::from_fields(&[("s", strings)])
    }
}
