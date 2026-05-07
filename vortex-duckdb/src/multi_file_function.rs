// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex implementation of [`MultiFileFunction`].
//!
//! Plugs Vortex into DuckDB's `MultiFileFunction<OP>` template: cross-file
//! orchestration (globbing, parallelism, virtual columns, hive partitioning)
//! is handled by DuckDB. This module supplies the per-file reader using
//! [`VortexFile`] directly so file-level statistics, dtype, and pruning are
//! available without going through `MultiLayoutDataSource`.

use std::fmt::Debug;

use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::iter::ArrayIterator;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::col;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::SESSION;
use crate::convert::try_from_table_filter;
use crate::duckdb::BaseFileReader;
use crate::duckdb::Cardinality;
use crate::duckdb::ClientContextRef;
use crate::duckdb::ColumnStatistics;
use crate::duckdb::DataChunkRef;
use crate::duckdb::DuckdbStringMapRef;
use crate::duckdb::LogicalType;
use crate::duckdb::MultiFileFunction;
use crate::duckdb::SchemaBuilder;
use crate::duckdb::TableFilterSetRef;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::filesystem::resolve_filesystem;
use crate::multi_file::parse_glob_url;

/// Open a [`VortexFile`] using whichever filesystem the user has configured
/// via the `vortex_filesystem` extension option. DuckDB has already expanded
/// any glob and chosen this exact path; we only need the right reader for
/// the URL scheme. Routing through the filesystem also lets HTTP/S3/etc.
/// transparently use DuckDB's `httpfs` when the user picks `'duckdb'`.
fn open_vortex_file(ctx: &ClientContextRef, path: &str) -> VortexResult<VortexFile> {
    let url = parse_glob_url(path)?;
    let mut base_url = url.clone();
    base_url.set_path("");
    let fs = resolve_filesystem(&base_url, ctx)?;
    RUNTIME.block_on(async move {
        let reader = fs.open_read(url.path()).await?;
        SESSION.open_options().open(reader).await
    })
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
///
/// Bind data is currently empty: the C++ adapter calls
/// [`MultiFileFunction::bind_reader`] with a read-only borrow, so we can't
/// stash per-bind state (e.g. first-file row count, column names) here yet.
/// Cardinality and EXPLAIN consumers fall back to defaults — see the inline
/// comment in `bind_reader`.
#[derive(Debug, Default)]
pub struct VortexBindData;

#[derive(Debug)]
pub struct VortexGlobal;

#[derive(Debug, Default)]
pub struct VortexLocal;

/// Per-file scan state. Holds the open [`VortexFile`] plus an iterator over the
/// arrays it produces. The iterator is created lazily on first
/// `try_initialize_scan` and drained one batch at a time by `scan`.
pub struct VortexFileReader {
    file: VortexFile,
    file_idx: usize,
    /// Sync-blocking iterator over the file's array stream. Lazily initialized
    /// inside `try_initialize_scan` so opening many files in parallel doesn't
    /// each spin up a scan task before they're needed.
    iter: Option<Box<dyn ArrayIterator>>,
    /// Current chunk being drained. `ArrayExporter::export` returns false when
    /// it's empty, at which point we pull the next array from `iter`.
    exporter: Option<ArrayExporter>,
    cache: ConversionCache,
    /// Cached (name, dtype) pairs from the file's struct schema. Used by stats
    /// lookups in `get_statistics` to avoid re-walking the dtype each call.
    column_dtypes: Vec<(String, DType)>,
    /// Projection set by [`Self::prepare_reader`]. `None` means prepare was
    /// never called (defensive — scan all columns). `Some(empty)` is the
    /// explicit zero-projection case (e.g. `SELECT count(*)`); the scan
    /// produces struct arrays with no fields, and `ArrayExporter` short-
    /// circuits on the empty fields list.
    projection: Option<Vec<FieldName>>,
    /// Filter expression set by [`Self::prepare_reader`]. None when no filters
    /// were pushed down or when conversion failed.
    filter: Option<Expression>,
    /// Set when a filter has been pushed down and file-level statistics prove
    /// the file can be skipped. Causes [`Self::try_initialize_scan`] to return
    /// false without opening a scan iterator.
    file_pruned: bool,
    /// Total rows in the file, cached for [`Self::progress_in_file`].
    total_rows: u64,
    /// Rows produced so far. Bumped after each chunk in [`Self::scan`].
    rows_scanned: u64,
}

impl Debug for VortexFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexFileReader")
            .field("file_idx", &self.file_idx)
            .field("row_count", &self.file.row_count())
            .finish_non_exhaustive()
    }
}

impl MultiFileFunction for VortexMultiFileFunction {
    type ReaderOptions = VortexReaderOptions;
    type BindData = VortexBindData;
    type GlobalState = VortexGlobal;
    type LocalState = VortexLocal;
    type Reader = VortexFileReader;

    fn create_options(_ctx: &ClientContextRef) -> VortexResult<Self::ReaderOptions> {
        Ok(VortexReaderOptions)
    }

    fn initialize_bind_data(_options: Self::ReaderOptions) -> VortexResult<Self::BindData> {
        Ok(VortexBindData)
    }

    fn bind_reader(
        ctx: &ClientContextRef,
        _bind_data: &Self::BindData,
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
        Ok(())
    }

    fn cardinality(_bind_data: &Self::BindData, file_count: usize) -> Cardinality {
        // We don't yet plumb per-file row counts from bind_reader through to
        // bind_data (see comment above), so estimate a moderate per-file size.
        // The estimate is only used by the optimizer; correctness doesn't
        // depend on it.
        const APPROX_ROWS_PER_FILE: u64 = 1_000_000;
        if file_count == 0 {
            Cardinality::Unknown
        } else {
            Cardinality::Estimate(APPROX_ROWS_PER_FILE.saturating_mul(file_count as u64))
        }
    }

    fn to_string(_bind_data: &Self::BindData, map: &mut DuckdbStringMapRef) {
        map.push("Function", "Vortex Multi-File Scan");
    }

    fn init_global(
        _ctx: &ClientContextRef,
        _bind_data: &Self::BindData,
    ) -> VortexResult<Self::GlobalState> {
        Ok(VortexGlobal)
    }

    fn init_local(_global: &Self::GlobalState) -> Self::LocalState {
        VortexLocal
    }

    fn create_reader(
        ctx: &ClientContextRef,
        _global: &Self::GlobalState,
        _bind_data: &Self::BindData,
        file_path: &str,
        file_idx: usize,
    ) -> VortexResult<Self::Reader> {
        let file = open_vortex_file(ctx, file_path)?;

        // Pre-compute (name, dtype) pairs once so per-column stats lookups in
        // `get_statistics` don't reparse the struct dtype on each call.
        let column_dtypes = file
            .dtype()
            .as_struct_fields_opt()
            .map(|fields| {
                fields
                    .names()
                    .iter()
                    .zip(fields.fields())
                    .map(|(name, dtype)| (name.to_string(), dtype))
                    .collect()
            })
            .unwrap_or_default();

        let total_rows = file.row_count();
        Ok(VortexFileReader {
            file,
            file_idx,
            iter: None,
            exporter: None,
            cache: ConversionCache {
                file_index: file_idx,
                ..Default::default()
            },
            column_dtypes,
            projection: None,
            filter: None,
            file_pruned: false,
            total_rows,
            rows_scanned: 0,
        })
    }
}

impl BaseFileReader for VortexFileReader {
    type GlobalState = VortexGlobal;
    type LocalState = VortexLocal;

    fn prepare_reader(
        &mut self,
        projection: &[&str],
        filters: Option<&TableFilterSetRef>,
    ) -> VortexResult<()> {
        // Capture the projection in chunk order. `Some(empty)` is the explicit
        // zero-projection case (e.g. SELECT count(*)) and is handled when the
        // scan starts.
        let proj: Vec<FieldName> = projection.iter().map(|n| FieldName::from(*n)).collect();

        // Build a Vortex filter expression from DuckDB's table filters. Filter
        // indices are positions into the *projected* column list (matching
        // the v1 datasource path).
        if let Some(filters) = filters {
            let dtype = self.file.dtype();
            let mut pieces: Vec<Expression> = Vec::new();
            for (idx, filter) in filters.into_iter() {
                let idx = usize::try_from(idx)
                    .map_err(|_| vortex_err!("filter column index does not fit usize"))?;
                let Some(name) = proj.get(idx) else {
                    continue;
                };
                if let Some(expr) = try_from_table_filter(filter, &col(name.as_ref()), dtype)? {
                    pieces.push(expr);
                }
            }
            self.filter = and_collect(pieces);
        }
        self.projection = Some(proj);

        // File-level pruning: if the filter combined with the file's stored
        // statistics proves no row can match, skip this file entirely.
        if let Some(filter) = &self.filter
            && self.file.can_prune(filter)?
        {
            self.file_pruned = true;
        }
        Ok(())
    }

    fn try_initialize_scan(
        &mut self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
    ) -> VortexResult<bool> {
        if self.file_pruned {
            return Ok(false);
        }
        // Single-batch model: hand DuckDB one batch covering the whole file,
        // then signal exhaustion. (Per-split parallelism is a follow-up.)
        if self.iter.is_some() && self.exporter.is_none() {
            return Ok(false);
        }
        if self.iter.is_none() {
            let mut builder = self.file.scan()?;
            // Apply projection. `None` (prepare not called) defaults to all
            // columns; `Some` (including the empty case for SELECT count(*))
            // applies an explicit `select` so the resulting struct arrays
            // contain exactly the columns DuckDB expects.
            if let Some(names) = &self.projection {
                let names = FieldNames::from_iter(names.iter().cloned());
                builder = builder.with_projection(select(names, root()));
            }
            if let Some(filter) = self.filter.clone() {
                builder = builder.with_filter(filter);
            }
            let iter = builder.into_array_iter(&*RUNTIME)?;
            self.iter = Some(Box::new(iter));
        }
        Ok(true)
    }

    fn scan(
        &mut self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
        chunk: &mut DataChunkRef,
    ) -> VortexResult<()> {
        loop {
            // Drain the in-flight array if we have one.
            if let Some(exporter) = self.exporter.as_mut() {
                let has_more = exporter.export(chunk, None, None)?;
                if has_more {
                    self.rows_scanned = self.rows_scanned.saturating_add(chunk.len());
                    return Ok(());
                }
                self.exporter = None;
            }

            let Some(iter) = self.iter.as_mut() else {
                chunk.set_len(0);
                return Ok(());
            };

            let Some(next) = iter.next() else {
                self.iter.take();
                chunk.set_len(0);
                return Ok(());
            };
            let array = next?;
            self.exporter = Some(make_exporter(array, &self.cache)?);
        }
    }

    fn get_statistics(&self, name: &str) -> Option<ColumnStatistics> {
        let stats = self.file.file_stats()?;
        let (stats_set, _) = stats.get_by_name(self.file.dtype(), name)?;
        let dtype = &self.column_dtypes.iter().find(|(n, _)| n == name)?.1;
        Some(make_column_statistics(stats_set, dtype))
    }

    fn progress_in_file(&self) -> f64 {
        if self.total_rows == 0 {
            return 100.0;
        }
        let pct = (self.rows_scanned as f64 / self.total_rows as f64) * 100.0;
        pct.clamp(0.0, 100.0)
    }
}

/// Convert the next array off the scan stream into a [`StructArray`] suitable
/// for [`ArrayExporter`].
fn make_exporter(array: ArrayRef, cache: &ConversionCache) -> VortexResult<ArrayExporter> {
    let mut ctx = SESSION.create_execution_ctx();
    let struct_array: StructArray = if let Some(s) = array.as_opt::<Struct>() {
        s.into_owned()
    } else {
        array.execute::<Canonical>(&mut ctx)?.into_struct()
    };
    ArrayExporter::try_new(&struct_array, cache, ctx)
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
