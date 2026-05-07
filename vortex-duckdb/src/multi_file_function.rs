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
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::SESSION;
use crate::duckdb::BaseFileReader;
use crate::duckdb::ClientContextRef;
use crate::duckdb::ColumnStatistics;
use crate::duckdb::DataChunkRef;
use crate::duckdb::LogicalType;
use crate::duckdb::MultiFileFunction;
use crate::duckdb::SchemaBuilder;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;

/// Multi-file Vortex scan registered via `MultiFileFunction<OP>`.
///
/// Compared to [`crate::multi_file::VortexMultiFileScan`] (the table-function
/// path), this delegates file globbing, virtual columns, and hive partitioning
/// to DuckDB's native machinery, and reads each file via [`VortexFile`].
#[derive(Debug)]
pub struct VortexMultiFileFunction;

#[derive(Default)]
pub struct VortexReaderOptions;

#[derive(Debug)]
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
    /// Cached schema name list for stats lookups.
    column_dtypes: Vec<(String, DType)>,
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
        _ctx: &ClientContextRef,
        _bind_data: &Self::BindData,
        first_file: &str,
        schema: &mut SchemaBuilder,
    ) -> VortexResult<()> {
        // Open the first file to discover the schema. DuckDB picks the first
        // file in the multi-file list for binding.
        let first_file = first_file.to_string();
        let file =
            RUNTIME.block_on(async move { SESSION.open_options().open_path(&first_file).await })?;
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
        _ctx: &ClientContextRef,
        _global: &Self::GlobalState,
        _bind_data: &Self::BindData,
        file_path: &str,
        file_idx: usize,
    ) -> VortexResult<Self::Reader> {
        let path = file_path.to_string();
        let file =
            RUNTIME.block_on(async move { SESSION.open_options().open_path(&path).await })?;

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
        })
    }
}

impl BaseFileReader for VortexFileReader {
    type GlobalState = VortexGlobal;
    type LocalState = VortexLocal;

    fn try_initialize_scan(
        &mut self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
    ) -> VortexResult<bool> {
        // After the iterator has produced everything, return false to signal
        // end-of-file to DuckDB.
        if self.iter.is_some() && self.exporter.is_none() {
            return Ok(false);
        }
        if self.iter.is_none() {
            let iter = self.file.scan()?.into_array_iter(&*RUNTIME)?;
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
                    return Ok(());
                }
                self.exporter = None;
            }

            let Some(iter) = self.iter.as_mut() else {
                // Can happen if scan is called without try_initialize_scan; treat as end.
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
        // We don't currently track byte-level progress per file; report 0% so
        // DuckDB falls back to file-count-based progress.
        0.0
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
