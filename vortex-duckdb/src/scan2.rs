// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan API implementation of the DuckDB `vortex_scan` table function.
//!
//! Uses [`MultiFileDataSource`] for file discovery and scanning via the Scan API.
//! Enabled by setting `VORTEX_USE_SCAN_API=1`.

use std::ffi::CString;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_compat::Compat;
use custom_labels::CURRENT_LABELSET;
use futures::StreamExt;
use glob::Pattern;
use itertools::Itertools;
use num_traits::AsPrimitive;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::ScalarFnVTable;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::StructVTable;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::Pack;
use vortex::expr::and_collect;
use vortex::expr::col;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::multi::FileDiscovery;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::ThreadSafeIterator;
use vortex::metrics::tracing::get_global_labels;
use vortex::scan::api::DataSource as _;
use vortex::scan::api::DataSourceRef;
use vortex::scan::api::ScanRequest;
use vortex::session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::RUNTIME;
use crate::SESSION;
use crate::convert::try_from_bound_expression;
use crate::convert::try_from_table_filter;
use crate::duckdb;
use crate::duckdb::BindInput;
use crate::duckdb::BindResult;
use crate::duckdb::Cardinality;
use crate::duckdb::ClientContext;
use crate::duckdb::DataChunk;
use crate::duckdb::ExtractedValue;
use crate::duckdb::LogicalType;
use crate::duckdb::TableFunction;
use crate::duckdb::TableInitInput;
use crate::duckdb::VirtualColumnsResult;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::scan::MultiScan;
use crate::utils::object_store::s3_store;

pub(crate) struct ScanApiBindData {
    data_source: DataSourceRef,
    filter_exprs: Vec<Expression>,
    column_names: Vec<String>,
    column_types: Vec<LogicalType>,
    max_threads: u64,
    file_count: usize,
    row_count_estimate: u64,
}

impl Clone for ScanApiBindData {
    fn clone(&self) -> Self {
        Self {
            data_source: self.data_source.clone(),
            // filter_exprs are consumed once in `init_global`.
            filter_exprs: vec![],
            column_names: self.column_names.clone(),
            column_types: self.column_types.clone(),
            max_threads: self.max_threads,
            file_count: self.file_count,
            row_count_estimate: self.row_count_estimate,
        }
    }
}

impl Debug for ScanApiBindData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanApiBindData")
            .field("column_names", &self.column_names)
            .field("column_types", &self.column_types)
            .field("filter_exprs", &self.filter_exprs)
            .field("file_count", &self.file_count)
            .finish()
    }
}

pub(crate) struct ScanApiGlobalData {
    iterator: ThreadSafeIterator<VortexResult<(ArrayRef, Arc<ConversionCache>)>>,
    batch_id: AtomicU64,
    ctx: ExecutionCtx,
}

pub(crate) struct ScanApiLocalData {
    iterator: ThreadSafeIterator<VortexResult<(ArrayRef, Arc<ConversionCache>)>>,
    exporter: Option<ArrayExporter>,
    batch_id: Option<u64>,
}

#[derive(Debug)]
pub(crate) struct ScanApiTableFunction;

/// Parse a URL glob string into an object store, base URL, and file discovery.
///
/// Supports S3 (`s3://bucket/path/*.vortex`) and local (`/path/*.vortex`) URLs.
fn parse_url_glob(url_glob: &str) -> VortexResult<(Arc<dyn ObjectStore>, Url, FileDiscovery)> {
    let is_s3 = url_glob.starts_with("s3://");

    // Find the first glob character to split base URL from pattern.
    let first_glob = url_glob.find(['*', '?', '[']);

    // Split at the last '/' before the first glob character.
    let split_before = first_glob.unwrap_or(url_glob.len());
    let split_pos = url_glob[..split_before]
        .rfind('/')
        .map(|i| i + 1)
        .unwrap_or(0);
    let base_url_str = &url_glob[..split_pos];
    let pattern_str = &url_glob[split_pos..];

    let pattern = Pattern::new(pattern_str)
        .map_err(|e| vortex_err!("Invalid glob pattern '{}': {}", pattern_str, e))?;

    if is_s3 {
        let base_url = Url::parse(base_url_str)
            .map_err(|e| vortex_err!("Invalid S3 URL '{}': {}", base_url_str, e))?;
        let bucket = base_url
            .host_str()
            .ok_or_else(|| vortex_err!("Missing bucket in S3 URL: {}", base_url_str))?;
        Ok((s3_store(bucket)?, base_url, FileDiscovery::Glob(pattern)))
    } else {
        let path_str = base_url_str.strip_prefix("file://").unwrap_or(base_url_str);
        let canonical = std::fs::canonicalize(path_str)
            .map_err(|e| vortex_err!("Failed to resolve path '{}': {}", path_str, e))?;
        let base_url = Url::from_directory_path(&canonical)
            .map_err(|_| vortex_err!("Invalid directory path: {}", canonical.display()))?;
        let store: Arc<dyn ObjectStore> = Arc::new(
            LocalFileSystem::new_with_prefix("/")
                .map_err(|e| vortex_err!("Failed to create local filesystem: {}", e))?,
        );
        Ok((store, base_url, FileDiscovery::Glob(pattern)))
    }
}

/// Extract column names and DuckDB types from a Vortex [`DType`].
fn extract_schema_from_dtype(dtype: &DType) -> VortexResult<(Vec<String>, Vec<LogicalType>)> {
    let struct_dtype = dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("Vortex file must contain a struct array at the top level"))?;

    let mut column_names = Vec::new();
    let mut column_types = Vec::new();

    for (field_name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
        let logical_type = LogicalType::try_from(&field_dtype)?;
        column_names.push(field_name.to_string());
        column_types.push(logical_type);
    }

    Ok((column_names, column_types))
}

/// Creates a projection expression from the table initialization input.
fn extract_projection_expr(init: &TableInitInput<ScanApiTableFunction>) -> Expression {
    let projection_ids = init.projection_ids().unwrap_or(&[]);
    let column_ids = init.column_ids();

    select(
        projection_ids
            .iter()
            .map(|p| {
                let idx: usize = p.as_();
                let val: usize = column_ids[idx].as_();
                val
            })
            .map(|idx| {
                init.bind_data()
                    .column_names
                    .get(idx)
                    .vortex_expect("prune idx in column names")
            })
            .map(|s| Arc::from(s.as_str()))
            .collect::<FieldNames>(),
        root(),
    )
}

/// Creates a filter expression from the table filter set.
fn extract_table_filter_expr(
    init: &TableInitInput<ScanApiTableFunction>,
    column_ids: &[u64],
) -> VortexResult<Option<Expression>> {
    let mut table_filter_exprs: HashSet<Expression> = if let Some(filter) = init.table_filter_set()
    {
        filter
            .into_iter()
            .map(|(idx, ex)| {
                let idx_u: usize = idx.as_();
                let col_idx: usize = column_ids[idx_u].as_();
                let name = init
                    .bind_data()
                    .column_names
                    .get(col_idx)
                    .vortex_expect("exists");
                try_from_table_filter(
                    &ex,
                    &col(name.as_str()),
                    init.bind_data().data_source.dtype(),
                )
            })
            .collect::<VortexResult<Option<HashSet<_>>>>()?
            .unwrap_or_else(HashSet::new)
    } else {
        HashSet::new()
    };

    table_filter_exprs.extend(init.bind_data().filter_exprs.clone());
    Ok(and_collect(table_filter_exprs.into_iter().collect_vec()))
}

// Taken from duckdb/common/constants.h COLUMN_IDENTIFIER_EMPTY
static EMPTY_COLUMN_IDX: u64 = 18446744073709551614;
static EMPTY_COLUMN_NAME: &str = "";

impl TableFunction for ScanApiTableFunction {
    type BindData = ScanApiBindData;
    type GlobalState = ScanApiGlobalData;
    type LocalState = ScanApiLocalData;

    const PROJECTION_PUSHDOWN: bool = true;
    const FILTER_PUSHDOWN: bool = true;
    const FILTER_PRUNE: bool = true;

    fn parameters() -> Vec<LogicalType> {
        vec![LogicalType::varchar()]
    }

    fn bind(
        ctx: &ClientContext,
        input: &BindInput,
        result: &mut BindResult,
    ) -> VortexResult<Self::BindData> {
        let file_glob_string = input
            .get_parameter(0)
            .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

        let max_threads_cstr = CString::new("vortex_max_threads")
            .map_err(|e| vortex_err!("Invalid setting name: {}", e))?;
        let max_threads = ctx
            .try_get_current_setting(&max_threads_cstr)
            .and_then(|v| match v.as_ref().extract() {
                ExtractedValue::UBigInt(val) => usize::try_from(val).ok(),
                ExtractedValue::BigInt(val) if val > 0 => usize::try_from(val as u64).ok(),
                _ => None,
            })
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });

        tracing::trace!("running scan with max_threads {max_threads}");

        let (object_store, base_url, discovery) =
            parse_url_glob(&file_glob_string.as_ref().as_string())?;

        let multi_ds = RUNTIME.block_on(Compat::new(
            MultiFileDataSource::builder(SESSION.clone(), object_store, base_url)
                .with_discovery(discovery)
                .with_prefetch(max_threads * 2)
                .build(),
        ))?;

        let (column_names, column_types) = extract_schema_from_dtype(multi_ds.dtype())?;

        for (column_name, column_type) in column_names.iter().zip(&column_types) {
            result.add_result_column(column_name, column_type);
        }

        let file_count = multi_ds.file_count();
        let estimate = multi_ds.row_count_estimate();
        let row_count_estimate = estimate.upper.unwrap_or(estimate.lower);
        let data_source: DataSourceRef = Arc::new(multi_ds);

        Ok(ScanApiBindData {
            data_source,
            filter_exprs: vec![],
            column_names,
            column_types,
            max_threads: max_threads as u64,
            file_count,
            row_count_estimate,
        })
    }

    fn scan(
        _client_context: &ClientContext,
        _bind_data: &Self::BindData,
        local_state: &mut Self::LocalState,
        global_state: &mut Self::GlobalState,
        chunk: &mut DataChunk,
    ) -> VortexResult<()> {
        loop {
            if local_state.exporter.is_none() {
                let Some(result) = local_state.iterator.next() else {
                    return Ok(());
                };

                let (array_result, conversion_cache) = result?;

                let array_result = array_result.optimize_recursive()?;
                let array_result = if let Some(array) = array_result.as_opt::<StructVTable>() {
                    array.clone()
                } else if let Some(array) = array_result.as_opt::<ScalarFnVTable>()
                    && let Some(pack_options) = array.scalar_fn().as_opt::<Pack>()
                {
                    StructArray::new(
                        pack_options.names.clone(),
                        array.children(),
                        array.len(),
                        pack_options.nullability.into(),
                    )
                } else {
                    array_result
                        .execute::<Canonical>(&mut global_state.ctx)?
                        .into_struct()
                };

                local_state.exporter = Some(ArrayExporter::try_new(
                    &array_result,
                    &conversion_cache,
                    &mut global_state.ctx,
                )?);
                local_state.batch_id = Some(global_state.batch_id.fetch_add(1, Ordering::Relaxed));
            }

            let exporter = local_state
                .exporter
                .as_mut()
                .vortex_expect("error: exporter missing");

            let has_more_data = exporter.export(chunk)?;

            if !has_more_data {
                local_state.exporter = None;
                local_state.batch_id = None;
            } else {
                break;
            }
        }

        assert!(!chunk.is_empty());

        Ok(())
    }

    fn init_global(init_input: &TableInitInput<Self>) -> VortexResult<Self::GlobalState> {
        let bind_data = init_input.bind_data();
        let projection_expr = extract_projection_expr(init_input);
        let filter_expr = extract_table_filter_expr(init_input, init_input.column_ids())?;

        tracing::trace!(
            "Global init Vortex scan (v2) SELECT {} WHERE {}",
            &projection_expr,
            filter_expr
                .as_ref()
                .map_or_else(|| "true".to_string(), |f| f.to_string())
        );

        #[expect(clippy::cast_possible_truncation, reason = "max_threads fits in usize")]
        let num_workers = bind_data.max_threads as usize;

        let request = ScanRequest {
            projection: Some(projection_expr),
            filter: filter_expr,
            ..Default::default()
        };

        let scan = bind_data.data_source.scan(request)?;
        let conversion_cache = Arc::new(ConversionCache::new(0));

        let scan_streams = scan.splits().then(move |split_result| {
            let cache = conversion_cache.clone();
            async move {
                let split = split_result?;
                let s = split.execute().await?;
                Ok(s.map(move |r| Ok((r?, cache.clone()))).boxed())
            }
        });

        let iterator = RUNTIME.block_on_stream_thread_safe(move |_| MultiScan {
            streams: scan_streams.boxed(),
            streams_finished: false,
            select_all: Default::default(),
            max_concurrency: num_workers * 2,
        });

        Ok(ScanApiGlobalData {
            iterator,
            batch_id: AtomicU64::new(0),
            ctx: ExecutionCtx::new(VortexSession::default()),
        })
    }

    fn init_local(
        _init: &TableInitInput<Self>,
        global: &mut Self::GlobalState,
    ) -> VortexResult<Self::LocalState> {
        unsafe {
            use custom_labels::sys;

            if sys::labelset_current().is_null() {
                let ls = sys::labelset_new(0);
                sys::labelset_replace(ls);
            };
        }

        let global_labels = get_global_labels();

        for (key, value) in global_labels {
            CURRENT_LABELSET.set(key, value);
        }

        Ok(ScanApiLocalData {
            iterator: global.iterator.clone(),
            exporter: None,
            batch_id: None,
        })
    }

    fn pushdown_complex_filter(
        bind_data: &mut Self::BindData,
        expr: &duckdb::Expression,
    ) -> VortexResult<bool> {
        let Some(expr) = try_from_bound_expression(expr)? else {
            return Ok(false);
        };
        bind_data.filter_exprs.push(expr);
        Ok(false)
    }

    fn cardinality(bind_data: &Self::BindData) -> Cardinality {
        if bind_data.file_count == 1 {
            Cardinality::Maximum(bind_data.row_count_estimate)
        } else {
            Cardinality::Estimate(bind_data.row_count_estimate)
        }
    }

    fn partition_data(
        _bind_data: &Self::BindData,
        _global_init_data: &mut Self::GlobalState,
        local_init_data: &mut Self::LocalState,
    ) -> VortexResult<u64> {
        local_init_data
            .batch_id
            .ok_or_else(|| vortex_err!("batch id missing, no batches exported"))
    }

    fn to_string(bind_data: &Self::BindData) -> Option<Vec<(String, String)>> {
        let mut result = Vec::new();
        result.push(("Function".to_string(), "Vortex Scan (v2)".to_string()));
        result.push(("Files".to_string(), bind_data.file_count.to_string()));
        if !bind_data.filter_exprs.is_empty() {
            let mut filters = bind_data.filter_exprs.iter().map(|f| format!("{}", f));
            result.push(("Filters".to_string(), filters.join(" /\\\n")));
        }
        Some(result)
    }

    fn virtual_columns(_bind_data: &Self::BindData, result: &mut VirtualColumnsResult) {
        result.register(EMPTY_COLUMN_IDX, EMPTY_COLUMN_NAME, &LogicalType::bool());
    }
}
