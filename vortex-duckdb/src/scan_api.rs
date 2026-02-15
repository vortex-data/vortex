// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use async_compat::Compat;
use custom_labels::CURRENT_LABELSET;
use futures::StreamExt;
use futures::TryStreamExt;
use itertools::Itertools;
use object_store::local::LocalFileSystem;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::array::ExecutionCtx;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::file::FooterCacheRef;
use vortex::file::filesystem::FileSystemRef;
use vortex::file::filesystem::object_store::ObjectStoreFileSystem;
use vortex::file::multi::FileDiscovery;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::runtime::BlockingRuntime;
use vortex::metrics::tracing::get_global_labels;
use vortex::scan::api::DataSourceRef;
use vortex::scan::api::ScanRequest;
use vortex::session::VortexSession;

use crate::RUNTIME;
use crate::SESSION;
use crate::convert::try_from_bound_expression;
use crate::duckdb;
use crate::duckdb::BindInput;
use crate::duckdb::BindResult;
use crate::duckdb::Cardinality;
use crate::duckdb::ClientContext;
use crate::duckdb::DataChunk;
use crate::duckdb::LogicalType;
use crate::duckdb::TableFunction;
use crate::duckdb::TableInitInput;
use crate::duckdb::VirtualColumnsResult;
use crate::duckdb::footer_cache::DuckDbFooterCache;
use crate::exporter::ConversionCache;
use crate::scan::EMPTY_COLUMN_IDX;
use crate::scan::EMPTY_COLUMN_NAME;
use crate::scan::VortexGlobalData;
use crate::scan::VortexLocalData;
use crate::scan::extract_projection_expr_from;
use crate::scan::extract_schema_from_dtype;
use crate::scan::extract_table_filter_expr_from;
use crate::scan::scan_shared;
use crate::utils::glob::expand_glob;
use crate::utils::object_store::s3_store;

/// Bind data for the scan API table function, holding a [`DataSourceRef`] instead of
/// per-file URLs.
pub struct VortexScanApiBindData {
    data_source: DataSourceRef,
    filter_exprs: Vec<Expression>,
    column_names: Vec<String>,
    column_types: Vec<LogicalType>,
    file_count: usize,
}

impl Clone for VortexScanApiBindData {
    fn clone(&self) -> Self {
        Self {
            data_source: self.data_source.clone(),
            // filter_exprs are consumed once in `init_global`.
            filter_exprs: vec![],
            column_names: self.column_names.clone(),
            column_types: self.column_types.clone(),
            file_count: self.file_count,
        }
    }
}

impl Debug for VortexScanApiBindData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexScanApiBindData")
            .field("column_names", &self.column_names)
            .field("column_types", &self.column_types)
            .field("filter_exprs", &self.filter_exprs)
            .finish()
    }
}

/// Creates a [`FileSystemRef`] and relative paths from a list of URLs.
///
/// For S3 URLs, creates an S3 object store scoped to the bucket.
/// For local URLs (file:// or bare paths), uses a local filesystem.
fn create_filesystem_and_paths(urls: &[Url]) -> VortexResult<(FileSystemRef, Vec<String>)> {
    let first = urls
        .first()
        .ok_or_else(|| vortex_err!("No URLs provided"))?;

    if first.scheme() == "s3" {
        let bucket = first
            .host_str()
            .ok_or_else(|| vortex_err!("Failed to extract bucket name from URL: {first}"))?;
        let store = s3_store(bucket)?;
        let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(store, RUNTIME.handle()));
        let paths = urls
            .iter()
            .map(|url| {
                url.path()
                    .strip_prefix('/')
                    .ok_or_else(|| vortex_err!("Invalid S3 path: {url}"))
                    .map(|s| s.to_string())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok((fs, paths))
    } else {
        let store = Arc::new(LocalFileSystem::default());
        let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(store, RUNTIME.handle()));
        let paths = urls
            .iter()
            .map(|url| {
                url.to_file_path()
                    .map_err(|_| vortex_err!("Invalid file URL: {url}"))
                    .map(|p| p.to_string_lossy().to_string())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok((fs, paths))
    }
}

#[derive(Debug)]
pub struct VortexScanApiTableFunction;

impl TableFunction for VortexScanApiTableFunction {
    type BindData = VortexScanApiBindData;
    type GlobalState = VortexGlobalData;
    type LocalState = VortexLocalData;

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

        let (file_urls, _metadata) = RUNTIME.block_on(Compat::new(expand_glob(
            file_glob_string.as_ref().as_string(),
        )))?;

        if file_urls.is_empty() {
            vortex_bail!("No files matched the glob");
        }

        let footer_cache: FooterCacheRef = Arc::new(DuckDbFooterCache::new(ctx.object_cache()));
        let (fs, paths) = create_filesystem_and_paths(&file_urls)?;

        let file_count = file_urls.len();
        let data_source: DataSourceRef = Arc::new(RUNTIME.block_on(async {
            MultiFileDataSource::builder(SESSION.clone(), fs)
                .with_discovery(FileDiscovery::Paths(paths))
                .with_footer_cache(footer_cache)
                .build()
                .await
        })?);

        let (column_names, column_types) = extract_schema_from_dtype(data_source.dtype())?;

        for (column_name, column_type) in column_names.iter().zip(&column_types) {
            result.add_result_column(column_name, column_type);
        }

        Ok(VortexScanApiBindData {
            data_source,
            filter_exprs: vec![],
            column_names,
            column_types,
            file_count,
        })
    }

    fn scan(
        _client_context: &ClientContext,
        _bind_data: &Self::BindData,
        local_state: &mut Self::LocalState,
        global_state: &mut Self::GlobalState,
        chunk: &mut DataChunk,
    ) -> VortexResult<()> {
        scan_shared(local_state, global_state, chunk)
    }

    fn init_global(init_input: &TableInitInput<Self>) -> VortexResult<Self::GlobalState> {
        let bind_data = init_input.bind_data();
        let projection_expr = extract_projection_expr_from(
            init_input.projection_ids(),
            init_input.column_ids(),
            &bind_data.column_names,
        );
        let filter_expr = extract_table_filter_expr_from(
            init_input.table_filter_set(),
            init_input.column_ids(),
            &bind_data.column_names,
            bind_data.data_source.dtype(),
            &bind_data.filter_exprs,
        )?;

        tracing::trace!(
            "Global init Vortex scan_api SELECT {} WHERE {}",
            &projection_expr,
            filter_expr
                .as_ref()
                .map_or_else(|| "true".to_string(), |f| f.to_string())
        );

        let request = ScanRequest {
            projection: Some(projection_expr),
            filter: filter_expr,
            ..Default::default()
        };

        let scan = bind_data.data_source.scan(request)?;
        let conversion_cache = Arc::new(ConversionCache::new(0));

        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Each split.execute() returns a lazy stream whose early polls do preparation
        // work (expression resolution, layout traversal, first I/O spawns). We use
        // try_flatten_unordered to poll multiple split streams concurrently so that
        // the next split is already warm when the current one finishes.
        let scan_streams = scan.splits().map(move |split_result| {
            let cache = conversion_cache.clone();
            let split = split_result?;
            let s = split.execute()?;
            VortexResult::Ok(s.map(move |r| Ok((r?, cache.clone()))).boxed())
        });

        let iterator = RUNTIME.block_on_stream_thread_safe(move |_| {
            scan_streams.try_flatten_unordered(Some(num_workers * 2))
        });

        Ok(VortexGlobalData {
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

        Ok(VortexLocalData {
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
        Ok(true)
    }

    fn cardinality(bind_data: &Self::BindData) -> Cardinality {
        let est = bind_data.data_source.row_count_estimate();
        match est.upper {
            Some(upper) if upper == est.lower => Cardinality::Maximum(upper),
            Some(upper) => Cardinality::Estimate(upper),
            // When the upper bound is unknown (e.g. deferred files not yet opened), scale the
            // lower bound by the file count to give DuckDB a reasonable cardinality estimate.
            None => Cardinality::Estimate(est.lower.saturating_mul(bind_data.file_count as u64)),
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

        result.push(("Function".to_string(), "Vortex Scan (scan API)".to_string()));

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
