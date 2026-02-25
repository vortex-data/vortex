// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::path::Path;
use std::sync::Arc;

use futures::StreamExt;
use futures::TryStreamExt;
use itertools::Itertools;
use num_traits::AsPrimitive;
use url::Url;
use vortex::dtype::FieldNames;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::col;
use vortex::expr::root;
use vortex::expr::select;
use vortex::expr::stats::Precision;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::runtime::BlockingRuntime;
use vortex::scan::api::DataSourceRef;
use vortex::scan::api::ScanRequest;
use vortex_utils::aliases::hash_set::HashSet;

use crate::RUNTIME;
use crate::SESSION;
use crate::convert::try_from_bound_expression;
use crate::convert::try_from_table_filter;
use crate::duckdb::BindInputRef;
use crate::duckdb::BindResultRef;
use crate::duckdb::Cardinality;
use crate::duckdb::ClientContextRef;
use crate::duckdb::DataChunkRef;
use crate::duckdb::ExpressionRef;
use crate::duckdb::LogicalType;
use crate::duckdb::TableFunction;
use crate::duckdb::TableInitInput;
use crate::duckdb::VirtualColumnsResultRef;
use crate::exporter::ConversionCache;
use crate::filesystem::resolve_filesystem;
use crate::scan::EMPTY_COLUMN_IDX;
use crate::scan::EMPTY_COLUMN_NAME;
use crate::scan::VortexGlobalData;
use crate::scan::VortexLocalData;
use crate::scan::extract_schema_from_dtype;
use crate::scan::init_local_shared;

/// Bind data for the scan API table function, holding a [`DataSourceRef`] instead of
/// per-file URLs.
pub struct VortexScanApiBindData {
    data_source: DataSourceRef,
    filter_exprs: Vec<Expression>,
    column_names: Vec<String>,
    column_types: Vec<LogicalType>,
}

impl Clone for VortexScanApiBindData {
    fn clone(&self) -> Self {
        Self {
            data_source: self.data_source.clone(),
            // filter_exprs are consumed once in `init_global`.
            filter_exprs: vec![],
            column_names: self.column_names.clone(),
            column_types: self.column_types.clone(),
        }
    }
}

impl Debug for VortexScanApiBindData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexScanApiBindData")
            .field("column_names", &self.column_names)
            .field("column_types", &self.column_types)
            .field(
                "filter_exprs",
                &self
                    .filter_exprs
                    .iter()
                    .map(|e| e.to_string())
                    .collect_vec(),
            )
            .finish()
    }
}

#[derive(Debug)]
pub struct VortexScanApiTableFunction;

/// Creates a projection expression from the table initialization input.
fn extract_projection_expr(init: &TableInitInput<VortexScanApiTableFunction>) -> Expression {
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

/// Creates a table filter expression from the table filter set.
fn extract_table_filter_expr(
    init: &TableInitInput<VortexScanApiTableFunction>,
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
                    ex,
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
        ctx: &ClientContextRef,
        input: &BindInputRef,
        result: &mut BindResultRef,
    ) -> VortexResult<Self::BindData> {
        let glob_url_parameter = input
            .get_parameter(0)
            .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

        // Parse the URL and separate the base URL (keep scheme, host, etc.) from the path.
        let glob_url_str = glob_url_parameter.as_string();
        let glob_url = match Url::parse(glob_url_str.as_str()) {
            Ok(url) => Ok(url),
            Err(_) => Url::from_file_path(Path::new(glob_url_str.as_str()))
                .map_err(|_| vortex_err!("Neither URL nor path: '{}' ", glob_url_str.as_str())),
        }?;

        let mut base_url = glob_url.clone();
        base_url.set_path("");

        let fs = resolve_filesystem(&base_url, ctx)?;

        let data_source: DataSourceRef = RUNTIME.block_on(async {
            let builder = MultiFileDataSource::new(SESSION.clone())
                .with_filesystem(fs)
                .with_glob(glob_url.path());
            let ds = builder.build().await?;
            VortexResult::Ok(Arc::new(ds))
        })?;

        let (column_names, column_types) = extract_schema_from_dtype(data_source.dtype())?;

        for (column_name, column_type) in column_names.iter().zip(&column_types) {
            result.add_result_column(column_name, column_type);
        }

        Ok(VortexScanApiBindData {
            data_source,
            filter_exprs: vec![],
            column_names,
            column_types,
        })
    }

    fn scan(
        _client_context: &ClientContextRef,
        _bind_data: &Self::BindData,
        local_state: &mut Self::LocalState,
        global_state: &mut Self::GlobalState,
        chunk: &mut DataChunkRef,
    ) -> VortexResult<()> {
        global_state.scan(local_state, chunk)
    }

    fn init_global(init_input: &TableInitInput<Self>) -> VortexResult<Self::GlobalState> {
        let bind_data = init_input.bind_data();
        let projection_expr = extract_projection_expr(init_input);
        let filter_expr = extract_table_filter_expr(init_input, init_input.column_ids())?;

        tracing::debug!(
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

        let scan = RUNTIME.block_on(bind_data.data_source.scan(request))?;
        let conversion_cache = Arc::new(ConversionCache::new(0));

        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Each split.execute() returns a lazy stream whose early polls do preparation
        // work (expression resolution, layout traversal, first I/O spawns). We use
        // try_flatten_unordered to poll multiple split streams concurrently so that
        // the next split is already warm when the current one finishes.
        let scan_streams = scan.partitions().map(move |split_result| {
            let cache = conversion_cache.clone();
            let split = split_result?;
            let s = split.execute()?;
            VortexResult::Ok(s.map(move |r| Ok((r?, cache.clone()))).boxed())
        });

        let iterator = RUNTIME.block_on_stream_thread_safe(move |_| {
            scan_streams.try_flatten_unordered(Some(num_workers * 2))
        });

        Ok(VortexGlobalData::new(iterator))
    }

    fn init_local(
        _init: &TableInitInput<Self>,
        global: &mut Self::GlobalState,
    ) -> VortexResult<Self::LocalState> {
        init_local_shared(global)
    }

    fn table_scan_progress(
        _client_context: &ClientContextRef,
        _bind_data: &mut Self::BindData,
        global_state: &mut Self::GlobalState,
    ) -> f64 {
        global_state.progress()
    }

    fn pushdown_complex_filter(
        bind_data: &mut Self::BindData,
        expr: &ExpressionRef,
    ) -> VortexResult<bool> {
        tracing::debug!("Attempting to push down filter expression: {expr}");
        let Some(expr) = try_from_bound_expression(expr)? else {
            return Ok(false);
        };
        bind_data.filter_exprs.push(expr);

        // NOTE(ngates): Vortex does indeed run exact filters, so in theory we should return `true`
        //  here to tell DuckDB we've handled the filter. However, DuckDB applies some crude
        //  cardinality estimation heuristics (e.g. an equality filter => 20% selectivity) that
        //  means by returning false, DuckDB runs an additional filter (a little bit of overhead)
        //  but tends to end up with a better query plan.
        //  If we plumb row count estimation into the layout tree, perhaps we could use zone maps
        //  etc. to return estimates. But this function is probably called too late anyway. Maybe
        //  we need our own cardinality heuristics.
        Ok(false)
    }

    fn cardinality(bind_data: &Self::BindData) -> Cardinality {
        match bind_data.data_source.row_count() {
            Some(Precision::Exact(v)) => Cardinality::Maximum(v),
            Some(Precision::Inexact(v)) => Cardinality::Estimate(v),
            None => Cardinality::Unknown,
        }
    }

    fn partition_data(
        _bind_data: &Self::BindData,
        _global_init_data: &mut Self::GlobalState,
        local_init_data: &mut Self::LocalState,
    ) -> VortexResult<u64> {
        VortexGlobalData::partition_data(local_init_data)
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

    fn virtual_columns(_bind_data: &Self::BindData, result: &mut VirtualColumnsResultRef) {
        result.register(EMPTY_COLUMN_IDX, EMPTY_COLUMN_NAME, &LogicalType::bool());
    }
}
