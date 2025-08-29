// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use itertools::Itertools;
use num_traits::AsPrimitive;
use tokio::task::block_in_place;
use url::Url;
use vortex::dtype::FieldNames;
use vortex::error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{ExprRef, and, and_collect, col, lit, root, select};
use vortex::file::{VortexFile, VortexOpenOptions};
use vortex::scan::{MultiScan, MultiScanIterator};
use vortex::{ArrayRef, ToCanonical};
use vortex_file::GenericVortexFile;

use crate::RUNTIME;
use crate::convert::{try_from_bound_expression, try_from_table_filter};
use crate::duckdb::footer_cache::FooterCache;
use crate::duckdb::{
    BindInput, BindResult, Cardinality, ClientContext, DataChunk, Expression, LogicalType,
    TableFunction, TableInitInput,
};
use crate::exporter::{ArrayExporter, ConversionCache};
use crate::utils::glob::expand_glob;
use crate::utils::object_store::s3_store;

pub struct VortexBindData {
    first_file: VortexFile,
    filter_exprs: Vec<ExprRef>,
    file_urls: Vec<Url>,
    column_names: Vec<String>,
    column_types: Vec<LogicalType>,
}

impl Clone for VortexBindData {
    /// `VortexBindData` is cloned in case of multiple scan nodes.
    fn clone(&self) -> Self {
        Self {
            first_file: self.first_file.clone(),
            // filter_expr don't need to be cloned as they are consumed once in `init_global`.
            filter_exprs: vec![],
            file_urls: self.file_urls.clone(),
            column_names: self.column_names.clone(),
            column_types: self.column_types.clone(),
        }
    }
}

impl Debug for VortexBindData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexBindData")
            .field("file_urls", &self.file_urls)
            .field("column_names", &self.column_names)
            .field("column_types", &self.column_types)
            .field("filter_expr", &self.filter_exprs)
            .finish()
    }
}

pub struct VortexGlobalData {
    scan: MultiScan<(ArrayRef, Arc<ConversionCache>)>,
    batch_id: AtomicU64,
}

pub struct VortexLocalData {
    iterator: MultiScanIterator<(ArrayRef, Arc<ConversionCache>)>,
    exporter: Option<ArrayExporter>,
    // The unique batch id the of the last chunk exported via scan()
    batch_id: Option<u64>,
}

#[derive(Debug)]
pub struct VortexTableFunction;

/// Extracts the schema from a Vortex file.
fn extract_schema_from_vortex_file(
    file: &VortexFile,
) -> VortexResult<(Vec<String>, Vec<LogicalType>)> {
    let dtype = file.dtype();

    // For now, we assume the top-level type to be a struct.
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

/// Creates a projection expression based on the table initialization input.
fn extract_projection_expr(init: &TableInitInput<VortexTableFunction>) -> ExprRef {
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
    init: &TableInitInput<VortexTableFunction>,
    column_ids: &[u64],
) -> VortexResult<Option<ExprRef>> {
    let table_filter_expr = init
        .table_filter_set()
        .and_then(|filter| {
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
                        init.bind_data().first_file.dtype(),
                    )
                })
                .reduce(|l, r| l?.zip(r?).map(|(l, r)| Ok(and(l, r))).transpose())
        })
        .transpose()?
        .flatten();

    let complex_filter_expr = and_collect(init.bind_data().filter_exprs.clone());
    let filter_expr = complex_filter_expr
        .into_iter()
        .chain(table_filter_expr)
        .reduce(and)
        .unwrap_or_else(|| lit(true));

    Ok(Some(filter_expr))
}

/// Helper function to open a Vortex file from either a local or S3 URL
async fn open_file(
    url: Url,
    options: VortexOpenOptions<GenericVortexFile>,
) -> VortexResult<VortexFile> {
    if url.scheme() == "s3" {
        assert!(url.scheme() == "s3");
        let bucket = url
            .host_str()
            .ok_or_else(|| vortex_err!("Failed to extract bucket name from URL: {url}"))?;

        let path = url
            .path()
            .strip_prefix("/")
            .ok_or_else(|| vortex_err!("Invalid S3 path: {url}"))?;

        options.open_object_store(&s3_store(bucket)?, path).await
    } else {
        let path = url
            .to_file_path()
            .map_err(|_| vortex_err!("Invalid file URL: {url}"))?;

        options.open(path).await
    }
}

impl TableFunction for VortexTableFunction {
    type BindData = VortexBindData;
    type GlobalState = VortexGlobalData;
    type LocalState = VortexLocalData;

    const PROJECTION_PUSHDOWN: bool = true;
    const FILTER_PUSHDOWN: bool = true;
    const FILTER_PRUNE: bool = true;

    /// Input parameter types of the `vortex_scan` table function.
    ///
    // `vortex_scan` takes a single file glob parameter.
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

        let (file_urls, _metadata) = block_in_place(|| {
            RUNTIME.block_on(expand_glob(file_glob_string.as_ref().as_string()))
        })?;

        // The first file is skipped in `create_file_paths_queue`.
        let Some(first_file_url) = file_urls.first() else {
            vortex_bail!("No files matched the glob");
        };

        let footer_cache = FooterCache::new(ctx.object_cache());
        let entry = footer_cache.entry(first_file_url.as_ref());
        let first_file = block_in_place(|| {
            RUNTIME.block_on(async {
                let options = entry.apply_to_file(VortexOpenOptions::file());
                let file = open_file(first_file_url.clone(), options).await?;
                entry.put_if_absent(|| file.footer().clone());
                VortexResult::Ok(file)
            })
        })?;

        let (column_names, column_types) = extract_schema_from_vortex_file(&first_file)?;

        // Add result columns based on the extracted schema.
        for (column_name, column_type) in column_names.iter().zip(&column_types) {
            result.add_result_column(column_name, column_type);
        }

        Ok(VortexBindData {
            file_urls,
            first_file,
            filter_exprs: vec![],
            column_names,
            column_types,
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

                local_state.exporter = Some(ArrayExporter::try_new(
                    &array_result.to_struct()?,
                    &conversion_cache,
                )?);
                // Relaxed since there is no intra-instruction ordering required.
                local_state.batch_id = Some(global_state.batch_id.fetch_add(1, Ordering::Relaxed));
            }

            let exporter = local_state
                .exporter
                .as_mut()
                .vortex_expect("error: exporter missing");

            let has_more_data = exporter.export(chunk)?;

            if !has_more_data {
                // This exporter is fully consumed.
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

        log::trace!(
            "Global init Vortex scan SELECT {} WHERE {}",
            &projection_expr,
            filter_expr
                .as_ref()
                .map_or("true".to_string(), |f| f.to_string())
        );

        let client_context = init_input.client_context()?;
        let object_cache = client_context.object_cache();

        let closures =
            bind_data
                .file_urls
                .clone()
                .into_iter()
                .enumerate()
                .map(move |(idx, path)| {
                    let first_file = bind_data.first_file.clone();
                    let filter_expr = filter_expr.clone();
                    let projection_expr = projection_expr.clone();
                    let conversion_cache = Arc::new(ConversionCache::new(idx as u64));
                    let object_cache = object_cache;

                    move || {
                        let file = if idx == 0 {
                            // The first path from `file_paths` is skipped as
                            // the first file was already opened during bind.
                            first_file
                        } else {
                            let cache = FooterCache::new(object_cache);
                            let entry = cache.entry(path.as_ref());
                            block_in_place(|| {
                                RUNTIME.block_on(async {
                                    let options = entry.apply_to_file(VortexOpenOptions::file());
                                    let file = open_file(path.clone(), options).await?;
                                    entry.put_if_absent(|| file.footer().clone());
                                    VortexResult::Ok(file)
                                })
                            })?
                        };

                        if let Some(ref filter) = filter_expr
                            && file.can_prune(filter)?
                        {
                            return Ok(vec![]);
                        };

                        file.scan()?
                            .with_some_filter(filter_expr)
                            .with_projection(projection_expr)
                            .map(move |split| Ok((split, conversion_cache.clone())))
                            .build()
                    }
                });

        Ok(VortexGlobalData {
            scan: MultiScan::new(closures),
            batch_id: AtomicU64::new(0),
        })
    }

    fn init_local(
        _init: &TableInitInput<Self>,
        global: &mut Self::GlobalState,
    ) -> VortexResult<Self::LocalState> {
        Ok(VortexLocalData {
            iterator: global.scan.clone().new_iterator(),
            exporter: None,
            batch_id: None,
        })
    }

    fn pushdown_complex_filter(
        bind_data: &mut Self::BindData,
        expr: &Expression,
    ) -> VortexResult<bool> {
        let Some(expr) = try_from_bound_expression(expr)? else {
            return Ok(false);
        };
        bind_data.filter_exprs.push(expr);
        Ok(true)
    }

    fn cardinality(bind_data: &Self::BindData) -> Cardinality {
        if bind_data.file_urls.len() == 1 {
            Cardinality::Maximum(bind_data.first_file.row_count())
        } else {
            // This is the same behavior as DuckDB's Parquet extension, although we could
            // test multiplying the row count by the number of files.
            Cardinality::Estimate(
                max(bind_data.first_file.row_count(), 1) * bind_data.file_urls.len() as u64,
            )
        }
    }

    fn partition_data(
        _bind_data: &Self::BindData,
        _global_init_data: &mut Self::GlobalState,
        _local_init_data: &mut Self::LocalState,
    ) -> VortexResult<u64> {
        _local_init_data
            .batch_id
            .ok_or_else(|| vortex_err!("batch id missing, no batches exported"))
    }

    fn to_string(bind_data: &Self::BindData) -> Option<Vec<(String, String)>> {
        let mut result = Vec::new();

        // Add function name
        result.push(("Function".to_string(), "Vortex Scan".to_string()));

        // Add file information
        if !bind_data.file_urls.is_empty() {
            result.push(("Files".to_string(), bind_data.file_urls.len().to_string()));
        }

        // Add filter information
        if !bind_data.filter_exprs.is_empty() {
            let mut filters = bind_data.filter_exprs.iter().map(|f| format!("{}", f));
            result.push(("Filters".to_string(), filters.join(" /\\\n")));
        }
        // NOTE: Projection is already printed by the planner.

        Some(result)
    }
}
