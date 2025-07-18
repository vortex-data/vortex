// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};

use bitvec::macros::internal::funty::Fundamental;
use vortex::dtype::FieldNames;
use vortex::error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{ExprRef, and, and_collect, lit, root, select};
use vortex::file::{VortexFile, VortexOpenOptions};
use vortex::scan::{MultiScan, MultiScanIterator};
use vortex::{ArrayRef, ToCanonical};

use crate::convert::{try_from_bound_expression, try_from_table_filter};
use crate::duckdb::{
    BindInput, BindResult, DataChunk, Expression, LogicalType, TableFunction, TableInitInput,
};
use crate::exporter::{ArrayExporter, ConversionCache};

pub struct VortexBindData {
    first_file: VortexFile,
    filter_exprs: Vec<ExprRef>,
    file_paths: Vec<PathBuf>,
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
            file_paths: self.file_paths.clone(),
            column_names: self.column_names.clone(),
            column_types: self.column_types.clone(),
        }
    }
}

impl std::fmt::Debug for VortexBindData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexBindData")
            .field("file_paths", &self.file_paths)
            .field("column_names", &self.column_names)
            .field("column_types", &self.column_types)
            .field("filter_expr", &self.filter_exprs)
            .finish()
    }
}

pub struct VortexGlobalData {
    multi_scan: MultiScan<(ArrayRef, Arc<ConversionCache>)>,
}

pub struct VortexLocalData {
    multi_scan_iterator: MultiScanIterator<(ArrayRef, Arc<ConversionCache>)>,
    exporter: Option<ArrayExporter>,
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
        .as_struct()
        .ok_or_else(|| vortex_err!("Vortex file must contain a struct array at the top level"))?;

    let mut column_names = Vec::new();
    let mut column_types = Vec::new();

    for (field_name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
        let logical_type = LogicalType::try_from(&field_dtype)
            .map_err(|e| vortex_err!("Failed to convert field '{}' type: {}", field_name, e))?;

        column_names.push(field_name.to_string());
        column_types.push(logical_type);
    }

    Ok((column_names, column_types))
}

/// Creates a projection expression based on the table initialization input.
fn extract_projection_expr(init: &TableInitInput<VortexTableFunction>) -> ExprRef {
    let projection_ids = init.projection_ids().unwrap_or(&[]);
    let column_ids = init.column_ids();

    let projected_ids = projection_ids.iter().map(|p| column_ids[p.as_usize()]);
    select(
        projected_ids
            .map(|idx| {
                init.bind_data()
                    .column_names
                    .get(idx.as_usize())
                    .vortex_expect("prune idx in column names")
            })
            .map(|s| Arc::from(s.clone()))
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
                    let name = init
                        .bind_data()
                        .column_names
                        .get(column_ids[idx.as_usize()].as_usize())
                        .vortex_expect("exists");
                    try_from_table_filter(&ex, name)
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

    fn bind(input: &BindInput, result: &mut BindResult) -> VortexResult<Self::BindData> {
        let file_glob_string = input
            .get_parameter(0)
            .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

        let paths = match glob::glob(&file_glob_string.as_string()) {
            Ok(paths) => paths,
            Err(e) => vortex_bail!("Failed to glob files: {}", e),
        };

        let file_paths: Vec<_> = paths
            .collect::<Result<_, _>>()
            .map_err(|e| vortex_err!("Failed to glob files: {}", e))?;

        // The first file is skipped in `create_file_paths_queue`.
        let first_file = VortexOpenOptions::file()
            .open_blocking(&file_paths[0])
            .map_err(|e| vortex_err!("Failed to open Vortex file: {}", e))?;

        let (column_names, column_types) = extract_schema_from_vortex_file(&first_file)?;

        // Add result columns based on the extracted schema.
        for (column_name, column_type) in column_names.iter().zip(&column_types) {
            result.add_result_column(column_name, column_type);
        }

        Ok(VortexBindData {
            file_paths,
            first_file,
            column_names,
            column_types,
            filter_exprs: vec![],
        })
    }

    fn scan(
        _bind_data: &Self::BindData,
        local_state: &mut Self::LocalState,
        _global_state: &mut Self::GlobalState,
        chunk: &mut DataChunk,
    ) -> VortexResult<()> {
        loop {
            if local_state.exporter.is_none() {
                let Some(result) = local_state.multi_scan_iterator.next() else {
                    return Ok(());
                };

                let (array_result, conversion_cache) = result?;

                local_state.exporter = Some(ArrayExporter::try_new(
                    &array_result.to_struct()?,
                    &conversion_cache,
                )?);
            }

            let exporter = local_state
                .exporter
                .as_mut()
                .vortex_expect("error: exporter missing");

            let has_more_data = exporter.export(chunk)?;

            if !has_more_data {
                // This exporter is fully consumed.
                local_state.exporter = None;
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

        // Atomic as the closures can be called from different threads.
        let is_first_file = Arc::new(AtomicBool::new(false));
        let cache_id = Arc::new(AtomicU64::new(0));

        let closures = bind_data.file_paths.clone().into_iter().map(move |path| {
            let first_file = bind_data.first_file.clone();
            let filter_expr = filter_expr.clone();
            let projection_expr = projection_expr.clone();
            let is_first_file = is_first_file.clone();
            let cache_id = cache_id.clone();
            let conversion_cache = Arc::new(ConversionCache::new(cache_id.fetch_add(1, SeqCst)));

            move || {
                let file = if !is_first_file.swap(true, SeqCst) {
                    // The first path from `file_paths` is skipped as
                    // the first file was already opened during bind.
                    first_file
                } else {
                    VortexOpenOptions::file()
                        .open_blocking(&path)
                        .vortex_expect("Failed to open Vortex file")
                };

                file.scan()
                    .vortex_expect("Failed to create scan builder")
                    .with_some_filter(filter_expr)
                    .with_projection(projection_expr)
                    .map(move |split| Ok((split, conversion_cache.clone())))
            }
        });

        Ok(VortexGlobalData {
            multi_scan: MultiScan::new().with_scan_builders(closures),
        })
    }

    fn init_local(
        _init: &TableInitInput<Self>,
        global: &mut Self::GlobalState,
    ) -> VortexResult<Self::LocalState> {
        Ok(VortexLocalData {
            multi_scan_iterator: global.multi_scan.new_scan_iterator(),
            exporter: None,
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
}
