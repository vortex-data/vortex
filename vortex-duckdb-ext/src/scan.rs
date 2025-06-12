use std::sync::atomic::AtomicBool;

use vortex::error::{VortexResult, vortex_bail, vortex_err};
use vortex::file::{VortexFile, VortexOpenOptions};

use crate::duckdb::{
    BindInput, BindResult, DataChunk, Expression, LogicalType, TableFunction, TableInitInput,
};
use crate::exporter::ArrayIteratorExporter;

pub struct VortexBindData {
    first_file: VortexFile,
    _column_names: Vec<String>,
    _column_types: Vec<LogicalType>,
}

pub struct VortexGlobalData {
    done: AtomicBool,
}

pub struct VortexLocalData {
    exporter: Option<ArrayIteratorExporter>,
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

impl TableFunction for VortexTableFunction {
    type BindData = VortexBindData;
    type GlobalState = VortexGlobalData;
    type LocalState = VortexLocalData;

    /// Input parameter types of the `vortex_scan` table function.
    ///
    // `vortex_scan` takes a single file glob parameter.
    fn parameters() -> Vec<LogicalType> {
        vec![LogicalType::varchar()]
    }

    fn bind(input: &BindInput, result: &mut BindResult) -> VortexResult<Self::BindData> {
        // TODO: expand glob & assign to file list
        let file_glob_string = input
            .get_parameter(0)
            .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

        let file_path: String = file_glob_string.as_string().to_string_lossy().into_owned();
        let first_file = VortexOpenOptions::file()
            .open_blocking(&file_path)
            .map_err(|e| vortex_err!("Failed to open Vortex file: {}", e))?;

        let (column_names, column_types) = extract_schema_from_vortex_file(&first_file)?;

        // Add result columns based on the extracted schema.
        for (name, logical_type) in column_names.iter().zip(&column_types) {
            result.add_result_column(name, logical_type);
        }

        Ok(VortexBindData {
            first_file,
            _column_names: column_names,
            _column_types: column_types,
        })
    }

    fn scan(
        bind_data: &Self::BindData,
        local_state: &mut Self::LocalState,
        global_state: &mut Self::GlobalState,
        chunk: &mut DataChunk,
    ) -> VortexResult<()> {
        if global_state.done.load(std::sync::atomic::Ordering::SeqCst) {
            // Signal to DuckDB that there's no work left by setting the chunk length to 0.
            chunk.set_len(0);
            return Ok(());
        }

        if local_state.exporter.is_none() {
            let array_iter = bind_data
                .first_file
                .scan()?
                .into_array_iter()
                .map_err(|e| vortex_err!("Failed to create array iterator: {}", e))?;

            local_state.exporter = Some(ArrayIteratorExporter::new(Box::new(array_iter)));
        }

        let Some(ref mut exporter) = local_state.exporter else {
            vortex_bail!("ArrayIteratorExporter is not set")
        };

        let is_data_left_to_scan = exporter
            .export(chunk)
            .map_err(|e| vortex_err!("Failed to export data: {}", e))?;

        if !is_data_left_to_scan {
            global_state
                .done
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }

        Ok(())
    }

    fn init_global(_init: &TableInitInput<Self>) -> VortexResult<Self::GlobalState> {
        Ok(VortexGlobalData {
            done: AtomicBool::new(false),
        })
    }

    fn init_local(
        _init: &TableInitInput<Self>,
        _global: &mut Self::GlobalState,
    ) -> VortexResult<Self::LocalState> {
        Ok(VortexLocalData { exporter: None })
    }

    fn pushdown_complex_filter(
        _bind_data: &mut Self::BindData,
        _expr: &Expression,
    ) -> VortexResult<bool> {
        Ok(false)
    }
}
