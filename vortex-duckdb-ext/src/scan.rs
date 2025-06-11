use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex::file::{VortexFile, VortexOpenOptions};

use crate::duckdb::{
    BindInput, BindResult, DataChunk, Expression, LogicalType, TableFunction, TableInitInput,
};

pub struct VortexBindData {
    _column_names: Vec<String>,
    _column_types: Vec<LogicalType>,
}

pub struct HelloInitData {
    done: AtomicBool,
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
    type InitGlobalData = HelloInitData;
    type InitLocalData = ();

    fn parameters() -> Vec<LogicalType> {
        vec![LogicalType::varchar()]
    }

    // TODO: expand glob, assign to file list

    fn bind(input: &BindInput, result: &mut BindResult) -> VortexResult<Self::BindData> {
        let file_glob_string = input
            .get_parameter(0)
            .ok_or_else(|| vortex_err!("Missing parameter for hello function"))?;

        let file_path: String = file_glob_string.as_string().to_string_lossy().into_owned();
        let first_file = VortexOpenOptions::file()
            .open_blocking(file_path)
            .map_err(|e| vortex_err!("Failed to open Vortex file: {}", e))?;

        let (column_names, column_types) = extract_schema_from_vortex_file(&first_file)?;

        // Add result columns based on the extracted schema.
        for (name, logical_type) in column_names.iter().zip(&column_types) {
            result.add_result_column(name, logical_type);
        }

        Ok(VortexBindData {
            _column_names: column_names,
            _column_types: column_types,
        })
    }

    fn function(
        _bind_data: &Self::BindData,
        _local_init: &mut Self::InitLocalData,
        global_init: &mut Self::InitGlobalData,
        chunk: &mut DataChunk,
    ) -> VortexResult<()> {
        // Check if the function has already been executed
        if global_init.done.swap(true, Ordering::Relaxed) {
            chunk.set_len(0);
            return Ok(());
        }

        chunk.set_len(1);
        let value = CString::new(format!("Hello {}", "hello"))
            .map_err(|_| vortex_err!("Name contains null byte"))
            .vortex_expect("Failed to create CString from greeting");
        chunk.get_vector(0).assign_string_element(0, &value);

        Ok(())
    }

    fn init_global(_init: &TableInitInput<Self>) -> VortexResult<Self::InitGlobalData> {
        Ok(HelloInitData {
            done: AtomicBool::new(false),
        })
    }

    fn init_local(
        _init: &TableInitInput<Self>,
        _global: &mut Self::InitGlobalData,
    ) -> VortexResult<Self::InitLocalData> {
        Ok(())
    }

    fn pushdown_complex_filter(
        _bind_data: &mut Self::BindData,
        _expr: &Expression,
    ) -> VortexResult<bool> {
        Ok(false)
    }
}
