use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

use vortex::error::{VortexExpect, VortexResult, vortex_err};

use crate::duckdb::{
    BindInput, BindResult, DataChunk, Expression, LogicalType, TableFunction, TableInitInput,
};

pub struct HelloBindData {
    name: String,
}

pub struct HelloInitData {
    done: AtomicBool,
}

#[derive(Debug)]
pub struct HelloTableFunction;

impl TableFunction for HelloTableFunction {
    type BindData = HelloBindData;
    type InitGlobalData = HelloInitData;
    type InitLocalData = ();

    fn parameters() -> Vec<LogicalType> {
        vec![LogicalType::varchar()]
    }

    fn bind(input: &BindInput, result: &mut BindResult) -> VortexResult<Self::BindData> {
        let value = input
            .get_parameter(0)
            .ok_or_else(|| vortex_err!("Missing parameter for hello function"))?;

        result.add_result_column("greeting", &LogicalType::varchar());

        Ok(HelloBindData {
            name: value.as_string().to_string_lossy().to_string(),
        })
    }

    fn function(
        bind_data: &Self::BindData,
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
        let value = CString::new(format!("Hello {}", bind_data.name))
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
