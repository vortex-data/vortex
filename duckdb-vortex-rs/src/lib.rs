extern crate duckdb;
extern crate duckdb_loadable_macros;
extern crate libduckdb_sys;

use std::cmp::min;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::{Connection, Result};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use libduckdb_sys as ffi;
use tokio::runtime::Builder;
use vortex_array::arrays::{BoolArray, PrimitiveArray, StructArray, VarBinArray};
use vortex_array::compute::slice;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::buffer;
use vortex_dtype::{DType, FieldNames, Nullability};
use vortex_duckdb::{ToDuckDBType, to_duckdb_chunk};
use vortex_file::VortexOpenOptions;
use vortex_io::TokioFile;

#[repr(C)]
struct HelloBindData {
    stream: ArrayRef,
    pos: usize,
}

#[repr(C)]
struct HelloInitData {
    position: AtomicUsize,
}

struct HelloVTab;

impl VTab for HelloVTab {
    type InitData = HelloInitData;
    type BindData = HelloBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let path = bind.get_parameter(0).to_string();

        let rt = Builder::new_current_thread().build().unwrap();

        let stream = rt.block_on(async {
            let file = TokioFile::open(path).unwrap();
            let vfile = VortexOpenOptions::file(file).open().await?;

            let stream = vfile.scan().into_array().await;
            stream
        })?;

        let dtype = stream.dtype().as_struct().unwrap();

        for (name, field) in dtype.names().iter().zip(dtype.fields()) {
            bind.add_result_column(name, field.to_duckdb_type().unwrap());
        }
        Ok(HelloBindData { stream, pos: 0 })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(HelloInitData {
            position: AtomicUsize::new(0),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();
        if init_data.position.load(Ordering::SeqCst) >= bind_data.stream.len() {
            output.set_len(0);
        } else {
            let arr = &bind_data.stream;

            let pos = init_data.position.load(Ordering::SeqCst);
            let next_pos = min(pos + 2048, arr.len());
            let arr = slice(arr, pos, next_pos).unwrap();
            init_data.position.store(next_pos, Ordering::SeqCst);

            let struct_a = arr.to_struct().unwrap();
            let _null = to_duckdb_chunk(&struct_a, output).unwrap();
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

const EXTENSION_NAME: &str = env!("CARGO_PKG_NAME");

#[duckdb_entrypoint_c_api()]
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_table_function::<HelloVTab>(EXTENSION_NAME)
        .expect("Failed to register hello table function");
    Ok(())
}
