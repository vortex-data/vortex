mod hello;

extern crate duckdb;
extern crate duckdb_loadable_macros;
extern crate libduckdb_sys;

use std::error::Error;
use std::sync::OnceLock;

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::{Connection, Result};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use futures::StreamExt;
use futures::stream::BoxStream;
use libduckdb_sys as ffi;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::Mutex;
use vortex_array::stream::ArrayStream;
use vortex_array::{ArrayRef, ToCanonical};
use vortex_duckdb::{ToDuckDBType, to_duckdb_chunk};
use vortex_error::VortexResult;
use vortex_file::executor::{TaskExecutor, TokioExecutor};
use vortex_file::{SplitBy, VortexOpenOptions};
use vortex_io::TokioFile;

#[repr(C)]
struct HelloBindData {
    stream: Mutex<BoxStream<'static, VortexResult<ArrayRef>>>,
    pos: usize,
}

#[repr(C)]
struct HelloInitData {}

pub fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Builder::new_multi_thread().enable_all().build().unwrap())
}

struct HelloVTab;

impl VTab for HelloVTab {
    type InitData = HelloInitData;
    type BindData = HelloBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let path = bind.get_parameter(0).to_string();

        let rt = runtime();
        let exe = TokioExecutor::new(rt.handle().clone());

        let (dtype, stream) = rt.block_on(async {
            let file = TokioFile::open(path).unwrap();
            let vfile = VortexOpenOptions::file(file).open().await?;
            let stream = vfile
                .scan()
                .with_split_by(SplitBy::RowCount(2048))
                .with_task_executor(TaskExecutor::Tokio(exe))
                .with_prefetch_conjuncts(true)
                .with_canonicalize(true)
                .into_array_stream()?;

            let dtype = stream.dtype().clone();

            VortexResult::Ok((dtype, StreamExt::boxed(stream)))
        })?;

        let dtype = dtype.as_struct().unwrap();

        for (name, field) in dtype.names().iter().zip(dtype.fields()) {
            bind.add_result_column(name, field.to_duckdb_type().unwrap());
        }
        Ok(HelloBindData {
            stream: Mutex::new(stream),
            pos: 0,
        })
    }

    fn init(init_info: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        init_info.set_max_threads(8);
        Ok(HelloInitData {})
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let bind_data = func.get_bind_data();
        let rt = runtime();

        let arr = rt.block_on(async {
            let mut stream = bind_data.stream.lock().await;
            stream.next().await
        });

        let Some(arr) = arr else {
            output.set_len(0);
            return Ok(());
        };

        let arr = arr.unwrap();
        let struct_a = arr.to_struct().unwrap();
        let _null = to_duckdb_chunk(&struct_a, output).unwrap();
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

    con.register_table_function::<hello::HelloVTabDict>("dict")
        .expect("Failed to register hello table function");
    Ok(())
}
