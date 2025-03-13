use std::error::Error;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use duckdb::core::{
    DataChunkHandle, FlatVector, LogicalTypeHandle, LogicalTypeId, SelectionVector,
};
use duckdb::vtab;
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::Mutex;
use vortex_array::ArrayRef;
use vortex_duckdb::to_duckdb_chunk;
use vortex_error::VortexResult;
use vortex_file::VortexOpenOptions;
use vortex_io::TokioFile;
use vortex_layout::scan::SplitBy;
use vortex_layout::scan::executor::{TaskExecutor, TokioExecutor};

#[repr(C)]
pub struct HelloBindData {}

#[repr(C)]
pub struct HelloInitData {
    done: AtomicBool,
}

pub struct HelloVTabDict;

impl VTab for HelloVTabDict {
    type InitData = HelloInitData;
    type BindData = HelloBindData;

    fn bind(bind: &BindInfo) -> duckdb::Result<Self::BindData, Box<dyn Error>> {
        let path = bind.get_parameter(0).to_string();

        bind.add_result_column("1", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(HelloBindData {})
    }

    fn init(_: &InitInfo) -> duckdb::Result<Self::InitData, Box<dyn Error>> {
        Ok(HelloInitData {
            done: AtomicBool::new(false),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> duckdb::Result<(), Box<dyn Error>> {
        let _bind_data = func.get_bind_data();
        let init_data = func.get_init_data();

        if init_data.done.swap(true, Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }

        let mut vec = output.flat_vector(0);

        let value = vtab::Value::from(5612);
        vec.constant(&value);

        output.set_len(100);

        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}
