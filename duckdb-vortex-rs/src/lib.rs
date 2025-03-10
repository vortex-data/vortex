extern crate duckdb;
extern crate duckdb_loadable_macros;
extern crate libduckdb_sys;

use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::{Connection, Result};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use libduckdb_sys as ffi;
use vortex_array::arrays::{BoolArray, PrimitiveArray, StructArray, VarBinArray};
use vortex_array::validity::Validity;
use vortex_array::variants::StructArrayTrait;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::buffer;
use vortex_dtype::{DType, FieldNames, Nullability};
use vortex_duckdb::{to_duckdb_chunk, ToDuckDBType};

#[repr(C)]
struct HelloBindData {
    dtype: DType,
}

#[repr(C)]
struct HelloInitData {
    done: AtomicBool,
}

struct HelloVTab;

impl HelloVTab {
    fn data() -> ArrayRef {
        let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::NonNullable);
        let ys = VarBinArray::from_vec(
            vec!["a", "b", "c", "d", "e"],
            DType::Utf8(Nullability::NonNullable),
        );
        let zs = BoolArray::from_iter([true, true, true, false, false]);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xxs".into(), "ys".into(), "zs".into()]),
            vec![xs.into_array(), ys.into_array(), zs.into_array()],
            5,
            Validity::NonNullable,
        )
        .unwrap();
        struct_a.to_array()
    }
}

impl VTab for HelloVTab {
    type InitData = HelloInitData;
    type BindData = HelloBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let data = Self::data().to_struct().unwrap();

        for (name, field) in data.names().iter().zip(data.fields()) {
            bind.add_result_column(name, field.dtype().to_duckdb_type().unwrap());
        }
        Ok(HelloBindData {
            dtype: data.dtype().clone(),
        })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(HelloInitData {
            done: AtomicBool::new(false),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let _bind_data = func.get_bind_data();
        if init_data.done.swap(true, Ordering::Relaxed) {
            output.set_len(0);
        } else {
            let arr = Self::data();

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
