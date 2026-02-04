// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This file is a simple C-compatible API that is called from the cudf-test-harness at CI time.
//!
//! The flow is
//!
//!     * test harness calls `dlopen` in this library
//!     * invokes the `export_array` function to get back the device array
//!     * pass the arrays to `cudf`'s `from_arrow_device_column`
//!     * run some operations on the loaded column view
//!     * call `array->release()` to drop the data allocated from the Rust side

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::LazyLock;

use arrow_schema::ffi::FFI_ArrowSchema;
use futures::executor::block_on;
use vortex::array::Array;
use vortex::array::IntoArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::session::ArraySession;
use vortex::array::validity::Validity;
use vortex::dtype::DecimalDType;
use vortex::dtype::FieldNames;
use vortex::expr::session::ExprSession;
use vortex::io::session::RuntimeSession;
use vortex::layout::session::LayoutSession;
use vortex::metrics::VortexMetrics;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::DeviceArrayExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<VortexMetrics>()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>()
        .with::<CudaSession>()
});

#[unsafe(no_mangle)]
pub extern "C" fn export_array(
    schema_ptr: &mut FFI_ArrowSchema,
    array_ptr: &mut ArrowDeviceArray,
) -> i32 {
    let mut ctx = CudaSession::create_execution_ctx(&SESSION).unwrap();

    let primitive = PrimitiveArray::from_iter(0u32..1024);
    let string =
        VarBinViewArray::from_iter_str((0..1024).map(|idx| format!("this is string {idx}")));
    let decimal = DecimalArray::from_iter(0i64..1024, DecimalDType::new(19, 2));

    let array = StructArray::new(
        FieldNames::from_iter(["prims", "strings", "decimals"]),
        vec![
            primitive.into_array(),
            string.into_array(),
            decimal.into_array(),
        ],
        1024,
        Validity::NonNullable,
    )
    .into_array();

    let data_type = array
        .dtype()
        .to_arrow_dtype()
        .expect("converting schema to Arrow DataType");

    *schema_ptr = FFI_ArrowSchema::try_from(data_type).expect("data_type to FFI_ArrowSchema");

    match block_on(array.export_device_array(&mut ctx)) {
        Ok(exported) => {
            *array_ptr = exported;
            0
        }
        Err(err) => {
            eprintln!("error in export_device_array: {err}");
            1
        }
    }
}
