// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This file is a simple C-compatible API that is called from the cudf-test-harness at CI time.

#![allow(clippy::unwrap_used)]

use arrow_schema::DataType;
use arrow_schema::ffi::FFI_ArrowSchema;
use futures::executor::block_on;
use std::sync::LazyLock;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::session::ArraySession;
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

/// External array
#[unsafe(no_mangle)]
pub extern "C" fn export_array(
    schema_ptr: &mut FFI_ArrowSchema,
    array_ptr: &mut ArrowDeviceArray,
) -> i32 {
    let mut ctx = CudaSession::create_execution_ctx(&SESSION).unwrap();

    let primitive = PrimitiveArray::from_iter(0u32..1024);

    *schema_ptr = FFI_ArrowSchema::try_from(DataType::UInt32).unwrap();

    match block_on(primitive.into_array().export_device_array(&mut ctx)) {
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
