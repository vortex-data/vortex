// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This file is a simple C-compatible API that is called from the cudf-test-harness at CI time.
//!
//! The flow is:
//!
//! * test harness calls `dlopen` in this library
//! * invokes the `export_array` function to get back the device array
//! * pass the arrays to `cudf`'s `from_arrow_device_column`
//! * run some operations on the loaded column view
//! * call `array->release()` to drop the data allocated from the Rust side

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::mem;
use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::Decimal128Array;
use arrow_array::UInt32Array;
use arrow_array::cast::AsArray;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::from_ffi;
use arrow_array::make_array;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::ffi::FFI_ArrowSchema;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
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

/// # Safety
/// called by C++ code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn export_array(
    schema_ptr: &mut FFI_ArrowSchema,
    array_ptr: &mut ArrowDeviceArray,
) -> i32 {
    let mut ctx = CudaSession::create_execution_ctx(&SESSION).unwrap();

    let primitive = PrimitiveArray::from_iter(0u32..5);
    let decimal = DecimalArray::from_iter(0i128..5, DecimalDType::new(38, 2));

    let array = StructArray::new(
        FieldNames::from_iter(["prims", "decimals"]),
        vec![primitive.into_array(), decimal.into_array()],
        5,
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

/// # Safety
/// called by C++ code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn validate_array(
    ffi_schema: &FFI_ArrowSchema,
    ffi_array: &mut FFI_ArrowArray,
) -> i32 {
    // SAFETY: the provided pointers must not be null, and must point at valid FFI Arrow types.
    let array_data = unsafe {
        let ffi_array = mem::replace(ffi_array, FFI_ArrowArray::empty());
        from_ffi(ffi_array, ffi_schema).expect("from_ffi failed")
    };

    let array = make_array(array_data);
    let struct_array = array.as_struct();

    let primitive = UInt32Array::from_iter(0..5);
    let decimal = Decimal128Array::from_iter_values(0..5)
        .with_precision_and_scale(38, 2)
        .expect("with_precision_and_scale");

    let expected_fields = Fields::from_iter([
        Field::new("prims", primitive.data_type().clone(), false),
        Field::new("decimals", decimal.data_type().clone(), false),
    ]);

    assert_eq!(
        &expected_fields,
        struct_array.fields(),
        "wrong fields for host array: {:?}",
        struct_array.fields()
    );

    let expected_fields: [ArrayRef; _] = [Arc::new(primitive), Arc::new(decimal)];

    for (expected, actual) in expected_fields.iter().zip(struct_array.columns()) {
        assert_eq!(expected.as_ref(), actual.as_ref());
    }

    0
}
