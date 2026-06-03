// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! C ABI used by `cudf-test-harness` to export and validate Arrow Device data in CI.

#![expect(clippy::expect_used)]

use std::env;
use std::mem;
use std::panic;
use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Date32Array;
use arrow_array::Decimal128Array;
use arrow_array::StringArray;
use arrow_array::cast::AsArray;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::from_ffi;
use arrow_array::make_array;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::ffi::FFI_ArrowSchema;
use futures::executor::block_on;
use vortex::array::ArrayRef as VortexArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrow::ArrowSessionExt;
use vortex::array::session::ArraySession;
use vortex::array::validity::Validity;
use vortex::dtype::DecimalDType;
use vortex::dtype::FieldNames;
use vortex::dtype::NativePType;
use vortex::extension::datetime::TimeUnit;
use vortex::io::session::RuntimeSession;
use vortex::layout::session::LayoutSession;
use vortex::scalar_fn::session::ScalarFnSession;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::DeviceArrayExt;

const PRIMITIVE_DTYPE_ENV: &str = "VORTEX_CUDF_PRIMITIVE_DTYPE";

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>()
        .with::<CudaSession>()
});

fn primitive_dtype_case() -> String {
    env::var(PRIMITIVE_DTYPE_ENV).unwrap_or_else(|_| "u32".to_string())
}

fn nullable_primitive<T: NativePType>(first: T, second: T, third: T) -> VortexArrayRef {
    PrimitiveArray::from_option_iter([Some(first), None, Some(second), Some(third), None])
        .into_array()
}

fn primitive_array() -> Result<VortexArrayRef, String> {
    Ok(match primitive_dtype_case().as_str() {
        "u8" => nullable_primitive(0u8, 2, 3),
        "u16" => nullable_primitive(10u16, 12, 13),
        "u32" => nullable_primitive(20u32, 22, 23),
        "u64" => nullable_primitive(30u64, 32, 33),
        "i8" => nullable_primitive(-4i8, -2, 3),
        "i16" => nullable_primitive(-14i16, -12, 13),
        "i32" => nullable_primitive(-24i32, -22, 23),
        "i64" => nullable_primitive(-34i64, -32, 33),
        "f32" => nullable_primitive(1.25f32, -2.5, 3.75),
        "f64" => nullable_primitive(10.25f64, -20.5, 30.75),
        other => {
            return Err(format!(
                "unsupported {PRIMITIVE_DTYPE_ENV}={other}; expected one of u8,u16,u32,u64,i8,i16,i32,i64,f32,f64"
            ));
        }
    })
}

fn list_array() -> VortexArrayRef {
    ListArray::try_new(
        PrimitiveArray::from_iter([10i32, 11, 12, 13, 14]).into_array(),
        PrimitiveArray::from_iter([0i32, 2, 2, 5, 5, 5]).into_array(),
        Validity::from_iter([true, false, true, true, false]),
    )
    .expect("list array")
    .into_array()
}

fn fixed_size_list_array() -> VortexArrayRef {
    FixedSizeListArray::new(
        PrimitiveArray::from_iter(20i32..30).into_array(),
        2,
        Validity::from_iter([true, false, true, true, false]),
        5,
    )
    .into_array()
}

fn fixed_size_list_as_list_array() -> VortexArrayRef {
    ListArray::try_new(
        PrimitiveArray::from_iter(20i32..30).into_array(),
        PrimitiveArray::from_iter([0i32, 2, 4, 6, 8, 10]).into_array(),
        Validity::from_iter([true, false, true, true, false]),
    )
    .expect("fixed-size-list as list array")
    .into_array()
}

/// # Safety
/// `schema_ptr` and `array_ptr` must be valid writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn export_array(
    schema_ptr: &mut FFI_ArrowSchema,
    array_ptr: &mut ArrowDeviceArray,
) -> i32 {
    ffi_boundary("export_array", || export_array_inner(schema_ptr, array_ptr))
}

fn export_array_inner(schema_ptr: &mut FFI_ArrowSchema, array_ptr: &mut ArrowDeviceArray) -> i32 {
    let mut ctx = match CudaSession::create_execution_ctx(&SESSION) {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("error creating CUDA execution context: {err}");
            return 1;
        }
    };

    let primitive = match primitive_array() {
        Ok(array) => array,
        Err(err) => {
            eprintln!("error in export_array: {err}");
            return 1;
        }
    };
    let decimal = DecimalArray::from_option_iter(
        [Some(0i128), Some(1), None, Some(3), Some(4)],
        DecimalDType::new(38, 2),
    );
    let strings = VarBinViewArray::from_iter_nullable_str([
        Some("one"),
        None,
        Some("this string is long three"),
        Some("four"),
        None,
    ]);
    let dates = TemporalArray::new_date(
        PrimitiveArray::from_option_iter([Some(100i32), None, Some(300), Some(400), None])
            .into_array(),
        TimeUnit::Days,
    );

    let array = StructArray::new(
        FieldNames::from_iter([
            "prims",
            "decimals",
            "strings",
            "dates",
            "lists",
            "fixed_lists",
        ]),
        vec![
            primitive,
            decimal.into_array(),
            strings.into_array(),
            dates.into_array(),
            list_array(),
            fixed_size_list_array(),
        ],
        5,
        Validity::NonNullable,
    )
    .into_array();

    match block_on(array.export_device_array_with_schema(&mut ctx)) {
        Ok(exported) => {
            *schema_ptr = exported.schema;
            *array_ptr = exported.array;
            0
        }
        Err(err) => {
            eprintln!("error in export_device_array: {err}");
            1
        }
    }
}

/// # Safety
/// `ffi_schema` and `ffi_array` must describe a valid Arrow C Data array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn validate_array(
    ffi_schema: &FFI_ArrowSchema,
    ffi_array: &mut FFI_ArrowArray,
) -> i32 {
    ffi_boundary("validate_array", || {
        validate_array_inner(ffi_schema, ffi_array)
    })
}

fn ffi_boundary(name: &str, f: impl FnOnce() -> i32) -> i32 {
    match panic::catch_unwind(panic::AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(_) => {
            eprintln!("panic in {name}");
            1
        }
    }
}

fn validate_array_inner(ffi_schema: &FFI_ArrowSchema, ffi_array: &mut FFI_ArrowArray) -> i32 {
    // SAFETY: guaranteed by the C ABI contract.
    let array_data = unsafe {
        let ffi_array = mem::replace(ffi_array, FFI_ArrowArray::empty());
        match from_ffi(ffi_array, ffi_schema) {
            Ok(array_data) => array_data,
            Err(err) => {
                eprintln!("from_ffi failed: {err}");
                return 1;
            }
        }
    };

    let array = make_array(array_data);
    let struct_array = array.as_struct();

    let primitive = SESSION
        .arrow()
        .execute_arrow(
            primitive_array().expect("expected primitive array"),
            None,
            &mut SESSION.create_execution_ctx(),
        )
        .expect("expected primitive Arrow array");
    let decimal = Decimal128Array::from_iter([Some(0i128), Some(1), None, Some(3), Some(4)])
        .with_precision_and_scale(38, 2)
        .expect("with_precision_and_scale");
    let string = StringArray::from_iter([
        Some("one"),
        None,
        Some("this string is long three"),
        Some("four"),
        None,
    ]);
    let date = Date32Array::from(vec![Some(100i32), None, Some(300), Some(400), None]);
    let list = SESSION
        .arrow()
        .execute_arrow(list_array(), None, &mut SESSION.create_execution_ctx())
        .expect("expected list Arrow array");
    let fixed_size_list = SESSION
        .arrow()
        .execute_arrow(
            fixed_size_list_as_list_array(),
            None,
            &mut SESSION.create_execution_ctx(),
        )
        .expect("expected fixed-size-list-as-list Arrow array");

    let expected_fields = Fields::from_iter([
        Field::new("prims", primitive.data_type().clone(), true),
        Field::new("decimals", decimal.data_type().clone(), true),
        Field::new("strings", string.data_type().clone(), true),
        Field::new("dates", date.data_type().clone(), true),
        cudf_list_field("lists"),
        cudf_list_field("fixed_lists"),
    ]);
    if &expected_fields != struct_array.fields() {
        eprintln!("wrong fields for host array");
        return 1;
    }

    let expected_arrays: [ArrowArrayRef; 4] = [
        primitive,
        Arc::new(decimal),
        Arc::new(string),
        Arc::new(date),
    ];

    for (idx, (expected, actual)) in expected_arrays
        .iter()
        .zip(struct_array.columns())
        .enumerate()
    {
        if expected.as_ref() != actual.as_ref() {
            eprintln!("wrong values for host column {idx}");
            return 1;
        }
    }

    if !list_values_eq(list.as_ref(), struct_array.column(4).as_ref()) {
        eprintln!("wrong values for lists column");
        return 1;
    }
    if !list_values_eq(fixed_size_list.as_ref(), struct_array.column(5).as_ref()) {
        eprintln!("wrong values for fixed_lists column");
        return 1;
    }

    0
}

fn cudf_list_field(name: &str) -> Field {
    Field::new_list(name, Field::new("element", DataType::Int32, false), true)
}

fn list_values_eq(expected: &dyn Array, actual: &dyn Array) -> bool {
    let expected = expected.as_list::<i32>();
    let actual = actual.as_list::<i32>();

    expected.len() == actual.len()
        && expected.value_offsets() == actual.value_offsets()
        && (0..expected.len()).all(|idx| expected.is_null(idx) == actual.is_null(idx))
        && expected.values().as_ref() == actual.values().as_ref()
}
