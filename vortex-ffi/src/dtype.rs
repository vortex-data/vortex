// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_int;
use std::ptr;
use std::sync::Arc;

use arrow_array::ffi::FFI_ArrowSchema;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::error::VortexExpect;
use vortex::error::vortex_panic;
use vortex::extension::datetime::AnyTemporal;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::Time;
use vortex::extension::datetime::Timestamp;

use crate::arc_wrapper;
use crate::error::try_or;
use crate::error::vx_error;
use crate::ptype::vx_ptype;
use crate::string::vx_string;
use crate::struct_fields::vx_struct_fields;

arc_wrapper!(
    /// A Vortex data type.
    ///
    /// Data types in Vortex are purely logical, meaning they confer no information about how the data
    /// is physically stored.
    DType,
    vx_dtype
);

/// The variant tag for a Vortex data type.
#[expect(non_camel_case_types)]
#[non_exhaustive]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum vx_dtype_variant {
    /// Null type.
    DTYPE_NULL = 0,
    /// Boolean type.
    DTYPE_BOOL = 1,
    /// Primitive types (e.g., u8, i16, f32, etc.).
    DTYPE_PRIMITIVE = 2,
    /// Variable-length UTF-8 string type.
    DTYPE_UTF8 = 3,
    /// Variable-length binary data type.
    DTYPE_BINARY = 4,
    /// Nested struct type.
    DTYPE_STRUCT = 5,
    /// Nested list type.
    DTYPE_LIST = 6,
    /// User-defined extension type.
    DTYPE_EXTENSION = 7,
    /// Decimal type with fixed precision and scale.
    DTYPE_DECIMAL = 8,
    /// Nested fixed-size list type.
    DTYPE_FIXED_SIZE_LIST = 9,
}

// TODO(connor)[Union]: Do we need to add union and variant here?
impl From<&DType> for vx_dtype_variant {
    fn from(value: &DType) -> Self {
        match value {
            DType::Null => vx_dtype_variant::DTYPE_NULL,
            DType::Bool(_) => vx_dtype_variant::DTYPE_BOOL,
            DType::Primitive(..) => vx_dtype_variant::DTYPE_PRIMITIVE,
            DType::Decimal(..) => vx_dtype_variant::DTYPE_DECIMAL,
            DType::Utf8(_) => vx_dtype_variant::DTYPE_UTF8,
            DType::Binary(_) => vx_dtype_variant::DTYPE_BINARY,
            DType::Struct(..) => vx_dtype_variant::DTYPE_STRUCT,
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::List(..) => vx_dtype_variant::DTYPE_LIST,
            DType::FixedSizeList(..) => vx_dtype_variant::DTYPE_FIXED_SIZE_LIST,
            DType::Extension(_) => vx_dtype_variant::DTYPE_EXTENSION,
            DType::Variant(_) => vortex_panic!("Variant DType is not supported in FFI yet"),
        }
    }
}

/// Create a new null data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_null() -> *const vx_dtype {
    vx_dtype::new(Arc::new(DType::Null))
}

/// Create a new boolean data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_bool(is_nullable: bool) -> *const vx_dtype {
    vx_dtype::new(Arc::new(DType::Bool(is_nullable.into())))
}

/// Create a new primitive data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_primitive(
    ptype: vx_ptype,
    is_nullable: bool,
) -> *const vx_dtype {
    vx_dtype::new(Arc::new(DType::Primitive(ptype.into(), is_nullable.into())))
}

/// Create a new variable length UTF-8 data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_utf8(is_nullable: bool) -> *const vx_dtype {
    vx_dtype::new(Arc::new(DType::Utf8(is_nullable.into())))
}

/// Create a new variable length binary data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_binary(is_nullable: bool) -> *const vx_dtype {
    vx_dtype::new(Arc::new(DType::Binary(is_nullable.into())))
}

/// Create a new list data type.
///
/// Takes ownership of the `element` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_list(
    element: *const vx_dtype,
    is_nullable: bool,
) -> *const vx_dtype {
    let element = vx_dtype::into_arc(element);
    vx_dtype::new(Arc::new(DType::List(element, is_nullable.into())))
}

/// Create a new fixed-size list data type.
///
/// Takes ownership of the `element` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_fixed_size_list(
    element: *const vx_dtype,
    size: u32,
    is_nullable: bool,
) -> *const vx_dtype {
    let element = vx_dtype::into_arc(element);
    vx_dtype::new(Arc::new(DType::FixedSizeList(
        element,
        size,
        is_nullable.into(),
    )))
}

/// Create a new struct data type.
///
/// Takes ownership of the `struct_dtype` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_struct(
    struct_dtype: *mut vx_struct_fields,
    is_nullable: bool,
) -> *const vx_dtype {
    let struct_dtype = vx_struct_fields::into_box(struct_dtype);
    vx_dtype::new(Arc::new(DType::Struct(*struct_dtype, is_nullable.into())))
}

/// Create a new decimal data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_decimal(
    precision: u8,
    scale: i8,
    is_nullable: bool,
) -> *const vx_dtype {
    vx_dtype::new(Arc::new(DType::Decimal(
        DecimalDType::new(precision, scale),
        is_nullable.into(),
    )))
}

/// Get the variant of a [`vx_dtype`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_get_variant(dtype: *const vx_dtype) -> vx_dtype_variant {
    vx_dtype::as_ref(dtype).into()
}

/// Return whether the given [`vx_dtype`] is nullable.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_is_nullable(dtype: *const vx_dtype) -> bool {
    vx_dtype::as_ref(dtype).is_nullable()
}

/// Returns the [`vx_ptype`] of a primitive.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_primitive_ptype(dtype: *const vx_dtype) -> vx_ptype {
    vx_dtype::as_ref(dtype).as_ptype().into()
}

/// Returns the precision of a decimal.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_decimal_precision(dtype: *const vx_dtype) -> u8 {
    // TODO(joe): propagate this error up instead of expecting
    vx_dtype::as_ref(dtype)
        .as_decimal_opt()
        .vortex_expect("not a decimal dtype")
        .precision()
}

/// Returns the scale of a decimal.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_decimal_scale(dtype: *const vx_dtype) -> i8 {
    // TODO(joe): propagate this error up instead of expecting
    vx_dtype::as_ref(dtype)
        .as_decimal_opt()
        .vortex_expect("not a decimal dtype")
        .scale()
}

/// Return a borrowed reference to the [`vx_struct_fields`] of a struct.
///
/// The returned pointer is valid as long as the struct dtype is valid.
/// Do NOT free the returned pointer - it shares the lifetime of the struct dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_struct_dtype(
    dtype: *const vx_dtype,
) -> *const vx_struct_fields {
    let Some(struct_dtype) = vx_dtype::as_ref(dtype).as_struct_fields_opt() else {
        return ptr::null();
    };
    vx_struct_fields::new_ref(struct_dtype)
}

/// Returns the element type of a list.
///
/// The returned pointer is valid as long as the list dtype is valid.
/// Do NOT free the returned dtype pointer - it shares the lifetime of the list dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_list_element(dtype: *const vx_dtype) -> *const vx_dtype {
    let Some(element_dtype) = vx_dtype::as_ref(dtype).as_list_element_opt() else {
        return ptr::null();
    };
    vx_dtype::new_ref(element_dtype)
}

/// Returns the element type of a fixed-size list.
///
/// The returned pointer is valid as long as the fixed-size list dtype is valid.
/// Do NOT free the returned dtype pointer - it shares the lifetime of the fixed-size list dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_fixed_size_list_element(
    dtype: *const vx_dtype,
) -> *const vx_dtype {
    // TODO(joe): propagate this error up instead of expecting
    let element_dtype = vx_dtype::as_ref(dtype)
        .as_fixed_size_list_element_opt()
        .vortex_expect("not a fixed-size list dtype");
    vx_dtype::new_ref(element_dtype)
}

/// Returns the size of a fixed-size list.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_fixed_size_list_size(dtype: *const vx_dtype) -> u32 {
    let dtype_ref = vx_dtype::as_ref(dtype);
    match dtype_ref {
        DType::FixedSizeList(_, size, _) => *size,
        _ => vortex_panic!("not a fixed-size list dtype"),
    }
}

/// Checks if the type is time.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_is_time(dtype: *const DType) -> bool {
    // TODO(joe): propagate this error up instead of expecting
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    match dtype {
        DType::Extension(ext_dtype) => ext_dtype.is::<Time>(),
        _ => false,
    }
}

/// Checks if the type is a date.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_is_date(dtype: *const DType) -> bool {
    // TODO(joe): propagate this error up instead of expecting
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    match dtype {
        DType::Extension(ext_dtype) => ext_dtype.is::<Date>(),
        _ => false,
    }
}

/// Checks if the type is a timestamp.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_is_timestamp(dtype: *const DType) -> bool {
    // TODO(joe): propagate this error up instead of expecting
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    match dtype {
        DType::Extension(ext_dtype) => ext_dtype.is::<Timestamp>(),
        _ => false,
    }
}

/// Returns the time unit, assuming the type is time.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_time_unit(dtype: *const DType) -> u8 {
    // TODO(joe): propagate this error up instead of expecting
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    let DType::Extension(ext_dtype) = dtype else {
        vortex_panic!("DType_time_unit: not a time dtype")
    };

    let Some(opts) = ext_dtype.metadata_opt::<AnyTemporal>() else {
        // TODO(ngates): propagate this error up instead of expecting
        vortex_panic!("DType_time_unit: not a temporal metadata: {ext_dtype:?}")
    };
    opts.time_unit().into()
}

/// Returns the time zone, assuming the type is time. Caller is responsible for freeing the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_time_zone(dtype: *const DType) -> *const vx_string {
    // TODO(joe): propagate this error up instead of expecting
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    let DType::Extension(ext_dtype) = dtype else {
        vortex_panic!("vx_dtype_time_unit: not a time dtype")
    };

    let Some(opts) = ext_dtype.metadata_opt::<Timestamp>() else {
        // TODO(joe): propagate this error up instead of expecting
        vortex_panic!("DType_time_zone: not a timestamp: {ext_dtype:?}")
    };

    match opts.tz.as_ref() {
        Some(zone) => vx_string::new(Arc::clone(zone)),
        None => ptr::null(),
    }
}

/// Convert a dtype to ArrowSchema.
/// You can use the dtype after conversion
/// On success, returns 0. On error, sets err and returns 1.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_to_arrow_schema(
    dtype: *const vx_dtype,
    schema: *mut FFI_ArrowSchema,
    err: *mut *mut vx_error,
) -> c_int {
    try_or(err, 1, || {
        let dtype = vx_dtype::as_ref(dtype);
        let arrow_schema = dtype.to_arrow_schema()?;
        let arrow_schema = FFI_ArrowSchema::try_from(&arrow_schema)?;
        unsafe { ptr::write(schema, arrow_schema) };
        Ok(0)
    })
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use std::slice;

    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::arrays::StructArray;
    use vortex::buffer::Buffer;
    use vortex::dtype::DType;
    use vortex::dtype::DecimalDType;

    use super::*;
    use crate::array::vx_array;
    use crate::array::vx_array_dtype;
    use crate::array::vx_array_free;
    use crate::dtype::vx_dtype;
    use crate::dtype::vx_dtype_free;
    use crate::dtype::vx_dtype_get_variant;
    use crate::dtype::vx_dtype_new_bool;
    use crate::dtype::vx_dtype_new_primitive;
    use crate::dtype::vx_dtype_new_utf8;
    use crate::dtype::vx_dtype_variant;
    use crate::ptype::vx_ptype;
    use crate::string::vx_string;
    use crate::string::vx_string_len;
    use crate::string::vx_string_ptr;
    use crate::struct_fields::vx_struct_fields_builder_add_field;
    use crate::struct_fields::vx_struct_fields_builder_finalize;
    use crate::struct_fields::vx_struct_fields_builder_new;
    use crate::struct_fields::vx_struct_fields_field_dtype;
    use crate::struct_fields::vx_struct_fields_field_name;
    use crate::struct_fields::vx_struct_fields_free;
    use crate::struct_fields::vx_struct_fields_nfields;

    #[test]
    fn test_simple() {
        unsafe {
            let ffi_dtype = vx_dtype_new_bool(true);

            // functions check
            assert_eq!(
                vx_dtype_get_variant(ffi_dtype),
                vx_dtype_variant::DTYPE_BOOL
            );

            // Field access checks.
            assert_eq!(vx_dtype::as_ref(ffi_dtype), &DType::Bool(true.into()));

            // Free the memory.
            vx_dtype_free(ffi_dtype);
        }
    }

    #[test]
    fn test_struct() {
        unsafe {
            let builder = vx_struct_fields_builder_new();
            vx_struct_fields_builder_add_field(
                builder,
                vx_string::new("name".into()),
                vx_dtype_new_utf8(false),
            );
            vx_struct_fields_builder_add_field(
                builder,
                vx_string::new("age".into()),
                vx_dtype_new_primitive(vx_ptype::PTYPE_U8, true),
            );
            let person = vx_struct_fields_builder_finalize(builder);
            assert_eq!(vx_struct_fields_nfields(person), 2);

            let name = vx_struct_fields_field_name(person, 0);
            assert_eq!(vx_string::as_str(name), "name");
            let age = vx_struct_fields_field_name(person, 1);
            assert_eq!(vx_string::as_str(age), "age");

            let dtype0 = vx_struct_fields_field_dtype(person, 0);
            let dtype1 = vx_struct_fields_field_dtype(person, 1);
            assert_eq!(vx_dtype_get_variant(dtype0), vx_dtype_variant::DTYPE_UTF8);
            assert_eq!(
                vx_dtype_get_variant(dtype1),
                vx_dtype_variant::DTYPE_PRIMITIVE
            );

            // Field names are now borrowed references - do not free them

            // Free field dtypes (owned references)
            vx_dtype_free(dtype0);
            vx_dtype_free(dtype1);

            // Free struct fields
            vx_struct_fields_free(person);
        }
    }

    #[test]
    fn test_dtype_null() {
        unsafe {
            let null_dtype = vx_dtype_new_null();
            assert_eq!(
                vx_dtype_get_variant(null_dtype),
                vx_dtype_variant::DTYPE_NULL
            );
            // Null dtype is always nullable
            assert!(vx_dtype_is_nullable(null_dtype));
            vx_dtype_free(null_dtype);
        }
    }

    #[test]
    fn test_dtype_binary() {
        unsafe {
            let binary_dtype = vx_dtype_new_binary(true);
            assert_eq!(
                vx_dtype_get_variant(binary_dtype),
                vx_dtype_variant::DTYPE_BINARY
            );
            assert!(vx_dtype_is_nullable(binary_dtype));
            vx_dtype_free(binary_dtype);

            let non_nullable = vx_dtype_new_binary(false);
            assert!(!vx_dtype_is_nullable(non_nullable));
            vx_dtype_free(non_nullable);
        }
    }

    #[test]
    fn test_dtype_list() {
        unsafe {
            let element_dtype = vx_dtype_new_primitive(vx_ptype::PTYPE_I32, false);
            let list_dtype = vx_dtype_new_list(element_dtype, true);

            assert_eq!(
                vx_dtype_get_variant(list_dtype),
                vx_dtype_variant::DTYPE_LIST
            );
            assert!(vx_dtype_is_nullable(list_dtype));

            let element = vx_dtype_list_element(list_dtype);
            assert_eq!(
                vx_dtype_get_variant(element),
                vx_dtype_variant::DTYPE_PRIMITIVE
            );
            assert_eq!(vx_dtype_primitive_ptype(element), vx_ptype::PTYPE_I32);

            vx_dtype_free(list_dtype);
        }
    }

    #[test]
    fn test_dtype_fixed_size_list() {
        unsafe {
            let element_dtype = vx_dtype_new_primitive(vx_ptype::PTYPE_F64, false);
            let fsl_dtype = vx_dtype_new_fixed_size_list(element_dtype, 3, true);

            assert_eq!(
                vx_dtype_get_variant(fsl_dtype),
                vx_dtype_variant::DTYPE_FIXED_SIZE_LIST
            );
            assert!(vx_dtype_is_nullable(fsl_dtype));

            // Test element accessor
            let element = vx_dtype_fixed_size_list_element(fsl_dtype);
            assert_eq!(
                vx_dtype_get_variant(element),
                vx_dtype_variant::DTYPE_PRIMITIVE
            );
            assert_eq!(vx_dtype_primitive_ptype(element), vx_ptype::PTYPE_F64);

            // Test size accessor
            let size = vx_dtype_fixed_size_list_size(fsl_dtype);
            assert_eq!(size, 3);

            vx_dtype_free(fsl_dtype);
        }
    }

    #[test]
    fn test_dtype_fixed_size_list_non_nullable() {
        unsafe {
            let element_dtype = vx_dtype_new_utf8(true);
            let fsl_dtype = vx_dtype_new_fixed_size_list(element_dtype, 10, false);

            assert_eq!(
                vx_dtype_get_variant(fsl_dtype),
                vx_dtype_variant::DTYPE_FIXED_SIZE_LIST
            );
            assert!(!vx_dtype_is_nullable(fsl_dtype));

            let element = vx_dtype_fixed_size_list_element(fsl_dtype);
            assert_eq!(vx_dtype_get_variant(element), vx_dtype_variant::DTYPE_UTF8);
            assert!(vx_dtype_is_nullable(element));

            let size = vx_dtype_fixed_size_list_size(fsl_dtype);
            assert_eq!(size, 10);

            vx_dtype_free(fsl_dtype);
        }
    }

    #[test]
    fn test_nested_fixed_size_lists() {
        unsafe {
            // Create inner fixed-size list: FSL<i32>[5]
            let inner_element = vx_dtype_new_primitive(vx_ptype::PTYPE_I32, false);
            let inner_fsl = vx_dtype_new_fixed_size_list(inner_element, 5, false);

            // Create outer fixed-size list: FSL<FSL<i32>[5]>[3]
            let outer_fsl = vx_dtype_new_fixed_size_list(inner_fsl, 3, true);

            assert_eq!(
                vx_dtype_get_variant(outer_fsl),
                vx_dtype_variant::DTYPE_FIXED_SIZE_LIST
            );
            assert!(vx_dtype_is_nullable(outer_fsl));
            assert_eq!(vx_dtype_fixed_size_list_size(outer_fsl), 3);

            // Check inner FSL
            let inner = vx_dtype_fixed_size_list_element(outer_fsl);
            assert_eq!(
                vx_dtype_get_variant(inner),
                vx_dtype_variant::DTYPE_FIXED_SIZE_LIST
            );
            assert!(!vx_dtype_is_nullable(inner));
            assert_eq!(vx_dtype_fixed_size_list_size(inner), 5);

            // Check innermost element
            let innermost = vx_dtype_fixed_size_list_element(inner);
            assert_eq!(
                vx_dtype_get_variant(innermost),
                vx_dtype_variant::DTYPE_PRIMITIVE
            );
            assert_eq!(vx_dtype_primitive_ptype(innermost), vx_ptype::PTYPE_I32);

            vx_dtype_free(outer_fsl);
        }
    }

    #[test]
    fn test_dtype_decimal() {
        unsafe {
            let decimal_dtype = vx_dtype_new_decimal(10, 2, true);
            assert_eq!(
                vx_dtype_get_variant(decimal_dtype),
                vx_dtype_variant::DTYPE_DECIMAL
            );
            assert!(vx_dtype_is_nullable(decimal_dtype));
            assert_eq!(vx_dtype_decimal_precision(decimal_dtype), 10);
            assert_eq!(vx_dtype_decimal_scale(decimal_dtype), 2);
            vx_dtype_free(decimal_dtype);

            let non_nullable = vx_dtype_new_decimal(18, -3, false);
            assert!(!vx_dtype_is_nullable(non_nullable));
            assert_eq!(vx_dtype_decimal_precision(non_nullable), 18);
            assert_eq!(vx_dtype_decimal_scale(non_nullable), -3);
            vx_dtype_free(non_nullable);
        }
    }

    #[test]
    fn test_dtype_primitive_ptype() {
        unsafe {
            let u8_dtype = vx_dtype_new_primitive(vx_ptype::PTYPE_U8, false);
            assert_eq!(vx_dtype_primitive_ptype(u8_dtype), vx_ptype::PTYPE_U8);
            vx_dtype_free(u8_dtype);

            let f64_dtype = vx_dtype_new_primitive(vx_ptype::PTYPE_F64, true);
            assert_eq!(vx_dtype_primitive_ptype(f64_dtype), vx_ptype::PTYPE_F64);
            vx_dtype_free(f64_dtype);
        }
    }

    #[test]
    fn test_dtype_variant_conversion() {
        // Important: Verifies the From trait implementation for FFI variant enum
        // These mappings are part of the ABI contract

        let dtypes = vec![
            DType::Null,
            DType::Bool(true.into()),
            DType::Primitive(vortex::dtype::PType::I32, false.into()),
            DType::Decimal(DecimalDType::new(10, 2), true.into()),
            DType::Utf8(false.into()),
            DType::Binary(true.into()),
            DType::FixedSizeList(
                Arc::new(DType::Primitive(vortex::dtype::PType::U8, false.into())),
                4,
                true.into(),
            ),
        ];

        for dtype in dtypes {
            let variant: vx_dtype_variant = (&dtype).into();
            match dtype {
                DType::Null => assert_eq!(variant, vx_dtype_variant::DTYPE_NULL),
                DType::Bool(_) => assert_eq!(variant, vx_dtype_variant::DTYPE_BOOL),
                DType::Primitive(..) => assert_eq!(variant, vx_dtype_variant::DTYPE_PRIMITIVE),
                DType::Decimal(..) => assert_eq!(variant, vx_dtype_variant::DTYPE_DECIMAL),
                DType::Utf8(_) => assert_eq!(variant, vx_dtype_variant::DTYPE_UTF8),
                DType::Binary(_) => assert_eq!(variant, vx_dtype_variant::DTYPE_BINARY),
                DType::FixedSizeList(..) => {
                    assert_eq!(variant, vx_dtype_variant::DTYPE_FIXED_SIZE_LIST)
                }
                _ => {}
            }
        }
    }

    // Helper function for struct introspection tests
    fn create_test_struct_array() -> ArrayRef {
        let nums: Buffer<i32> = (0..1000).collect();
        let floats: Buffer<f32> = (0..1000).map(|x| x as f32).collect();

        StructArray::try_from_iter([("nums", nums.into_array()), ("floats", floats.into_array())])
            .unwrap()
            .into_array()
    }

    // TODO: re-enable under miri once parking_lot_core fixes strict-provenance violations
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_struct_introspection_simple() {
        let array = create_test_struct_array();
        let vx_arr = vx_array::new(Arc::new(array));
        let dtype_ptr = unsafe { vx_array_dtype(vx_arr) };

        let struct_fields_ptr = unsafe { vx_dtype_struct_dtype(dtype_ptr) };
        let n_fields = unsafe { vx_struct_fields_nfields(struct_fields_ptr) };
        assert_eq!(n_fields, 2);

        // Cleanup in reverse order - this is the safest order
        unsafe {
            vx_array_free(vx_arr);
        }
    }

    // TODO: re-enable under miri once parking_lot_core fixes strict-provenance violations
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_field_name_access() {
        let array = create_test_struct_array();
        let vx_arr = vx_array::new(Arc::new(array));
        let dtype_ptr = unsafe { vx_array_dtype(vx_arr) };

        let struct_fields_ptr = unsafe { vx_dtype_struct_dtype(dtype_ptr) };

        // Test field name access
        let field_name_ptr = unsafe { vx_struct_fields_field_name(struct_fields_ptr, 0) };
        assert!(!field_name_ptr.is_null());

        let name_len = unsafe { vx_string_len(field_name_ptr) };
        let name_ptr = unsafe { vx_string_ptr(field_name_ptr) };
        let name_slice = unsafe { slice::from_raw_parts(name_ptr.cast::<u8>(), name_len) };
        let name_str = str::from_utf8(name_slice).unwrap();
        assert_eq!(name_str, "nums");

        // Cleanup in careful order
        unsafe {
            // Field name is now a borrowed reference - do not free it
            vx_array_free(vx_arr);
        }
    }

    // TODO: re-enable under miri once parking_lot_core fixes strict-provenance violations
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_comprehensive_struct_introspection() {
        let array = create_test_struct_array();
        let vx_arr = vx_array::new(Arc::new(array));
        let dtype_ptr = unsafe { vx_array_dtype(vx_arr) };

        let struct_fields_ptr = unsafe { vx_dtype_struct_dtype(dtype_ptr) };
        let n_fields = unsafe { vx_struct_fields_nfields(struct_fields_ptr) };
        assert_eq!(n_fields, 2);

        // Test both field names
        for i in 0..n_fields {
            let field_name_ptr =
                unsafe { vx_struct_fields_field_name(struct_fields_ptr, i as usize) };
            assert!(!field_name_ptr.is_null());

            let name_len = unsafe { vx_string_len(field_name_ptr) };
            let name_ptr = unsafe { vx_string_ptr(field_name_ptr) };
            let name_slice = unsafe { slice::from_raw_parts(name_ptr.cast::<u8>(), name_len) };
            let name_str = str::from_utf8(name_slice).unwrap();

            let expected_name = if i == 0 { "nums" } else { "floats" };
            assert_eq!(name_str, expected_name);

            // Field name is now a borrowed reference - do not free it
        }

        // Cleanup
        unsafe {
            vx_array_free(vx_arr);
        }
    }
}
