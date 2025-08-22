// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::{c_int, c_void};
use std::sync::Arc;

use vortex::dtype::datetime::{DATE_ID, TIME_ID, TIMESTAMP_ID, TemporalMetadata};
use vortex::dtype::{DType, DecimalDType};
use vortex::error::{VortexExpect, VortexUnwrap, vortex_panic};

use crate::arc_wrapper;
use crate::ptype::vx_ptype;
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
#[allow(non_camel_case_types)]
#[non_exhaustive]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum vx_dtype_variant {
    /// Null type
    DTYPE_NULL = 0,
    /// Boolean type
    DTYPE_BOOL = 1,
    /// Primitive types (e.g., u8, i16, f32, etc.)
    DTYPE_PRIMITIVE = 2,
    /// Variable-length UTF-8 string type
    DTYPE_UTF8 = 3,
    /// Variable-length binary data type
    DTYPE_BINARY = 4,
    /// Nested struct type
    DTYPE_STRUCT = 5,
    /// Nested list type
    DTYPE_LIST = 6,
    /// User-defined extension type
    DTYPE_EXTENSION = 7,
    /// Decimal type with fixed precision and scale
    DTYPE_DECIMAL = 8,
}

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
            DType::List(..) => vx_dtype_variant::DTYPE_LIST,
            DType::Extension(_) => vx_dtype_variant::DTYPE_EXTENSION,
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

/// Create a new struct data type.
///
/// Takes ownership of the `struct_dtype` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_struct(
    struct_dtype: *const vx_struct_fields,
    is_nullable: bool,
) -> *const vx_dtype {
    let struct_dtype = vx_struct_fields::as_ref(struct_dtype).clone();
    vx_dtype::new(Arc::new(DType::Struct(struct_dtype, is_nullable.into())))
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

/// Return the [`vx_ptype`] of a primitive data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_primitive_ptype(dtype: *const vx_dtype) -> vx_ptype {
    vx_dtype::as_ref(dtype).as_ptype().into()
}

/// Return the precision of a decimal data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_decimal_precision(dtype: *const vx_dtype) -> u8 {
    vx_dtype::as_ref(dtype)
        .as_decimal_opt()
        .vortex_expect("not a decimal dtype")
        .precision()
}

/// Return the scale of a decimal data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_decimal_scale(dtype: *const vx_dtype) -> i8 {
    vx_dtype::as_ref(dtype)
        .as_decimal_opt()
        .vortex_expect("not a decimal dtype")
        .scale()
}

/// Return an owned reference to the [`vx_struct_fields`] of a struct data type.
///
/// The caller is responsible for freeing the returned pointer with [`vx_struct_fields_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_struct_dtype(
    dtype: *const vx_dtype,
) -> *const vx_struct_fields {
    let struct_dtype = vx_dtype::as_ref(dtype)
        .as_struct_opt()
        .vortex_expect("not a struct dtype");
    vx_struct_fields::new_ref(struct_dtype)
}

/// Return the `element` type of a list data type.
///
/// The returned pointer is valid as long as the list dtype is valid.
/// Do NOT free the returned dtype pointer - it shares the lifetime of the list dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_list_element(dtype: *const vx_dtype) -> *const vx_dtype {
    let element_dtype = vx_dtype::as_ref(dtype)
        .as_list_element_opt()
        .vortex_expect("not a list dtype");
    vx_dtype::new_ref(element_dtype)
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_is_time(dtype: *const DType) -> bool {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    match dtype {
        DType::Extension(ext_dtype) => ext_dtype.id() == &*TIME_ID,
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_is_date(dtype: *const DType) -> bool {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    match dtype {
        DType::Extension(ext_dtype) => ext_dtype.id() == &*DATE_ID,
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_is_timestamp(dtype: *const DType) -> bool {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    match dtype {
        DType::Extension(ext_dtype) => ext_dtype.id() == &*TIMESTAMP_ID,
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_time_unit(dtype: *const DType) -> u8 {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    let DType::Extension(ext_dtype) = dtype else {
        vortex_panic!("DType_time_unit: not a time dtype")
    };

    let metadata = ext_dtype.metadata().vortex_expect("time unit metadata");

    metadata.as_ref()[0]
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_time_zone(
    dtype: *const DType,
    dst: *mut c_void,
    len: *mut c_int,
) {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    let DType::Extension(ext_dtype) = dtype else {
        vortex_panic!("vx_dtype_time_unit: not a time dtype")
    };

    match TemporalMetadata::try_from(ext_dtype).vortex_expect("timestamp") {
        TemporalMetadata::Timestamp(_, zone) => {
            if let Some(zone) = zone {
                let bytes = zone.as_bytes();
                let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, bytes.len()) };
                dst.copy_from_slice(bytes);
                unsafe { *len = bytes.len().try_into().vortex_unwrap() };
            } else {
                // No time zone, using local timestamps.
                unsafe { *len = 0 };
            }
        }
        _ => vortex_panic!("DType_time_zone: not a timestamp metadata: {ext_dtype:?}"),
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use vortex::dtype::DType;

    use super::*;
    use crate::dtype::{
        vx_dtype, vx_dtype_free, vx_dtype_get_variant, vx_dtype_new_bool, vx_dtype_new_primitive,
        vx_dtype_new_utf8, vx_dtype_variant,
    };
    use crate::ptype::vx_ptype;
    use crate::string::{vx_string, vx_string_free};
    use crate::struct_fields::{
        vx_struct_fields_builder_add_field, vx_struct_fields_builder_finalize,
        vx_struct_fields_builder_new, vx_struct_fields_field_dtype, vx_struct_fields_field_name,
        vx_struct_fields_free, vx_struct_fields_nfields,
    };

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

            // Free field names (owned references)

            vx_string_free(name);
            vx_string_free(age);

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
        use vortex::dtype::{DType, DecimalDType};

        let dtypes = vec![
            DType::Null,
            DType::Bool(true.into()),
            DType::Primitive(vortex::dtype::PType::I32, false.into()),
            DType::Decimal(DecimalDType::new(10, 2), true.into()),
            DType::Utf8(false.into()),
            DType::Binary(true.into()),
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
                _ => {}
            }
        }
    }

    // Helper function for struct introspection tests
    fn create_test_struct_array() -> vortex::ArrayRef {
        use vortex::IntoArray;
        use vortex::arrays::StructArray;
        use vortex::buffer::Buffer;

        let nums: Buffer<i32> = (0..1000).collect();
        let floats: Buffer<f32> = (0..1000).map(|x| x as f32).collect();

        StructArray::try_from_iter([("nums", nums.into_array()), ("floats", floats.into_array())])
            .unwrap()
            .into_array()
    }

    #[test]
    fn test_struct_introspection_simple() {
        use crate::array::vx_array;
        use crate::struct_fields::vx_struct_fields_nfields;

        let array = create_test_struct_array();
        let vx_arr = vx_array::new(array);
        let dtype_ptr = unsafe { crate::array::vx_array_dtype(vx_arr) };

        let struct_fields_ptr = unsafe { vx_dtype_struct_dtype(dtype_ptr) };
        let n_fields = unsafe { vx_struct_fields_nfields(struct_fields_ptr) };
        assert_eq!(n_fields, 2);

        // Cleanup in reverse order - this is the safest order
        unsafe {
            crate::array::vx_array_free(vx_arr);
        }
    }

    #[test]
    fn test_field_name_access() {
        use crate::array::vx_array;
        use crate::string::{vx_string_free, vx_string_len, vx_string_ptr};
        use crate::struct_fields::vx_struct_fields_field_name;

        let array = create_test_struct_array();
        let vx_arr = vx_array::new(array);
        let dtype_ptr = unsafe { crate::array::vx_array_dtype(vx_arr) };

        let struct_fields_ptr = unsafe { vx_dtype_struct_dtype(dtype_ptr) };

        // Test field name access
        let field_name_ptr = unsafe { vx_struct_fields_field_name(struct_fields_ptr, 0) };
        assert!(!field_name_ptr.is_null());

        let name_len = unsafe { vx_string_len(field_name_ptr) };
        let name_ptr = unsafe { vx_string_ptr(field_name_ptr) };
        let name_slice = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len) };
        let name_str = std::str::from_utf8(name_slice).unwrap();
        assert_eq!(name_str, "nums");

        // Cleanup in careful order
        unsafe {
            vx_string_free(field_name_ptr);
            crate::array::vx_array_free(vx_arr);
        }
    }

    #[test]
    fn test_comprehensive_struct_introspection() {
        use crate::array::vx_array;
        use crate::string::{vx_string_free, vx_string_len, vx_string_ptr};
        use crate::struct_fields::{vx_struct_fields_field_name, vx_struct_fields_nfields};

        let array = create_test_struct_array();
        let vx_arr = vx_array::new(array);
        let dtype_ptr = unsafe { crate::array::vx_array_dtype(vx_arr) };

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
            let name_slice = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len) };
            let name_str = std::str::from_utf8(name_slice).unwrap();

            let expected_name = if i == 0 { "nums" } else { "floats" };
            assert_eq!(name_str, expected_name);

            unsafe {
                vx_string_free(field_name_ptr);
            }
        }

        // Cleanup
        unsafe {
            crate::array::vx_array_free(vx_arr);
        }
    }
}
