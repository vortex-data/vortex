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
    let struct_dtype = vx_struct_fields::into_arc(struct_dtype);
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
        .as_decimal()
        .vortex_expect("not a decimal dtype")
        .precision()
}

/// Return the scale of a decimal data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_decimal_scale(dtype: *const vx_dtype) -> i8 {
    vx_dtype::as_ref(dtype)
        .as_decimal()
        .vortex_expect("not a decimal dtype")
        .scale()
}

/// Return a borrowed reference to the [`vx_struct_fields`] of a struct data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_struct_dtype(
    dtype: *const vx_dtype,
) -> *const vx_struct_fields {
    let struct_dtype = vx_dtype::as_ref(dtype)
        .as_struct()
        .vortex_expect("not a struct dtype");
    vx_struct_fields::new_ref(struct_dtype)
}

/// Return a borrowed reference to the `element` typee of a list data type.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_list_element(dtype: *const vx_dtype) -> *const vx_dtype {
    let element_dtype = vx_dtype::as_ref(dtype)
        .as_list_element()
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
pub unsafe extern "C-unwind" fn vx_dype_is_date(dtype: *const DType) -> bool {
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
mod tests {
    use vortex::dtype::DType;

    use crate::dtype::{
        vx_dtype, vx_dtype_free, vx_dtype_get_variant, vx_dtype_new_bool, vx_dtype_new_primitive,
        vx_dtype_new_utf8, vx_dtype_variant,
    };
    use crate::ptype::vx_ptype;
    use crate::string::vx_string;
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
            let name = vx_struct_fields_field_name(person, 1);
            assert_eq!(vx_string::as_str(name), "age");

            let dtype0 = vx_struct_fields_field_dtype(person, 0);
            let dtype1 = vx_struct_fields_field_dtype(person, 1);
            assert_eq!(vx_dtype_get_variant(dtype0), vx_dtype_variant::DTYPE_UTF8);
            assert_eq!(
                vx_dtype_get_variant(dtype1),
                vx_dtype_variant::DTYPE_PRIMITIVE
            );
            vx_dtype_free(dtype0);
            vx_dtype_free(dtype1);

            vx_struct_fields_free(person);
        }
    }
}
