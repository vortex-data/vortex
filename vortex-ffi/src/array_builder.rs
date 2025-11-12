// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FFI interface for building Vortex arrays.

use std::slice;

use vortex::IntoArray;
use vortex::arrays::{DecimalArray, NullArray, PrimitiveArray, StructArray};
use vortex::buffer::BufferMut;
use vortex::builders::{ArrayBuilder, BoolBuilder, VarBinViewBuilder};
use vortex::dtype::half::f16;
use vortex::dtype::{DType, DecimalDType, Nullability};
use vortex::error::vortex_err;
use vortex::validity::Validity;

use crate::array::vx_array;
use crate::box_wrapper;
use crate::dtype::vx_dtype;
use crate::error::{try_or_default, vx_error};

// =============================================================================
// Primitive Array Builders
// =============================================================================

macro_rules! ffi_primitive_array_new {
    ($ptype:ident, $rust_type:ty) => {
        paste::paste! {
            #[doc = "Create a new `" $ptype "` primitive array from raw data.\n\n"]
            #[doc = "The `data` pointer must point to a valid array of `len` elements.\n"]
            #[doc = "The `validity` pointer, if not null, must point to a valid array of `len` booleans.\n"]
            #[doc = "If `validity` is null, the array is assumed to have no null values.\n\n"]
            #[doc = "Returns a new array, or null on error."]
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_primitive_new_ $ptype>](
                data: *const $rust_type,
                len: usize,
                validity: *const bool,
                error_out: *mut *mut vx_error,
            ) -> *const vx_array {
                try_or_default(error_out, || {
                    if data.is_null() {
                        return Err(vortex_err!("data pointer is null"));
                    }

                    let data_slice = unsafe { slice::from_raw_parts(data, len) };
                    let mut buffer = BufferMut::<$rust_type>::with_capacity(len);
                    buffer.extend_from_slice(data_slice);

                    let validity = if validity.is_null() {
                        Validity::NonNullable
                    } else {
                        let validity_slice = unsafe { slice::from_raw_parts(validity, len) };
                        Validity::from_iter(validity_slice.iter().copied())
                    };

                    let array = PrimitiveArray::new(buffer.freeze(), validity).into_array();
                    Ok(vx_array::new(array))
                })
            }
        }
    };
}

ffi_primitive_array_new!(u8, u8);
ffi_primitive_array_new!(u16, u16);
ffi_primitive_array_new!(u32, u32);
ffi_primitive_array_new!(u64, u64);
ffi_primitive_array_new!(i8, i8);
ffi_primitive_array_new!(i16, i16);
ffi_primitive_array_new!(i32, i32);
ffi_primitive_array_new!(i64, i64);
ffi_primitive_array_new!(f16, f16);
ffi_primitive_array_new!(f32, f32);
ffi_primitive_array_new!(f64, f64);

// =============================================================================
// Bool Array Builder
// =============================================================================

/// Create a new boolean array from raw data.
///
/// The `data` pointer must point to a valid array of `len` booleans.
/// The `validity` pointer, if not null, must point to a valid array of `len` booleans.
/// If `validity` is null, the array is assumed to have no null values.
///
/// Returns a new array, or null on error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_bool_new(
    data: *const bool,
    len: usize,
    validity: *const bool,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error_out, || {
        if data.is_null() {
            return Err(vortex_err!("data pointer is null"));
        }

        let data_slice = unsafe { slice::from_raw_parts(data, len) };

        let nullability = if validity.is_null() {
            Nullability::NonNullable
        } else {
            Nullability::Nullable
        };

        let mut builder = BoolBuilder::with_capacity(nullability, len);

        for (i, &value) in data_slice.iter().enumerate() {
            if !validity.is_null() {
                let validity_slice = unsafe { slice::from_raw_parts(validity, len) };
                if !validity_slice[i] {
                    builder.append_null();
                    continue;
                }
            }
            builder.append_value(value);
        }

        let array = builder.finish().into_array();
        Ok(vx_array::new(array))
    })
}

// =============================================================================
// Decimal Array Builder
// =============================================================================

/// Create a new decimal array from i128 values.
///
/// The `data` pointer must point to a valid array of `len` i128 elements.
/// The `validity` pointer, if not null, must point to a valid array of `len` booleans.
/// If `validity` is null, the array is assumed to have no null values.
///
/// The `precision` must be between 1 and 38 for Decimal128.
/// The `scale` must be between 0 and `precision`.
///
/// Returns a new array, or null on error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_decimal128_new(
    data: *const i128,
    len: usize,
    precision: u8,
    scale: i8,
    validity: *const bool,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error_out, || {
        if data.is_null() {
            return Err(vortex_err!("data pointer is null"));
        }

        let data_slice = unsafe { slice::from_raw_parts(data, len) };
        let mut buffer = BufferMut::<i128>::with_capacity(len);
        buffer.extend_from_slice(data_slice);

        let validity = if validity.is_null() {
            Validity::NonNullable
        } else {
            let validity_slice = unsafe { slice::from_raw_parts(validity, len) };
            Validity::from_iter(validity_slice.iter().copied())
        };

        let decimal_dtype = DecimalDType::try_new(precision, scale)?;
        let array =
            DecimalArray::new::<i128>(buffer.freeze(), decimal_dtype, validity).into_array();
        Ok(vx_array::new(array))
    })
}

// =============================================================================
// Null Array
// =============================================================================

/// Create a new null array with the given length.
///
/// All values in a null array are null by definition.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_null_new(len: usize) -> *const vx_array {
    vx_array::new(NullArray::new(len).into_array())
}

// =============================================================================
// VarBinView Builder (UTF8 and Binary)
// =============================================================================

pub(crate) struct VarBinViewBuilderWrapper {
    builder: VarBinViewBuilder,
}

box_wrapper!(
    /// Builder for creating UTF8 or Binary arrays.
    VarBinViewBuilderWrapper,
    vx_varbinview_builder
);

/// Create a new UTF8 array builder.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_utf8_builder_new(
    nullable: bool,
) -> *mut vx_varbinview_builder {
    let nullability = if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let builder = VarBinViewBuilder::with_capacity(DType::Utf8(nullability), 0);
    vx_varbinview_builder::new(Box::new(VarBinViewBuilderWrapper { builder }))
}

/// Create a new Binary array builder.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_binary_builder_new(
    nullable: bool,
) -> *mut vx_varbinview_builder {
    let nullability = if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let builder = VarBinViewBuilder::with_capacity(DType::Binary(nullability), 0);
    vx_varbinview_builder::new(Box::new(VarBinViewBuilderWrapper { builder }))
}

/// Append a UTF8 string to the builder.
///
/// The `value` pointer must point to a valid UTF-8 string of `len` bytes.
/// This function takes ownership of neither the builder nor the string data.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_varbinview_builder_append_utf8(
    builder: *mut vx_varbinview_builder,
    value: *const u8,
    len: usize,
    error_out: *mut *mut vx_error,
) {
    try_or_default(error_out, || {
        if value.is_null() && len > 0 {
            return Err(vortex_err!("value pointer is null but len > 0"));
        }

        let builder_wrapper = vx_varbinview_builder::as_mut(builder);
        let value_slice = unsafe { slice::from_raw_parts(value, len) };
        let utf8_str =
            std::str::from_utf8(value_slice).map_err(|e| vortex_err!("Invalid UTF-8: {}", e))?;
        builder_wrapper.builder.append_value(utf8_str);
        Ok(())
    })
}

/// Append a binary value to the builder.
///
/// The `value` pointer must point to a valid array of `len` bytes.
/// This function takes ownership of neither the builder nor the binary data.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_varbinview_builder_append_binary(
    builder: *mut vx_varbinview_builder,
    value: *const u8,
    len: usize,
    error_out: *mut *mut vx_error,
) {
    try_or_default(error_out, || {
        if value.is_null() && len > 0 {
            return Err(vortex_err!("value pointer is null but len > 0"));
        }

        let builder_wrapper = vx_varbinview_builder::as_mut(builder);
        let value_slice = unsafe { slice::from_raw_parts(value, len) };
        builder_wrapper.builder.append_value(value_slice);
        Ok(())
    })
}

/// Append a null value to the builder.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_varbinview_builder_append_null(
    builder: *mut vx_varbinview_builder,
) {
    let builder_wrapper = vx_varbinview_builder::as_mut(builder);
    builder_wrapper.builder.append_null();
}

/// Finish building and return the array.
///
/// Takes ownership of the builder.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_varbinview_builder_finish(
    builder: *mut vx_varbinview_builder,
) -> *const vx_array {
    let mut builder_wrapper = vx_varbinview_builder::into_box(builder);
    let array = builder_wrapper.builder.finish().into_array();
    vx_array::new(array)
}

// =============================================================================
// Struct Array Builder
// =============================================================================

/// Create a new struct array from field arrays.
///
/// The `dtype` must be a struct dtype created with `vx_dtype_struct`.
/// The `field_arrays` pointer must point to an array of `n_fields` array pointers.
/// The `len` parameter specifies the length of each field array.
/// The `validity` pointer, if not null, must point to a valid array of `len` booleans.
///
/// This function does NOT take ownership of the field arrays.
/// Returns a new array, or null on error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_struct_new(
    dtype: *const vx_dtype,
    field_arrays: *const *const vx_array,
    n_fields: usize,
    len: usize,
    validity: *const bool,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error_out, || {
        if field_arrays.is_null() {
            return Err(vortex_err!("field_arrays pointer is null"));
        }

        let dtype_ref = vx_dtype::as_ref(dtype);
        let struct_fields = match dtype_ref {
            DType::Struct(fields, _) => fields,
            _ => return Err(vortex_err!("dtype is not a struct")),
        };

        if struct_fields.nfields() != n_fields {
            return Err(vortex_err!(
                "dtype has {} fields but {} field arrays provided",
                struct_fields.nfields(),
                n_fields
            ));
        }

        let field_array_ptrs = unsafe { slice::from_raw_parts(field_arrays, n_fields) };
        let fields: Vec<_> = field_array_ptrs
            .iter()
            .map(|ptr| vx_array::as_ref(*ptr).clone())
            .collect();

        let validity = if validity.is_null() {
            Validity::NonNullable
        } else {
            let validity_slice = unsafe { slice::from_raw_parts(validity, len) };
            Validity::from_iter(validity_slice.iter().copied())
        };

        let array = StructArray::try_new(struct_fields.names().clone(), fields, len, validity)?
            .into_array();
        Ok(vx_array::new(array))
    })
}

// =============================================================================
// List Array Builder
// =============================================================================
//
// Note: List array builder is not yet exposed in the FFI due to complexity.
// ListViewBuilder requires ListScalar values which would need additional FFI
// wrapper functions. This can be added in a future iteration.

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::array::*;

    #[test]
    fn test_primitive_array_new_i32() {
        unsafe {
            let data = [1i32, 2, 3, 4, 5];
            let mut error = ptr::null_mut();

            let array =
                vx_array_primitive_new_i32(data.as_ptr(), data.len(), ptr::null(), &raw mut error);

            assert!(error.is_null());
            assert!(!array.is_null());
            assert_eq!(vx_array_len(array), 5);
            assert_eq!(vx_array_get_i32(array, 0), 1);
            assert_eq!(vx_array_get_i32(array, 4), 5);

            vx_array_free(array);
        }
    }

    #[test]
    fn test_primitive_array_new_with_validity() {
        unsafe {
            let data = [1i32, 2, 3];
            let validity = [true, false, true];
            let mut error = ptr::null_mut();

            let array = vx_array_primitive_new_i32(
                data.as_ptr(),
                data.len(),
                validity.as_ptr(),
                &raw mut error,
            );

            assert!(error.is_null());
            assert!(!array.is_null());
            assert_eq!(vx_array_len(array), 3);

            assert!(!vx_array_is_null(array, 0, &raw mut error));
            assert!(vx_array_is_null(array, 1, &raw mut error));
            assert!(!vx_array_is_null(array, 2, &raw mut error));

            vx_array_free(array);
        }
    }

    #[test]
    fn test_null_array() {
        unsafe {
            let array = vx_array_null_new(5);
            assert_eq!(vx_array_len(array), 5);

            let mut error = ptr::null_mut();
            for i in 0..5 {
                assert!(vx_array_is_null(array, i, &raw mut error));
            }

            vx_array_free(array);
        }
    }

    #[test]
    fn test_utf8_builder() {
        unsafe {
            let builder = vx_array_utf8_builder_new(false);
            assert!(!builder.is_null());

            let mut error = ptr::null_mut();

            let hello = b"hello";
            vx_varbinview_builder_append_utf8(builder, hello.as_ptr(), hello.len(), &raw mut error);
            assert!(error.is_null());

            let world = b"world";
            vx_varbinview_builder_append_utf8(builder, world.as_ptr(), world.len(), &raw mut error);
            assert!(error.is_null());

            let array = vx_varbinview_builder_finish(builder);
            assert_eq!(vx_array_len(array), 2);

            vx_array_free(array);
        }
    }

    #[test]
    fn test_bool_array() {
        unsafe {
            let data = [true, false, true, true, false];
            let mut error = ptr::null_mut();

            let array = vx_array_bool_new(data.as_ptr(), data.len(), ptr::null(), &raw mut error);

            assert!(error.is_null());
            assert!(!array.is_null());
            assert_eq!(vx_array_len(array), 5);

            vx_array_free(array);
        }
    }
}
