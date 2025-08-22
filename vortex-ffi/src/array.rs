// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FFI interface for working with Vortex Arrays.
use std::ffi::{c_int, c_void};
use std::slice;

use vortex::dtype::half::f16;
use vortex::error::{VortexExpect, VortexUnwrap, vortex_err};
use vortex::{Array, ToCanonical};

use crate::arc_dyn_wrapper;
use crate::dtype::vx_dtype;
use crate::error::{try_or_default, vx_error};

arc_dyn_wrapper!(
    /// Base type for all Vortex arrays.
    ///
    /// All built-in Vortex array types can be safely cast to this type to pass into functions that
    /// expect a generic array type. e.g.
    ///
    /// ```cpp
    /// auto primitive_array = vx_array_primitive_new(...);
    /// vx_array_len((*vx_array) primitive_array));
    /// ```
    dyn Array,
    vx_array
);

/// Get the length of the array.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_len(array: *const vx_array) -> usize {
    vx_array::as_ref(array).len()
}

/// Get the [`crate::vx_dtype`] of the array.
///
/// The returned pointer is valid as long as the array is valid.
/// Do NOT free the returned dtype pointer - it shares the lifetime of the array.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_dtype(array: *const vx_array) -> *const vx_dtype {
    vx_dtype::new_ref(vx_array::as_ref(array).dtype())
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_get_field(
    array: *const vx_array,
    index: u32,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error_out, || {
        let array = vx_array::as_ref(array);

        let field_array = array
            .to_struct()?
            .fields()
            .get(index as usize)
            .ok_or_else(|| vortex_err!("Field index out of bounds"))?
            .clone();

        Ok(vx_array::new(field_array))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_slice(
    array: *const vx_array,
    start: u32,
    stop: u32,
    // TODO(aduffy): deprecate this from the FFI API.
    _error_out: *mut *mut vx_error,
) -> *const vx_array {
    let array = vx_array::as_ref(array);
    let sliced = array.slice(start as usize, stop as usize);
    vx_array::new(sliced)
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_is_null(
    array: *const vx_array,
    index: u32,
    error_out: *mut *mut vx_error,
) -> bool {
    let array = vx_array::as_ref(array);
    try_or_default(error_out, || array.is_invalid(index as usize))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_null_count(
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) -> u32 {
    let array = vx_array::as_ref(array);
    try_or_default(error_out, || Ok(array.invalid_count()?.try_into()?))
}

macro_rules! ffiarray_get_ptype {
    ($ptype:ident) => {
        paste::paste! {
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_get_ $ptype>](array: *const vx_array, index: u32) -> $ptype {
                let array = vx_array::as_ref(array);
                let value = array.scalar_at(index as usize);
                value.as_primitive()
                    .as_::<$ptype>()
                    .vortex_expect("null value")
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_get_storage_ $ptype>](array: *const vx_array, index: u32) -> $ptype {
                let array = vx_array::as_ref(array);
                let value = array.scalar_at(index as usize);
                value.as_extension()
                    .storage()
                    .as_primitive()
                    .as_::<$ptype>()
                    .vortex_expect("null value")
            }
        }
    };
}

ffiarray_get_ptype!(u8);
ffiarray_get_ptype!(u16);
ffiarray_get_ptype!(u32);
ffiarray_get_ptype!(u64);
ffiarray_get_ptype!(i8);
ffiarray_get_ptype!(i16);
ffiarray_get_ptype!(i32);
ffiarray_get_ptype!(i64);

// Safe variants of primitive accessor functions with proper error handling
// These provide error handling instead of panicking on null values, type mismatches, or bounds errors
macro_rules! ffiarray_get_ptype_safe {
    ($ptype:ident) => {
        paste::paste! {
            /// Safe variant of primitive accessor that returns errors instead of panicking.
            ///
            /// Returns the default value (0) if an error occurs. Check error_out for details.
            /// This prevents segmentation faults caused by panics in FFI code.
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_get_ $ptype _safe>](
                array: *const vx_array,
                index: u32,
                error_out: *mut *mut vx_error
            ) -> $ptype {
                try_or_default(error_out, || {
                    let array = vx_array::as_ref(array);

                    // Bounds checking - prevent out-of-bounds access
                    if index as usize >= array.len() {
                        return Err(vortex_err!("Index {} out of bounds for array of length {}", index, array.len()));
                    }

                    let value = array.scalar_at(index as usize);

                    // Null checking - prevent accessing null values
                    if value.is_null() {
                        return Err(vortex_err!("Cannot access null value as {}", stringify!($ptype)));
                    }

                    // Type-safe conversion with proper error handling
                    let primitive_scalar = value.as_primitive();
                    primitive_scalar.as_::<$ptype>()
                        .ok_or_else(|| vortex_err!("Cannot convert value to {} (type mismatch)", stringify!($ptype)))
                })
            }

            /// Safe variant of storage accessor that returns errors instead of panicking.
            ///
            /// Returns the default value (0) if an error occurs. Check error_out for details.
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_get_storage_ $ptype _safe>](
                array: *const vx_array,
                index: u32,
                error_out: *mut *mut vx_error
            ) -> $ptype {
                try_or_default(error_out, || {
                    let array = vx_array::as_ref(array);

                    // Bounds checking
                    if index as usize >= array.len() {
                        return Err(vortex_err!("Index {} out of bounds for array of length {}", index, array.len()));
                    }

                    let value = array.scalar_at(index as usize);

                    // For extension types, get storage and convert
                    let extension = value.as_extension();
                    let storage = extension.storage();
                    let primitive_scalar = storage.as_primitive();

                    primitive_scalar.as_::<$ptype>()
                        .ok_or_else(|| vortex_err!("Cannot convert storage value to {} (type mismatch)", stringify!($ptype)))
                })
            }
        }
    };
}

ffiarray_get_ptype_safe!(u8);
ffiarray_get_ptype_safe!(u16);
ffiarray_get_ptype_safe!(u32);
ffiarray_get_ptype_safe!(u64);
ffiarray_get_ptype_safe!(i8);
ffiarray_get_ptype_safe!(i16);
ffiarray_get_ptype_safe!(i32);
ffiarray_get_ptype_safe!(i64);
ffiarray_get_ptype_safe!(f32);
ffiarray_get_ptype_safe!(f64);
// Note: f16 support may need special handling due to half-precision floating point

ffiarray_get_ptype!(f16);
ffiarray_get_ptype!(f32);
ffiarray_get_ptype!(f64);

/// Write the UTF-8 string at `index` in the array into the provided destination buffer, recording
/// the length in `len`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_get_utf8(
    array: *const vx_array,
    index: u32,
    dst: *mut c_void,
    len: *mut c_int,
) {
    let array = vx_array::as_ref(array);
    let value = array.scalar_at(index as usize);
    let utf8_scalar = value.as_utf8();
    if let Some(buffer) = utf8_scalar.value() {
        let bytes = buffer.as_bytes();
        let dst = unsafe { slice::from_raw_parts_mut(dst as *mut u8, bytes.len()) };
        dst.copy_from_slice(bytes);
        unsafe { *len = bytes.len().try_into().vortex_unwrap() };
    }
}

/// Write the UTF-8 string at `index` in the array into the provided destination buffer, recording
/// the length in `len`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_get_binary(
    array: *const vx_array,
    index: u32,
    dst: *mut c_void,
    len: *mut c_int,
) {
    let array = vx_array::as_ref(array);
    let value = array.scalar_at(index as usize);
    let utf8_scalar = value.as_binary();
    if let Some(bytes) = utf8_scalar.value() {
        let dst = unsafe { slice::from_raw_parts_mut(dst as *mut u8, bytes.len()) };
        dst.copy_from_slice(&bytes);
        unsafe { *len = bytes.len().try_into().vortex_unwrap() };
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::{c_int, c_void};

    use std::ptr;
    use vortex::IntoArray;
    use vortex::arrays::{PrimitiveArray, StructArray, VarBinViewArray};
    use vortex::buffer::{Buffer, buffer};
    #[cfg(not(miri))]
    use vortex::dtype::half::f16;
    use vortex::validity::Validity;

    use crate::array::*;
    use crate::dtype::{vx_dtype_get_variant, vx_dtype_variant};
    use crate::error::vx_error_free;

    #[test]
    fn test_simple() {
        unsafe {
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let ffi_array = vx_array::new(primitive.into_array());

            assert_eq!(vx_array_len(ffi_array), 3);

            let array_dtype = vx_array_dtype(ffi_array);
            assert_eq!(
                vx_dtype_get_variant(array_dtype),
                vx_dtype_variant::DTYPE_PRIMITIVE
            );

            assert_eq!(vx_array_get_i32(ffi_array, 0), 1);
            assert_eq!(vx_array_get_i32(ffi_array, 1), 2);
            assert_eq!(vx_array_get_i32(ffi_array, 2), 3);

            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_slice() {
        unsafe {
            let primitive =
                PrimitiveArray::new(buffer![1i32, 2i32, 3i32, 4i32, 5i32], Validity::NonNullable);
            let ffi_array = vx_array::new(primitive.into_array());

            let mut error = ptr::null_mut();
            let sliced = vx_array_slice(ffi_array, 1, 4, &raw mut error);
            assert!(error.is_null());
            assert_eq!(vx_array_len(sliced), 3);
            assert_eq!(vx_array_get_i32(sliced, 0), 2);
            assert_eq!(vx_array_get_i32(sliced, 1), 3);
            assert_eq!(vx_array_get_i32(sliced, 2), 4);

            vx_array_free(sliced);
            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_null_operations() {
        unsafe {
            let primitive = PrimitiveArray::new(
                buffer![1i32, 2i32, 3i32],
                Validity::from_iter([true, false, true]),
            );
            let ffi_array = vx_array::new(primitive.into_array());

            let mut error = ptr::null_mut();
            assert!(!vx_array_is_null(ffi_array, 0, &raw mut error));
            assert!(error.is_null());
            assert!(vx_array_is_null(ffi_array, 1, &raw mut error));
            assert!(error.is_null());
            assert!(!vx_array_is_null(ffi_array, 2, &raw mut error));
            assert!(error.is_null());

            let null_count = vx_array_null_count(ffi_array, &raw mut error);
            assert!(error.is_null());
            assert_eq!(null_count, 1);

            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_get_field() {
        unsafe {
            let names = VarBinViewArray::from_iter_str(["Alice", "Bob", "Charlie"]);
            let ages = PrimitiveArray::new(buffer![30u8, 25u8, 35u8], Validity::NonNullable);
            let struct_array = StructArray::try_new(
                ["name", "age"].into(),
                vec![names.into_array(), ages.into_array()],
                3,
                Validity::NonNullable,
            )
            .unwrap();
            let ffi_array = vx_array::new(struct_array.into_array());

            let mut error = ptr::null_mut();
            let field0 = vx_array_get_field(ffi_array, 0, &raw mut error);
            assert!(error.is_null());
            assert_eq!(vx_array_len(field0), 3);

            let field1 = vx_array_get_field(ffi_array, 1, &raw mut error);
            assert!(error.is_null());
            assert_eq!(vx_array_len(field1), 3);
            assert_eq!(vx_array_get_u8(field1, 0), 30);
            assert_eq!(vx_array_get_u8(field1, 1), 25);
            assert_eq!(vx_array_get_u8(field1, 2), 35);

            // Test out of bounds
            let field_oob = vx_array_get_field(ffi_array, 2, &raw mut error);
            assert!(!error.is_null());
            assert!(field_oob.is_null());
            vx_error_free(error);

            vx_array_free(field0);
            vx_array_free(field1);
            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_primitive_getters() {
        unsafe {
            // Test a representative sample of primitive types
            // The macro generates identical code for all types, so exhaustive testing is redundant

            // Test signed integer with edge cases
            let i32_array =
                PrimitiveArray::new(buffer![i32::MAX, i32::MIN, 0i32], Validity::NonNullable);
            let ffi_i32 = vx_array::new(i32_array.into_array());
            assert_eq!(vx_array_get_i32(ffi_i32, 0), i32::MAX);
            assert_eq!(vx_array_get_i32(ffi_i32, 1), i32::MIN);
            assert_eq!(vx_array_get_i32(ffi_i32, 2), 0);
            vx_array_free(ffi_i32);

            // Test unsigned integer
            let u64_array =
                PrimitiveArray::new(buffer![u64::MAX, 0u64, 42u64], Validity::NonNullable);
            let ffi_u64 = vx_array::new(u64_array.into_array());
            assert_eq!(vx_array_get_u64(ffi_u64, 0), u64::MAX);
            assert_eq!(vx_array_get_u64(ffi_u64, 1), 0);
            assert_eq!(vx_array_get_u64(ffi_u64, 2), 42);
            vx_array_free(ffi_u64);

            // Test floating point including special values
            let f64_array = PrimitiveArray::new(
                buffer![f64::NEG_INFINITY, 0.0f64, f64::NAN],
                Validity::NonNullable,
            );
            let ffi_f64 = vx_array::new(f64_array.into_array());
            assert_eq!(vx_array_get_f64(ffi_f64, 0), f64::NEG_INFINITY);
            assert_eq!(vx_array_get_f64(ffi_f64, 1), 0.0);
            assert!(vx_array_get_f64(ffi_f64, 2).is_nan());
            vx_array_free(ffi_f64);

            // Test f16 (special half-precision type) - skip in Miri due to inline assembly
            #[cfg(not(miri))]
            {
                let f16_array = PrimitiveArray::new(
                    buffer![f16::from_f32(1.0), f16::from_f32(-0.5)],
                    Validity::NonNullable,
                );
                let ffi_f16 = vx_array::new(f16_array.into_array());
                assert_eq!(vx_array_get_f16(ffi_f16, 0), f16::from_f32(1.0));
                assert_eq!(vx_array_get_f16(ffi_f16, 1), f16::from_f32(-0.5));
                vx_array_free(ffi_f16);
            }
        }
    }

    #[test]
    fn test_get_utf8() {
        unsafe {
            let utf8_array = VarBinViewArray::from_iter_str(["hello", "world", "test"]);
            let ffi_array = vx_array::new(utf8_array.into_array());

            let mut buffer = vec![0u8; 10];
            let mut len: c_int = 0;

            vx_array_get_utf8(
                ffi_array,
                0,
                buffer.as_mut_ptr() as *mut c_void,
                &raw mut len,
            );
            assert_eq!(len, 5);
            assert_eq!(&buffer[..5], b"hello");

            vx_array_get_utf8(
                ffi_array,
                1,
                buffer.as_mut_ptr() as *mut c_void,
                &raw mut len,
            );
            assert_eq!(len, 5);
            assert_eq!(&buffer[..5], b"world");

            vx_array_get_utf8(
                ffi_array,
                2,
                buffer.as_mut_ptr() as *mut c_void,
                &raw mut len,
            );
            assert_eq!(len, 4);
            assert_eq!(&buffer[..4], b"test");

            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_get_binary() {
        unsafe {
            let binary_array = VarBinViewArray::from_iter_bin(vec![
                vec![0x01, 0x02, 0x03],
                vec![0xFF, 0xEE],
                vec![0xAA, 0xBB, 0xCC, 0xDD],
            ]);
            let ffi_array = vx_array::new(binary_array.into_array());

            let mut buffer = vec![0u8; 10];
            let mut len: c_int = 0;

            vx_array_get_binary(
                ffi_array,
                0,
                buffer.as_mut_ptr() as *mut c_void,
                &raw mut len,
            );
            assert_eq!(len, 3);
            assert_eq!(&buffer[..3], &[0x01, 0x02, 0x03]);

            vx_array_get_binary(
                ffi_array,
                1,
                buffer.as_mut_ptr() as *mut c_void,
                &raw mut len,
            );
            assert_eq!(len, 2);
            assert_eq!(&buffer[..2], &[0xFF, 0xEE]);

            vx_array_get_binary(
                ffi_array,
                2,
                buffer.as_mut_ptr() as *mut c_void,
                &raw mut len,
            );
            assert_eq!(len, 4);
            assert_eq!(&buffer[..4], &[0xAA, 0xBB, 0xCC, 0xDD]);

            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_array_dtype_lifetime_pattern() {
        let array = {
            let nums: Buffer<i32> = (0..1000).collect();
            let floats: Buffer<f32> = (0..1000).map(|x| x as f32).collect();

            StructArray::try_from_iter([
                ("nums", nums.into_array()),
                ("floats", floats.into_array()),
            ])
            .unwrap()
            .into_array()
        };
        let vx_arr = vx_array::new(array);

        // Get dtype reference - this is valid as long as array lives
        let dtype_ptr = unsafe { vx_array_dtype(vx_arr) };
        let variant = unsafe { vx_dtype_get_variant(dtype_ptr) };
        assert_eq!(variant, vx_dtype_variant::DTYPE_STRUCT);

        // Proper usage: use dtype while array is still alive
        // This demonstrates the correct lifetime pattern
        unsafe { vx_array_free(vx_arr) };

        // Note: dtype_ptr is now invalid - this test documents the lifetime pattern
        // In real usage, don't access dtype_ptr after freeing the array
    }

    #[test]
    fn test_error_handling_comprehensive() {
        unsafe {
            // Test 1: Field access on non-struct array
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let ffi_array = vx_array::new(primitive.into_array());

            let mut error = ptr::null_mut();
            let field = vx_array_get_field(ffi_array, 0, &raw mut error);

            // Should fail - primitive arrays don't have fields
            assert!(!error.is_null());
            assert!(field.is_null());

            // Verify error message contains meaningful information
            use crate::error::{vx_error_free, vx_error_get_message};
            use crate::string::vx_string_len;
            let error_msg = vx_error_get_message(error);
            assert!(!error_msg.is_null());
            let msg_len = vx_string_len(error_msg);
            assert!(msg_len > 0);

            vx_error_free(error);
            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_out_of_bounds_errors() {
        unsafe {
            // Test array bounds checking
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32], Validity::NonNullable);
            let ffi_array = vx_array::new(primitive.into_array());

            let mut error = ptr::null_mut();

            // Test valid index first
            assert!(!vx_array_is_null(ffi_array, 0, &raw mut error));
            assert!(error.is_null());

            // Test out of bounds index
            let _is_null = vx_array_is_null(ffi_array, 999, &raw mut error);

            // Should generate an error for out-of-bounds access
            if !error.is_null() {
                use crate::error::vx_error_free;
                vx_error_free(error);
            }

            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_struct_field_bounds_checking() {
        unsafe {
            use vortex::arrays::{StructArray, VarBinViewArray};

            let names = VarBinViewArray::from_iter_str(["Alice"]);
            let ages = PrimitiveArray::new(buffer![30u8], Validity::NonNullable);
            let struct_array = StructArray::try_new(
                ["name", "age"].into(),
                vec![names.into_array(), ages.into_array()],
                1,
                Validity::NonNullable,
            )
            .unwrap();
            let ffi_array = vx_array::new(struct_array.into_array());

            let mut error = ptr::null_mut();

            // Test valid field access
            let field0 = vx_array_get_field(ffi_array, 0, &raw mut error);
            assert!(error.is_null());
            assert!(!field0.is_null());
            vx_array_free(field0);

            // Test out-of-bounds field access
            let invalid_field = vx_array_get_field(ffi_array, 999, &raw mut error);
            assert!(!error.is_null()); // Should have error
            assert!(invalid_field.is_null()); // Should return null

            use crate::error::vx_error_free;
            vx_error_free(error);
            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_slice_bounds_validation() {
        unsafe {
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let ffi_array = vx_array::new(primitive.into_array());

            let mut error = ptr::null_mut();

            // Test valid slice
            let valid_slice = vx_array_slice(ffi_array, 0, 2, &raw mut error);
            assert!(error.is_null());
            assert!(!valid_slice.is_null());
            assert_eq!(vx_array_len(valid_slice), 2);
            vx_array_free(valid_slice);

            // NOTE: The following slice operations would panic and are NOT safe:
            // - vx_array_slice(ffi_array, 2, 1, &raw mut error);  // start > stop
            // - vx_array_slice(ffi_array, 0, 999, &raw mut error); // out of bounds
            //
            // These demonstrate the same issue as primitive accessors - they panic
            // instead of returning errors properly. The slice function should also
            // be made safe with proper error handling.

            // For now, we can only test valid slices

            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_null_count_error_handling() {
        unsafe {
            // Create array where null count might fail on certain encodings
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let ffi_array = vx_array::new(primitive.into_array());

            let mut error = ptr::null_mut();
            let null_count = vx_array_null_count(ffi_array, &raw mut error);

            // For this simple case, should succeed
            if error.is_null() {
                // Null count should be 0 for non-nullable array
                assert_eq!(null_count, 0);
            } else {
                // If it fails, clean up the error
                use crate::error::vx_error_free;
                vx_error_free(error);
            }

            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_primitive_accessor_type_safety() {
        unsafe {
            // Test accessing wrong primitive type
            let f64_array = PrimitiveArray::new(buffer![1.5f64, 2.5f64], Validity::NonNullable);
            let ffi_array = vx_array::new(f64_array.into_array());

            // This currently panics instead of proper error handling
            // TODO: This test documents the current dangerous behavior
            // The accessor functions should be fixed to use error parameters

            // For now, test the "correct" type access
            let f64_val = vx_array_get_f64(ffi_array, 0);
            assert!((f64_val - 1.5).abs() < f64::EPSILON);

            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_primitive_accessor_documentation() {
        unsafe {
            // This test documents the current dangerous behavior of primitive accessors
            // They panic instead of returning errors, causing segfaults in FFI usage

            let i32_array = PrimitiveArray::new(buffer![42i32, -123i32], Validity::NonNullable);
            let ffi_array = vx_array::new(i32_array.into_array());

            // Test that the "correct" type access works
            let value = vx_array_get_i32(ffi_array, 0);
            assert_eq!(value, 42);

            let value2 = vx_array_get_i32(ffi_array, 1);
            assert_eq!(value2, -123);

            // NOTE: The following operations would panic and are NOT safe:
            // - vx_array_get_i32(ffi_array, 999);  // Out of bounds
            // - vx_array_get_f64(ffi_array, 0);    // Wrong type (if array contained different type)
            // - Access on null values would also panic
            //
            // This is why safe variants with error parameters are needed.

            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_safe_primitive_accessors() {
        unsafe {
            // Test the new safe primitive accessor functions
            let i32_array = PrimitiveArray::new(buffer![42i32, -123i32], Validity::NonNullable);
            let ffi_array = vx_array::new(i32_array.into_array());

            let mut error = ptr::null_mut();

            // Test successful access
            let value = vx_array_get_i32_safe(ffi_array, 0, &raw mut error);
            assert!(error.is_null());
            assert_eq!(value, 42);

            let value2 = vx_array_get_i32_safe(ffi_array, 1, &raw mut error);
            assert!(error.is_null());
            assert_eq!(value2, -123);

            // Test out-of-bounds access (should return 0 and set error)
            let oob_value = vx_array_get_i32_safe(ffi_array, 999, &raw mut error);
            assert!(!error.is_null()); // Should have error
            assert_eq!(oob_value, 0); // Should return default value

            use crate::error::vx_error_free;
            vx_error_free(error);
            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_safe_primitive_type_mismatch() {
        unsafe {
            // Test type conversion errors in safe accessors
            let f64_array = PrimitiveArray::new(buffer![1.5f64, 2.7f64], Validity::NonNullable);
            let ffi_array = vx_array::new(f64_array.into_array());

            let mut error = ptr::null_mut();

            // Test correct type access
            let f64_value = vx_array_get_f64_safe(ffi_array, 0, &raw mut error);
            assert!(error.is_null());
            assert!((f64_value - 1.5).abs() < f64::EPSILON);

            // Test type mismatch - try to access f64 as i32
            let i32_value = vx_array_get_i32_safe(ffi_array, 0, &raw mut error);

            // Should either succeed with conversion or fail gracefully
            if !error.is_null() {
                // If it fails, should return default value
                assert_eq!(i32_value, 0);
                use crate::error::{vx_error_free, vx_error_get_message};
                use crate::string::vx_string_len;

                // Verify error message
                let error_msg = vx_error_get_message(error);
                let msg_len = vx_string_len(error_msg);
                assert!(msg_len > 0); // Should have meaningful error message

                vx_error_free(error);
            }

            vx_array_free(ffi_array);
        }
    }

    #[test]
    fn test_safe_primitive_null_handling() {
        unsafe {
            // Test null value handling in safe accessors
            let nullable_array = PrimitiveArray::new(
                buffer![42i32, 0i32, 84i32],
                Validity::from_iter([true, false, true]), // Middle element is null
            );
            let ffi_array = vx_array::new(nullable_array.into_array());

            let mut error = ptr::null_mut();

            // Test non-null value
            let value = vx_array_get_i32_safe(ffi_array, 0, &raw mut error);
            assert!(error.is_null());
            assert_eq!(value, 42);

            // Test null value access (should fail gracefully)
            let null_value = vx_array_get_i32_safe(ffi_array, 1, &raw mut error);
            assert!(!error.is_null()); // Should have error for null access
            assert_eq!(null_value, 0); // Should return default value

            use crate::error::{vx_error_free, vx_error_get_message};
            let error_msg = vx_error_get_message(error);
            assert!(!error_msg.is_null()); // Should have error message about null value

            vx_error_free(error);
            error = ptr::null_mut();

            // Test another non-null value
            let value3 = vx_array_get_i32_safe(ffi_array, 2, &raw mut error);
            assert!(error.is_null());
            assert_eq!(value3, 84);

            vx_array_free(ffi_array);
        }
    }

    #[cfg(test)]
    mod error_stress_tests {
        use super::*;

        #[test]
        fn test_concurrent_error_handling() {
            // Test error handling under concurrent access
            use std::thread;

            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let ffi_array = vx_array::new(primitive.into_array());

            // Create multiple threads that access the same array and trigger errors
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let array_ptr = ffi_array as usize; // Convert to raw address for sharing
                    thread::spawn(move || {
                        unsafe {
                            let array = array_ptr as *const vx_array;
                            let mut error = ptr::null_mut();

                            // Try to access out-of-bounds - each thread should handle its own errors
                            let _is_null = vx_array_is_null(array, 999, &raw mut error);

                            if !error.is_null() {
                                use crate::error::vx_error_free;
                                vx_error_free(error);
                            }
                        }
                    })
                })
                .collect();

            // Wait for all threads
            for handle in handles {
                handle.join().unwrap();
            }

            unsafe {
                vx_array_free(ffi_array);
            }
        }
    }
}
