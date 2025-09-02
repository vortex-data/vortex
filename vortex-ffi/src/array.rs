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
            .to_struct()
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
    let sliced = array.slice(start as usize..stop as usize);
    vx_array::new(sliced)
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_is_null(
    array: *const vx_array,
    index: u32,
    _error_out: *mut *mut vx_error,
) -> bool {
    let array = vx_array::as_ref(array);
    array.is_invalid(index as usize)
}

// TODO(robert): Make this return usize and remove error
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_null_count(
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) -> u32 {
    let array = vx_array::as_ref(array);
    try_or_default(error_out, || Ok(array.invalid_count().try_into()?))
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
}
