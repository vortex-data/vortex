// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FFI interface for working with Vortex Arrays.
use std::ptr;
use std::sync::Arc;

use vortex::array::DynArray;
use vortex::array::ToCanonical;
use vortex::dtype::half::f16;
use vortex::error::VortexExpect;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::arc_dyn_wrapper;
use crate::binary::vx_binary;
use crate::dtype::vx_dtype;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::expression::vx_expression;
use crate::string::vx_string;

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
    dyn DynArray,
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

        let struct_array = array.to_struct();
        let idx = index as usize;
        if idx >= struct_array.struct_fields().nfields() {
            return Err(vortex_err!("Field index out of bounds"));
        }
        let field_array = struct_array.unmasked_field(idx).clone();

        Ok(vx_array::new(field_array))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_slice(
    array: *const vx_array,
    start: u32,
    stop: u32,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error_out, || {
        let array = vx_array::as_ref(array);
        let sliced = array.slice(start as usize..stop as usize)?;
        Ok(vx_array::new(sliced))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_is_null(
    array: *const vx_array,
    index: u32,
    _error_out: *mut *mut vx_error,
) -> bool {
    let array = vx_array::as_ref(array);
    // TODO(joe): propagate this error up instead of expecting
    array
        .is_invalid(index as usize)
        .vortex_expect("is_invalid failed")
}

// TODO(robert): Make this return usize and remove error
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
                // TODO(joe): propagate this error up instead of expecting
                let value = array.scalar_at(index as usize).vortex_expect("scalar_at failed");
                // TODO(joe): propagate this error up instead of expecting
                value.as_primitive()
                    .as_::<$ptype>()
                    .vortex_expect("null value")
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_get_storage_ $ptype>](array: *const vx_array, index: u32) -> $ptype {
                let array = vx_array::as_ref(array);
                // TODO(joe): propagate this error up instead of expecting
                let value = array.scalar_at(index as usize).vortex_expect("scalar_at failed");
                // TODO(joe): propagate this error up instead of expecting
                value.as_extension()
                    .to_storage_scalar()
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

/// Return the utf-8 string at `index` in the array. The pointer will be null if the value at `index` is null.
/// The caller must free the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_get_utf8(
    array: *const vx_array,
    index: u32,
) -> *const vx_string {
    let array = vx_array::as_ref(array);
    // TODO(joe): propagate this error up instead of expecting
    let value = array
        .scalar_at(index as usize)
        .vortex_expect("scalar_at failed");
    let utf8_scalar = value.as_utf8();
    if let Some(buffer) = utf8_scalar.value() {
        vx_string::new(Arc::from(buffer.as_str()))
    } else {
        ptr::null()
    }
}

/// Return the binary at `index` in the array. The pointer will be null if the value at `index` is null.
/// The caller must free the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_get_binary(
    array: *const vx_array,
    index: u32,
) -> *const vx_binary {
    let array = vx_array::as_ref(array);
    // TODO(joe): propagate this error up instead of expecting
    let value = array
        .scalar_at(index as usize)
        .vortex_expect("scalar_at failed");
    let binary_scalar = value.as_binary();
    if let Some(bytes) = binary_scalar.value() {
        vx_binary::new(Arc::from(bytes.as_bytes()))
    } else {
        ptr::null()
    }
}

/// Apply the expression to the array, wrapping it with a ScalarFnArray.
/// This operation takes constant time as it doesn't execute the underlying
/// array. Executing the underlying array still takes O(n) time.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_array_apply(
    array: *const vx_array,
    expression: *const vx_expression,
    error: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error, || {
        vortex_ensure!(!array.is_null());
        vortex_ensure!(!expression.is_null());
        let array = vx_array::as_ref(array);
        let expression = vx_expression::as_ref(expression);
        Ok(vx_array::new(Arc::new(array.apply(expression)?)))
    })
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use vortex::array::IntoArray;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    #[cfg(not(miri))]
    use vortex::dtype::half::f16;
    use vortex::expr::eq;
    use vortex::expr::lit;
    use vortex::expr::root;

    use crate::array::*;
    use crate::binary::vx_binary_free;
    use crate::dtype::vx_dtype_get_variant;
    use crate::dtype::vx_dtype_variant;
    use crate::error::vx_error_free;
    use crate::expression::vx_expression_free;
    use crate::string::vx_string_free;

    #[test]
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
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
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
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
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
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
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
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
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
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
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
    fn test_get_utf8() {
        unsafe {
            let utf8_array = VarBinViewArray::from_iter_str(["hello", "world", "test"]);
            let ffi_array = vx_array::new(utf8_array.into_array());

            let vx_str1 = vx_array_get_utf8(ffi_array, 0);
            assert_eq!(vx_string::as_str(vx_str1), "hello");
            vx_string_free(vx_str1);

            let vx_str2 = vx_array_get_utf8(ffi_array, 1);
            assert_eq!(vx_string::as_str(vx_str2), "world");
            vx_string_free(vx_str2);

            let vx_str3 = vx_array_get_utf8(ffi_array, 2);
            assert_eq!(vx_string::as_str(vx_str3), "test");
            vx_string_free(vx_str3);

            vx_array_free(ffi_array);
        }
    }

    #[test]
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
    fn test_get_binary() {
        unsafe {
            let binary_array = VarBinViewArray::from_iter_bin(vec![
                vec![0x01, 0x02, 0x03],
                vec![0xFF, 0xEE],
                vec![0xAA, 0xBB, 0xCC, 0xDD],
            ]);
            let ffi_array = vx_array::new(binary_array.into_array());

            let vx_bin1 = vx_array_get_binary(ffi_array, 0);
            assert_eq!(vx_binary::as_slice(vx_bin1), &[0x01, 0x02, 0x03]);
            vx_binary_free(vx_bin1);

            let vx_bin2 = vx_array_get_binary(ffi_array, 1);
            assert_eq!(vx_binary::as_slice(vx_bin2), &[0xFF, 0xEE]);
            vx_binary_free(vx_bin2);

            let vx_bin3 = vx_array_get_binary(ffi_array, 2);
            assert_eq!(vx_binary::as_slice(vx_bin3), &[0xAA, 0xBB, 0xCC, 0xDD]);
            vx_binary_free(vx_bin3);

            vx_array_free(ffi_array);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_apply() {
        let primitive = PrimitiveArray::new(
            buffer![1i32, 2i32, 3i32, 3i32],
            Validity::from_iter([true, false, true, true]),
        );

        unsafe {
            let mut error = ptr::null_mut();

            let res = vx_array_apply(ptr::null(), ptr::null(), &raw mut error);
            assert!(res.is_null());
            assert!(!error.is_null());
            vx_error_free(error);

            let array = vx_array::new(primitive.into_array());

            let res = vx_array_apply(array, ptr::null(), &raw mut error);
            assert!(res.is_null());
            assert!(!error.is_null());
            vx_error_free(error);

            // Test with Vortex Rust-side expressions here, test C API for
            // expressions in src/expressions.rs
            let expression = eq(root(), lit(3i32));
            let expression = vx_expression::new(Box::new(expression));

            let res = vx_array_apply(ptr::null(), expression, &raw mut error);
            assert!(res.is_null());
            assert!(!error.is_null());
            vx_error_free(error);

            let res = vx_array_apply(array, expression, &raw mut error);
            assert!(!res.is_null());
            assert!(error.is_null());
            {
                let res = vx_array::as_ref(res);
                let buffer = res.to_bool().to_bit_buffer();
                let expected = BoolArray::from_iter(vec![false, false, true, true]);
                assert_eq!(buffer, expected.to_bit_buffer());
            }
            vx_array_free(res);

            vx_expression_free(expression);
            vx_array_free(array);
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
