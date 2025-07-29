// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FFI interface for working with Vortex Arrays.
use std::ffi::{c_int, c_void};

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
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    let array = vx_array::as_ref(array);
    try_or_default(error_out, || {
        let sliced = array.slice(start as usize, stop as usize)?;
        Ok(vx_array::new(sliced))
    })
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
                let value = array.scalar_at(index as usize).vortex_expect("scalar_at");
                value.as_primitive()
                    .as_::<$ptype>()
                    .vortex_expect("as_")
                    .vortex_expect("null value")
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_get_storage_ $ptype>](array: *const vx_array, index: u32) -> $ptype {
                let array = vx_array::as_ref(array);
                let value = array.scalar_at(index as usize).vortex_expect("scalar_at");
                value.as_extension()
                    .storage()
                    .as_primitive()
                    .as_::<$ptype>()
                    .vortex_expect("as_")
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
    let value = array.scalar_at(index as usize).vortex_expect("scalar_at");
    let utf8_scalar = value.as_utf8();
    if let Some(buffer) = utf8_scalar.value() {
        let bytes = buffer.as_bytes();
        let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, bytes.len()) };
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
    let value = array.scalar_at(index as usize).vortex_expect("scalar_at");
    let utf8_scalar = value.as_binary();
    if let Some(bytes) = utf8_scalar.value() {
        let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, bytes.len()) };
        dst.copy_from_slice(&bytes);
        unsafe { *len = bytes.len().try_into().vortex_unwrap() };
    }
}

#[cfg(test)]
mod tests {
    use vortex::arrays::PrimitiveArray;
    use vortex::buffer::buffer;
    use vortex::validity::Validity;

    use crate::array::{vx_array, vx_array_dtype, vx_array_free, vx_array_get_i32, vx_array_len};
    use crate::dtype::{vx_dtype_get_variant, vx_dtype_variant};

    #[test]
    fn test_simple() {
        unsafe {
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let ffi_array = vx_array::new(primitive.to_array());

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
}
