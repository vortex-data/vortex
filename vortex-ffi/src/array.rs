//! FFI interface for working with Vortex Arrays.
//!
//! The VXArray provides both type-erased and type-aware access behind an `ArrayRef`.

use std::ffi::{c_int, c_void};
use std::ptr;

use vortex::dtype::DType;
use vortex::dtype::half::f16;
use vortex::error::{VortexExpect, VortexUnwrap, vortex_err};
use vortex::{Array, ArrayRef, ArrayVariants};

use crate::error::{try_or, vx_error};

/// The FFI interface for an [`Array`].
///
/// Because dyn Trait pointers cannot be shared across FFI, we create a new struct to hold
/// the wide pointer. The C FFI only seems a pointer to this structure, and can pass it into
/// one of the various `vx_array_*` functions.
#[allow(non_camel_case_types)]
pub struct vx_array {
    pub inner: ArrayRef,
}

/// Get the length of the array.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_len(array: *const vx_array) -> u64 {
    unsafe { array.as_ref() }
        .vortex_expect("array null")
        .inner
        .len() as u64
}

/// Get a pointer to the data type for an array.
///
/// Note that this pointer is tied to the lifetime of the array, and the caller is responsible
/// for ensuring that it is never dereferenced after the array has been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_dtype(array: *const vx_array) -> *const DType {
    unsafe { array.as_ref() }
        .vortex_expect("array null")
        .inner
        .dtype()
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_get_field(
    array: *const vx_array,
    index: u32,
    error: *mut *mut vx_error,
) -> *const vx_array {
    try_or(error, ptr::null(), || {
        let array = unsafe { array.as_ref() }.vortex_expect("array null");

        let field_array = array
            .inner
            .as_struct_typed()
            .ok_or_else(|| vortex_err!("vx_array_get_field: expected struct-typed array"))?
            .maybe_null_field_by_idx(index as usize)?;

        let ffi_array = field_array;

        Ok(Box::into_raw(Box::new(vx_array { inner: ffi_array })))
    })
}

// Get a pointer to the child array reference here instead...we have no concept of references
// and ownership lifetimes. Holy shit this is a bit scary tbh.

/// Free the array and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_free(array: *mut vx_array) {
    assert!(!array.is_null());
    drop(unsafe { Box::from_raw(array) });
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_slice(
    array: *const vx_array,
    start: u32,
    stop: u32,
    error: *mut *mut vx_error,
) -> *const vx_array {
    let array = unsafe { array.as_ref() }.vortex_expect("array null");
    try_or(error, ptr::null_mut(), || {
        let sliced = array.inner.as_ref().slice(start as usize, stop as usize)?;
        Ok(Box::into_raw(Box::new(vx_array { inner: sliced })))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_is_null(
    array: *const vx_array,
    index: u32,
    error: *mut *mut vx_error,
) -> bool {
    let array = unsafe { array.as_ref() }.vortex_expect("array null");
    try_or(error, false, || array.inner.is_invalid(index as usize))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_null_count(
    array: *const vx_array,
    error: *mut *mut vx_error,
) -> u32 {
    let array = unsafe { array.as_ref() }.vortex_expect("array null");
    try_or(error, 0, || {
        Ok(array.inner.as_ref().invalid_count()?.try_into()?)
    })
}

macro_rules! ffiarray_get_ptype {
    ($ptype:ident) => {
        paste::paste! {
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_get_ $ptype>](array: *const vx_array, index: u32) -> $ptype {
                let array = unsafe { array.as_ref() } .vortex_expect("array null");
                let value = array.inner.scalar_at(index as usize).vortex_expect("scalar_at");
                value.as_primitive()
                    .as_::<$ptype>()
                    .vortex_expect("as_")
                    .vortex_expect("null value")
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_get_storage_ $ptype>](array: *const vx_array, index: u32) -> $ptype {
                let array = unsafe { array.as_ref() }.vortex_expect("array null");
                let value = array.inner.scalar_at(index as usize).vortex_expect("scalar_at");
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
    let array = unsafe { array.as_ref() }.vortex_expect("array null");
    let value = array
        .inner
        .as_ref()
        .scalar_at(index as usize)
        .vortex_expect("scalar_at");
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
    let array = unsafe { array.as_ref() }.vortex_expect("array null");
    let value = array
        .inner
        .scalar_at(index as usize)
        .vortex_expect("scalar_at");
    let utf8_scalar = value.as_binary();
    if let Some(bytes) = utf8_scalar.value() {
        let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, bytes.len()) };
        dst.copy_from_slice(&bytes);
        unsafe { *len = bytes.len().try_into().vortex_unwrap() };
    }
}

#[cfg(test)]
mod tests {
    use vortex::Array;
    use vortex::arrays::PrimitiveArray;
    use vortex::buffer::buffer;
    use vortex::validity::Validity;

    use crate::array::{vx_array, vx_array_dtype, vx_array_free, vx_array_get_i32, vx_array_len};
    use crate::dtype::{DTYPE_PRIMITIVE_I32, vx_dtype_get};

    #[test]
    fn test_simple() {
        unsafe {
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let vx_array = Box::new(vx_array {
                inner: primitive.to_array(),
            });

            assert_eq!(vx_array_len(&*vx_array), 3);

            let array_dtype = vx_array_dtype(&*vx_array);
            assert_eq!(vx_dtype_get(array_dtype), DTYPE_PRIMITIVE_I32);

            assert_eq!(vx_array_get_i32(&*vx_array, 0), 1);
            assert_eq!(vx_array_get_i32(&*vx_array, 1), 2);
            assert_eq!(vx_array_get_i32(&*vx_array, 2), 3);

            vx_array_free(Box::into_raw(vx_array));
        }
    }
}
