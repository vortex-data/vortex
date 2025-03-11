//! FFI interface for working with Vortex Arrays.
//!
//! The FFIArray provides both type-erased and type-aware access behind an `ArrayRef`.

use std::ffi::{c_int, c_void};

use vortex::compute::{scalar_at, slice};
use vortex::dtype::DType;
use vortex::dtype::half::f16;
use vortex::error::{VortexExpect, VortexUnwrap};
use vortex::{Array, ArrayRef, ArrayVariants};

/// The FFI interface for an [`Array`].
///
/// Because dyn Trait pointers cannot be shared across FFI, we create a new struct to hold
/// the wide pointer. The C FFI only seems a pointer to this structure, and can pass it into
/// one of the various `FFIArray_*` functions.
#[repr(C)]
pub struct FFIArray {
    pub inner: ArrayRef,
}

/// Get the length of the array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_len(ffi_array: *const FFIArray) -> u64 {
    let array = &*ffi_array;

    array.inner.len() as u64
}

/// Get a pointer to the data type for an array.
///
/// Note that this pointer is tied to the lifetime of the array, and the caller is responsible
/// for ensuring that it is never dereferenced after the array has been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_dtype(ffi_array: *const FFIArray) -> *const DType {
    let array = &*ffi_array;

    array.inner.dtype()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_get_field(
    ffi_array: *const FFIArray,
    index: u32,
) -> *const FFIArray {
    let array = &*ffi_array;

    let field_array = array
        .inner
        .as_struct_typed()
        .vortex_expect("FFIArray_get_field: expected struct-typed array")
        .maybe_null_field_by_idx(index as usize)
        .vortex_expect("FFIArray_get_field: field by index");

    let ffi_array = Box::new(FFIArray { inner: field_array });

    Box::into_raw(ffi_array)
}

// Get a pointer to the child array reference here instead...we have no concept of references
// and ownership lifetimes. Holy shit this is a bit scary tbh.

/// Free the array and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_free(ffi_array: *mut FFIArray) -> i32 {
    let boxed = Box::from_raw(ffi_array);
    drop(boxed);

    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_slice(
    array: *const FFIArray,
    start: u32,
    stop: u32,
) -> *mut FFIArray {
    let array = &*array;
    let sliced = slice(array.inner.as_ref(), start as usize, stop as usize)
        .vortex_expect("FFIArray_slice: slice");
    Box::into_raw(Box::new(FFIArray { inner: sliced }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_is_null(array: *const FFIArray, index: u32) -> bool {
    let array = &*array;
    array
        .inner
        .is_invalid(index as usize)
        .vortex_expect("FFIArray_is_null: is_invalid")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_null_count(array: *const FFIArray) -> u32 {
    let array = &*array;
    array
        .inner
        .as_ref()
        .invalid_count()
        .vortex_expect("FFIArray_null_count: invalid count")
        .try_into()
        .vortex_expect("FFIArray_null_count: invalid count to u32")
}

macro_rules! ffiarray_get_ptype {
    ($ptype:ident) => {
        paste::paste! {
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<FFIArray_get_ $ptype>](array: *const FFIArray, index: u32) -> $ptype {
                let array = &*array;
                let value = scalar_at(array.inner.as_ref(), index as usize).vortex_expect("scalar_at");
                value.as_primitive()
                    .as_::<$ptype>()
                    .vortex_expect("as_")
                    .vortex_expect("null value")
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<FFIArray_get_storage_ $ptype>](array: *const FFIArray, index: u32) -> $ptype {
                let array = &*array;
                let value = scalar_at(array.inner.as_ref(), index as usize).vortex_expect("scalar_at");
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
pub unsafe extern "C" fn FFIArray_get_utf8(
    array: *const FFIArray,
    index: u32,
    dst: *mut c_void,
    len: *mut c_int,
) {
    let array = &*array;
    let value = scalar_at(array.inner.as_ref(), index as usize).vortex_expect("scalar_at");
    let utf8_scalar = value.as_utf8();
    if let Some(buffer) = utf8_scalar.value() {
        let bytes = buffer.as_bytes();
        let dst = std::slice::from_raw_parts_mut(dst as *mut u8, bytes.len());
        dst.copy_from_slice(bytes);
        *len = bytes.len().try_into().vortex_unwrap();
    }
}

/// Write the UTF-8 string at `index` in the array into the provided destination buffer, recording
/// the length in `len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_get_binary(
    array: *const FFIArray,
    index: u32,
    dst: *mut c_void,
    len: *mut c_int,
) {
    let array = &*array;
    let value = scalar_at(array.inner.as_ref(), index as usize).vortex_expect("scalar_at");
    let utf8_scalar = value.as_binary();
    if let Some(bytes) = utf8_scalar.value() {
        let dst = std::slice::from_raw_parts_mut(dst as *mut u8, bytes.len());
        dst.copy_from_slice(&bytes);
        *len = bytes.len().try_into().vortex_unwrap();
    }
}

#[cfg(test)]
mod tests {
    use vortex::Array;
    use vortex::arrays::PrimitiveArray;
    use vortex::buffer::buffer;
    use vortex::validity::Validity;

    use crate::array::{FFIArray, FFIArray_dtype, FFIArray_free, FFIArray_get_i32, FFIArray_len};
    use crate::dtype::{DTYPE_PRIMITIVE_I32, DType_get};

    #[test]
    fn test_simple() {
        unsafe {
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let ffi_array = Box::new(FFIArray {
                inner: primitive.to_array(),
            });

            assert_eq!(FFIArray_len(&*ffi_array), 3);

            let array_dtype = FFIArray_dtype(&*ffi_array);
            assert_eq!(DType_get(array_dtype), DTYPE_PRIMITIVE_I32);

            assert_eq!(FFIArray_get_i32(&*ffi_array, 0), 1);
            assert_eq!(FFIArray_get_i32(&*ffi_array, 1), 2);
            assert_eq!(FFIArray_get_i32(&*ffi_array, 2), 3);

            FFIArray_free(Box::into_raw(ffi_array));
        }
    }
}
