//! FFI interface for working with Vortex Arrays.
//!
//! The FFIArray provides both type-erased and type-aware access behind an `ArrayRef`.

use vortex::{Array, ArrayRef};

use crate::dtype::FFIDType;

/// The FFI interface for an [`Array`].
#[repr(C)]
pub struct FFIArray {
    pub inner: ArrayRef,
    pub dtype: Box<FFIDType>,
}

/// Free the API and drop any data associated with it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_len(ffi_array: *const FFIArray) -> u64 {
    let ffi_array = &*ffi_array;

    ffi_array.inner.len() as u64
}

/// Get a pointer to the data type for an array.
///
/// Note that this pointer is tied to the lifetime of the array, and the caller is responsible
/// for ensuring that it is never dereferenced after the array has been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_dtype(ffi_array: *const FFIArray) -> *const FFIDType {
    let ffi_array = &*ffi_array;

    ffi_array.dtype.as_ref()
}

// Get a pointer to the child array reference here instead...we have no concept of references
// and ownership lifetimes. Holy shit this is a bit scary tbh.

/// Free the array and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_free(ffi_array: *mut FFIArray) {
    // Convert the raw pointer back into Box so Rust can manage the memory again.
    drop(Box::from_raw(ffi_array));
}

#[cfg(test)]
mod tests {
    use vortex::Array;
    use vortex::arrays::PrimitiveArray;
    use vortex::buffer::buffer;
    use vortex::validity::Validity;

    use crate::array::{FFIArray, FFIArray_dtype, FFIArray_free, FFIArray_len};
    use crate::dtype::{DTYPE_PRIMITIVE_I32, FFIDType_get};

    #[test]
    fn test_simple() {
        unsafe {
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let ffi_array = Box::new(FFIArray {
                inner: primitive.to_array(),
                dtype: Box::new(primitive.dtype().into()),
            });

            assert_eq!(FFIArray_len(&*ffi_array), 3);

            let array_dtype = FFIArray_dtype(&*ffi_array);
            assert_eq!(FFIDType_get(array_dtype), DTYPE_PRIMITIVE_I32);

            FFIArray_free(Box::into_raw(ffi_array));
        }
    }
}
