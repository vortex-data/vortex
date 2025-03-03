//! FFI interface for working with Vortex Arrays.
//!
//! The FFIArray provides both type-erased and type-aware access behind an `ArrayRef`.

use vortex::compute::scalar_at;
use vortex::dtype::DType;
use vortex::dtype::half::f16;
use vortex::error::VortexExpect;
use vortex::{Array, ArrayRef};

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
    println!("ARRAY LEN");
    let array = non_null!(&*ffi_array, returning: u64::MAX);
    println!("LEN out");
    println!("LEN IS {}", array.inner.len());
    array.inner.len() as u64
}

/// Get a pointer to the data type for an array.
///
/// Note that this pointer is tied to the lifetime of the array, and the caller is responsible
/// for ensuring that it is never dereferenced after the array has been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_dtype(ffi_array: *const FFIArray) -> *const DType {
    let array = non_null!(&*ffi_array, returning: std::ptr::null());

    array.inner.dtype()
}

// Get a pointer to the child array reference here instead...we have no concept of references
// and ownership lifetimes. Holy shit this is a bit scary tbh.

/// Free the array and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_free(ffi_array: *mut FFIArray) -> i32 {
    let boxed = non_null!(Box::from_raw(ffi_array), returning: -1);
    drop(boxed);

    0
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
