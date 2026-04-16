// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![expect(non_camel_case_types)]

//! FFI interface for working with Vortex Arrays.
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use paste::paste;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
#[expect(deprecated)]
use vortex::array::ToCanonical;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::NullArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::half::f16;
use vortex::error::VortexExpect;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::arc_wrapper;
use crate::binary::vx_binary;
use crate::dtype::vx_dtype;
use crate::dtype::vx_dtype_variant;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::error::write_error;
use crate::expression::vx_expression;
use crate::ptype::vx_ptype;
use crate::string::vx_string;

arc_wrapper!(
    /// Arrays are reference-counted handles to owned memory buffers that hold
    /// scalars. These buffers can be held in a number of physical encodings to
    /// perform lightweight compression that exploits the particular data
    /// distribution of the array's values.
    ///
    /// Every data type recognized by Vortex also has a canonical physical
    /// encoding format, which arrays can be canonicalized into for ease of
    /// access in compute functions.
    ///
    /// As an implementation detail, vx_array Arc'ed inside, so cloning an
    /// array is a cheap operation.
    ///
    /// Unless stated explicitly, all operations with vx_array don't take
    /// ownership of it, and thus it must be freed by the caller.
    ArrayRef,
    vx_array
);

/// Check if array's dtype is nullable.
/// As a particular example, a Null array is nullable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_array_is_nullable(array: *const vx_array) -> bool {
    if array.is_null() {
        return false;
    }
    vx_array::as_ref(array).dtype().is_nullable()
}

/// Check array's dtype against a variant.
/// Equivalent to vx_get_dtype_variant(vx_array_dtype(array)).
///
/// Example:
///
/// const vx_array* array = vx_array_new_null(1);
/// assert(vx_array_has_dtype(array, DTYPE_NULL));
/// vx_array_free(array);
///
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_has_dtype(
    array: *const vx_array,
    variant: vx_dtype_variant,
) -> bool {
    if array.is_null() {
        return false;
    }
    let other: vx_dtype_variant = vx_array::as_ref(array).dtype().into();
    other == variant
}

/// Check whether array has a Primitive dtype with a specific ptype.
///
/// const vx_array* array = vx_array_new_null(1);
/// assert(!vx_array_is_primitive(array, PTYPE_U32));
/// vx_array_free(array);
///
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_is_primitive(
    array: *const vx_array,
    ptype: vx_ptype,
) -> bool {
    if array.is_null() {
        return false;
    }
    let ptype = ptype.into();
    match vx_array::as_ref(array).dtype() {
        DType::Primitive(other, _) => other == &ptype,
        _ => false,
    }
}

#[repr(C)]
pub enum vx_validity_type {
    /// Items can't be null
    VX_VALIDITY_NON_NULLABLE = 0,
    /// All items are valid
    VX_VALIDITY_ALL_VALID = 1,
    /// All items are invalid
    VX_VALIDITY_ALL_INVALID = 2,
    /// Items validity is determined by a boolean array. True values in boolean
    /// array are valid, false values are invalid (null)
    VX_VALIDITY_ARRAY = 3,
}

#[repr(C)]
pub struct vx_validity {
    pub r#type: vx_validity_type,
    /// If type is not VX_VALIDITY_ARRAY, this is NULL.
    /// If type is VX_VALIDITY_ARRAY, this is set to an owned boolean validity
    /// array which must be freed by the caller.
    pub array: *const vx_array,
}

impl From<&vx_validity> for Validity {
    fn from(validity: &vx_validity) -> Self {
        match validity.r#type {
            vx_validity_type::VX_VALIDITY_NON_NULLABLE => Validity::NonNullable,
            vx_validity_type::VX_VALIDITY_ALL_VALID => Validity::AllValid,
            vx_validity_type::VX_VALIDITY_ALL_INVALID => Validity::AllInvalid,
            vx_validity_type::VX_VALIDITY_ARRAY => {
                Validity::Array(vx_array::as_ref(validity.array).clone())
            }
        }
    }
}

impl From<Validity> for vx_validity {
    fn from(validity: Validity) -> Self {
        match validity {
            Validity::NonNullable => vx_validity {
                r#type: vx_validity_type::VX_VALIDITY_NON_NULLABLE,
                array: ptr::null(),
            },
            Validity::AllValid => vx_validity {
                r#type: vx_validity_type::VX_VALIDITY_ALL_VALID,
                array: ptr::null(),
            },
            Validity::AllInvalid => vx_validity {
                r#type: vx_validity_type::VX_VALIDITY_ALL_INVALID,
                array: ptr::null(),
            },
            Validity::Array(array) => vx_validity {
                r#type: vx_validity_type::VX_VALIDITY_ARRAY,
                array: vx_array::new(Arc::new(array)),
            },
        }
    }
}

/// Return array's validity as a type and a boolean array.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_get_validity(
    array: *const vx_array,
    validity: *mut vx_validity,
    error: *mut *mut vx_error,
) {
    try_or_default(error, || {
        vortex_ensure!(!array.is_null());
        vortex_ensure!(!validity.is_null());
        let array = vx_array::as_ref(array);
        *unsafe { &mut *validity } = array.validity()?.into();
        Ok(())
    });
}

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
    index: usize,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error_out, || {
        let array = vx_array::as_ref(array);

        #[expect(deprecated)]
        let field_array = array
            .to_struct()
            .unmasked_fields()
            .get(index)
            .ok_or_else(|| vortex_err!("Field index out of bounds"))?
            .clone();

        Ok(vx_array::new(Arc::new(field_array)))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_slice(
    array: *const vx_array,
    start: usize,
    stop: usize,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error_out, || {
        let array = vx_array::as_ref(array);
        let sliced = array.slice(start..stop)?;
        Ok(vx_array::new(Arc::new(sliced)))
    })
}

/// Check whether array's element at index is invalid (null) according to the
/// validity array. Sets error if index is out of bounds or underlying validity
/// array is corrupted.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_element_is_invalid(
    array: *const vx_array,
    index: usize,
    error: *mut *mut vx_error,
) -> bool {
    try_or_default(error, || {
        vortex_ensure!(!array.is_null());
        vx_array::as_ref(array).is_invalid(index, &mut LEGACY_SESSION.create_execution_ctx())
    })
}

/// Check how many items in the array are invalid (null).
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_invalid_count(
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) -> usize {
    try_or_default(error_out, || {
        vortex_ensure!(!array.is_null());
        let array = vx_array::as_ref(array);
        array.invalid_count(&mut LEGACY_SESSION.create_execution_ctx())
    })
}

/// Create a new array with DTYPE_NULL dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_new_null(len: usize) -> *const vx_array {
    vx_array::new(Arc::new(NullArray::new(len).into_array()))
}

/// SAFETY:
/// `ptr` must be valid for `len` reads of `T`, properly aligned,
/// and must not be null if `len > 0`.
unsafe fn primitive_from_raw<T: vortex::dtype::NativePType>(
    ptr: *const T,
    len: usize,
    validity: &vx_validity,
) -> *const vx_array {
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    let buffer = Buffer::copy_from(slice);
    let array = PrimitiveArray::new(buffer, validity.into());
    vx_array::new(Arc::new(array.into_array()))
}

/// Create a new primitive array from an existing buffer.
/// It is caller's responsibility to ensure ptr points to a buffer of correct
/// type. ptr buffer contents are copied.
/// validity can't be NULL.
///
/// Example:
///
/// const vx_error* error = NULL;
/// vx_validity validity = {};
/// validity.type = VX_VALIDITY_NON_NULLABLE;
/// uint32_t buffer[] = {1, 2, 3};
/// const vx_array* array = vx_array_new_primitive(PTYPE_U32, buffer, 3,
///     &validity, &error);
/// vx_array_free(array);
///
#[unsafe(no_mangle)]
pub extern "C-unwind" fn vx_array_new_primitive(
    ptype: vx_ptype,
    ptr: *const c_void,
    len: usize,
    validity: *const vx_validity,
    error: *mut *mut vx_error,
) -> *const vx_array {
    if validity.is_null() {
        write_error(error, "validity is NULL");
        return ptr::null_mut();
    }
    let validity = unsafe { &*validity };

    match ptype {
        vx_ptype::PTYPE_U8 => unsafe { primitive_from_raw(ptr as *const u8, len, validity) },
        vx_ptype::PTYPE_U16 => unsafe { primitive_from_raw(ptr as *const u16, len, validity) },
        vx_ptype::PTYPE_U32 => unsafe { primitive_from_raw(ptr as *const u32, len, validity) },
        vx_ptype::PTYPE_U64 => unsafe { primitive_from_raw(ptr as *const u64, len, validity) },
        vx_ptype::PTYPE_I8 => unsafe { primitive_from_raw(ptr as *const i8, len, validity) },
        vx_ptype::PTYPE_I16 => unsafe { primitive_from_raw(ptr as *const i16, len, validity) },
        vx_ptype::PTYPE_I32 => unsafe { primitive_from_raw(ptr as *const i32, len, validity) },
        vx_ptype::PTYPE_I64 => unsafe { primitive_from_raw(ptr as *const i64, len, validity) },
        vx_ptype::PTYPE_F16 => unsafe { primitive_from_raw(ptr as *const f16, len, validity) },
        vx_ptype::PTYPE_F32 => unsafe { primitive_from_raw(ptr as *const f32, len, validity) },
        vx_ptype::PTYPE_F64 => unsafe { primitive_from_raw(ptr as *const f64, len, validity) },
    }
}

macro_rules! ffiarray_get_ptype {
    ($ptype:ident) => {
        paste! {
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_get_ $ptype>](array: *const vx_array, index: usize) -> $ptype {
                let array = vx_array::as_ref(array);
                // TODO(joe): propagate this error up instead of expecting
                let value = array
                    .execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at failed");
                // TODO(joe): propagate this error up instead of expecting
                value.as_primitive()
                    .as_::<$ptype>()
                    .vortex_expect("null value")
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_array_get_storage_ $ptype>](array: *const vx_array, index: usize) -> $ptype {
                let array = vx_array::as_ref(array);
                // TODO(joe): propagate this error up instead of expecting
                let value = array
                    .execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at failed");
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
        .execute_scalar(index as usize, &mut LEGACY_SESSION.create_execution_ctx())
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
        .execute_scalar(index as usize, &mut LEGACY_SESSION.create_execution_ctx())
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
        Ok(vx_array::new(Arc::new(array.clone().apply(expression)?)))
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
    use vortex::array::arrays::bool::BoolArrayExt;
    use vortex::array::validity::Validity;
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
    use crate::error::vx_error_get_message;
    use crate::expression::vx_expression_free;
    use crate::string::vx_string_free;

    fn assert_no_error(error: *mut vx_error) {
        if !error.is_null() {
            let message;
            unsafe {
                message = vx_string::as_str(vx_error_get_message(error)).to_owned();
                vx_error_free(error);
            }
            panic!("{message}");
        }
    }

    #[test]
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
    fn test_simple() {
        unsafe {
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let ffi_array = vx_array::new(Arc::new(primitive.into_array()));

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
    #[cfg_attr(miri, ignore)]
    fn test_simple_is() {
        unsafe {
            let primitive =
                PrimitiveArray::new(buffer![1i32, 2i32, 3i32, 4i32, 5i32], Validity::NonNullable);
            let array = vx_array::new(Arc::new(primitive.into_array()));
            assert!(!vx_array_is_nullable(array));
            assert!(vx_array_is_primitive(array, vx_ptype::PTYPE_I32));
            vx_array_free(array);
        }
    }

    #[test]
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
    fn test_slice() {
        unsafe {
            let primitive =
                PrimitiveArray::new(buffer![1i32, 2i32, 3i32, 4i32, 5i32], Validity::NonNullable);
            let ffi_array = vx_array::new(Arc::new(primitive.into_array()));

            let mut error = ptr::null_mut();
            let sliced = vx_array_slice(ffi_array, 1, 4, &raw mut error);
            assert_no_error(error);
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
            let ffi_array = vx_array::new(Arc::new(primitive.into_array()));

            let mut error = ptr::null_mut();
            assert!(!vx_array_element_is_invalid(ffi_array, 0, &raw mut error));
            assert_no_error(error);
            assert!(vx_array_element_is_invalid(ffi_array, 1, &raw mut error));
            assert_no_error(error);
            assert!(!vx_array_element_is_invalid(ffi_array, 2, &raw mut error));
            assert_no_error(error);

            let null_count = vx_array_invalid_count(ffi_array, &raw mut error);
            assert_no_error(error);
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
            let ffi_array = vx_array::new(Arc::new(struct_array.into_array()));

            let mut error = ptr::null_mut();
            let field0 = vx_array_get_field(ffi_array, 0, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_array_len(field0), 3);

            let field1 = vx_array_get_field(ffi_array, 1, &raw mut error);
            assert_no_error(error);
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
            let mut error = ptr::null_mut();
            let validity = vx_validity {
                r#type: vx_validity_type::VX_VALIDITY_NON_NULLABLE,
                array: ptr::null(),
            };

            let i32_array = [i32::MAX, i32::MIN, 0];
            let ffi_i32 = vx_array_new_primitive(
                vx_ptype::PTYPE_I32,
                i32_array.as_ptr() as *const c_void,
                i32_array.len(),
                &raw const validity,
                &raw mut error,
            );
            assert_no_error(error);
            assert!(!ffi_i32.is_null());

            assert!(vx_array_is_primitive(ffi_i32, vx_ptype::PTYPE_I32));
            assert_eq!(vx_array_get_i32(ffi_i32, 0), i32::MAX);
            assert_eq!(vx_array_get_i32(ffi_i32, 1), i32::MIN);
            assert_eq!(vx_array_get_i32(ffi_i32, 2), 0);
            vx_array_free(ffi_i32);

            // Test unsigned integer
            let u64_array = [u64::MAX, 0u64, 42u64];
            let ffi_u64 = vx_array_new_primitive(
                vx_ptype::PTYPE_U64,
                u64_array.as_ptr() as *const c_void,
                u64_array.len(),
                &raw const validity,
                &raw mut error,
            );
            assert_no_error(error);
            assert!(!ffi_u64.is_null());
            assert!(vx_array_is_primitive(ffi_u64, vx_ptype::PTYPE_U64));
            assert_eq!(vx_array_get_u64(ffi_u64, 0), u64::MAX);
            assert_eq!(vx_array_get_u64(ffi_u64, 1), 0);
            assert_eq!(vx_array_get_u64(ffi_u64, 2), 42);
            vx_array_free(ffi_u64);

            // Test floating point including special values
            let f64_array = [f64::NEG_INFINITY, 0.0f64, f64::NAN];
            let ffi_f64 = vx_array_new_primitive(
                vx_ptype::PTYPE_F64,
                f64_array.as_ptr() as *const c_void,
                f64_array.len(),
                &raw const validity,
                &raw mut error,
            );
            assert_no_error(error);
            assert!(!ffi_f64.is_null());
            assert!(vx_array_is_primitive(ffi_f64, vx_ptype::PTYPE_F64));
            assert_eq!(vx_array_get_f64(ffi_f64, 0), f64::NEG_INFINITY);
            assert_eq!(vx_array_get_f64(ffi_f64, 1), 0.0);
            assert!(vx_array_get_f64(ffi_f64, 2).is_nan());
            vx_array_free(ffi_f64);

            // Test f16 (special half-precision type) - skip in Miri due to inline assembly
            #[cfg(not(miri))]
            {
                let f16_array = [f16::from_f32(1.0), f16::from_f32(-0.5)];
                let ffi_f16 = vx_array_new_primitive(
                    vx_ptype::PTYPE_F16,
                    f16_array.as_ptr() as *const c_void,
                    f16_array.len(),
                    &raw const validity,
                    &raw mut error,
                );
                assert_no_error(error);
                assert!(!ffi_f16.is_null());
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
            let ffi_array = vx_array::new(Arc::new(utf8_array.into_array()));
            assert!(vx_array_has_dtype(ffi_array, vx_dtype_variant::DTYPE_UTF8));

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
            let ffi_array = vx_array::new(Arc::new(binary_array.into_array()));
            assert!(vx_array_has_dtype(
                ffi_array,
                vx_dtype_variant::DTYPE_BINARY
            ));

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

            let array = vx_array::new(Arc::new(primitive.into_array()));

            let res = vx_array_apply(array, ptr::null(), &raw mut error);
            assert!(res.is_null());
            assert!(!error.is_null());
            vx_error_free(error);

            // Test with Vortex Rust-side expressions here, test C API for
            // expressions in src/expressions.rs
            let expression = eq(root(), lit(3i32));
            let expression = vx_expression::new(expression);

            let res = vx_array_apply(ptr::null(), expression, &raw mut error);
            assert!(res.is_null());
            assert!(!error.is_null());
            vx_error_free(error);

            let res = vx_array_apply(array, expression, &raw mut error);
            assert_no_error(error);
            assert!(!res.is_null());
            {
                let res = vx_array::as_ref(res);
                #[expect(deprecated)]
                let bool_array = res.to_bool();
                let buffer = bool_array.to_bit_buffer();
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
        let vx_arr = vx_array::new(Arc::new(array));
        assert!(unsafe { vx_array_has_dtype(vx_arr, vx_dtype_variant::DTYPE_STRUCT) });

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
