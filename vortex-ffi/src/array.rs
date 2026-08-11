// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FFI interface for working with Vortex Arrays.
use std::ffi::c_void;
use std::ptr;
use std::ptr::NonNull;

use arrow_array::array::make_array;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi::from_ffi;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Bool;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::NullArray;
use vortex::array::arrays::Primitive;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinView;
use vortex::array::arrays::bool::BoolArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::legacy_session;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::half::f16;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
use vortex_arrow::FromArrowArray;

use crate::box_wrapper;
use crate::dtype::vx_dtype;
use crate::dtype::vx_dtype_variant;
use crate::error::try_or;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::error::write_error;
use crate::expression::vx_expression;
use crate::ptype::vx_ptype;
use crate::scalar::vx_scalar;
use crate::session::vx_session;
use crate::session::vx_session_ref;
use crate::vx_view;

box_wrapper!(
    /// Arrays are reference-counted handles to owned memory buffers that hold
    /// scalars. These buffers can be held in a number of physical encodings to
    /// perform lightweight compression that exploits the particular data
    /// distribution of the array's values.
    ///
    /// Every data type recognized by Vortex also has a canonical physical
    /// encoding format, which arrays can be canonicalized into for ease of
    /// access in compute functions.
    ///
    /// Cloning an array is a cheap operation.
    ///
    /// Unless stated explicitly, all operations with vx_array don't take
    /// ownership of it, and thus the array must be freed by the caller.
    ArrayRef,
    vx_array
);

/// Readonly view over bitpacked booleans.
///
/// "elements" is the number of bits/elements. Use vx_bool_view_words(view) to
/// get the number of uint8_t words.
/// Bits are laid out LSB-first.
///
/// "bit_offset" is in [0; 8) and lets a view start at a non-byte-aligned bit.
/// Use vx_bool_view_nth(view, index) macro to read a single element.
///
/// Example:
/// "view" holds 6 boolean elements, bit_offset=2, first 5 elements are "true",
/// last is "false".
///
/// uint8_t word = 0b01111100;
/// vx_bool_view view = {&word, 6, 2};
#[repr(C)]
pub struct vx_bool_view {
    /// Element 0 is bit "bit_offset" of "ptr".
    pub ptr: *const u8,
    /// Number of elements represented by "ptr".
    pub elements: usize,
    /// Bit offset of element 0 within the first byte of "ptr".
    pub bit_offset: usize,
}

impl vx_bool_view {
    /// {NULL, 0, 0}
    const fn null() -> vx_bool_view {
        vx_bool_view {
            ptr: ptr::null(),
            elements: 0,
            bit_offset: 0,
        }
    }

    fn len(&self) -> usize {
        (self.elements + self.bit_offset).div_ceil(8)
    }
}

/// Borrow the [`ArrayRef`] behind a [`vx_array`] handle, erroring on a null pointer.
///
/// A building block for FFI crates layered on top of the base Vortex C API.
///
/// # Safety
///
/// `array` must be null or a valid `vx_array` pointer created by this crate, and must stay valid
/// for the returned reference.
pub unsafe fn vx_array_ref<'a>(array: *const vx_array) -> VortexResult<&'a ArrayRef> {
    vortex_ensure!(!array.is_null(), "null vx_array");
    Ok(vx_array::as_ref(array))
}

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

/// Validity representation for arrays constructed through the C FFI.
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

/// Array validity descriptor used by C FFI constructors.
#[repr(C)]
pub struct vx_validity {
    /// The kind of validity represented by this descriptor.
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
                array: vx_array::new(array),
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

/// Get array's dtype
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_dtype(array: *const vx_array) -> *const vx_dtype {
    vx_dtype::new(vx_array::as_ref(array).dtype().clone())
}

// Return a field for array at index.
// Returns NULL and sets error_out if index is out of bounds or array doesn't
// have dtype DTYPE_STRUCT.
#[unsafe(no_mangle)]
#[allow(clippy::disallowed_methods)]
pub unsafe extern "C-unwind" fn vx_array_get_field(
    array: *const vx_array,
    index: usize,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error_out, || {
        let array = vx_array::as_ref(array);

        let mut ctx = legacy_session().create_execution_ctx();
        let struct_array = array.clone().execute::<StructArray>(&mut ctx)?;
        let field_array = struct_array
            .unmasked_field_opt(index)
            .ok_or_else(|| vortex_err!("Field index out of bounds"))?
            .clone();

        Ok(vx_array::new(field_array))
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
        Ok(vx_array::new(sliced))
    })
}

/// Check whether array's element at index is invalid (null) according to the
/// validity array. Sets error if index is out of bounds or underlying validity
/// array is corrupted.
#[unsafe(no_mangle)]
#[allow(clippy::disallowed_methods)]
pub unsafe extern "C-unwind" fn vx_array_element_is_invalid(
    session: *const vx_session,
    array: *const vx_array,
    index: usize,
    error: *mut *mut vx_error,
) -> bool {
    try_or_default(error, || {
        let session = unsafe { vx_session_ref(session) }?;
        vx_array::as_ref(array).is_invalid(index, &mut session.create_execution_ctx())
    })
}

/// Check how many items in the array are invalid (null).
#[unsafe(no_mangle)]
#[allow(clippy::disallowed_methods)]
pub unsafe extern "C-unwind" fn vx_array_invalid_count(
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) -> usize {
    try_or_default(error_out, || {
        vortex_ensure!(!array.is_null());
        let array = vx_array::as_ref(array);
        array.invalid_count(&mut legacy_session().create_execution_ctx())
    })
}

/// Increase reference count on vx_array
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_clone(ptr: *const vx_array) -> *const vx_array {
    vx_array::new(vx_array::as_ref(ptr).clone())
}

/// Create a new array with DTYPE_NULL dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_new_null(len: usize) -> *const vx_array {
    vx_array::new(NullArray::new(len).into_array())
}

/// SAFETY:
/// `ptr` must be valid for `len` reads of `T`, properly aligned,
/// and must not be null if `len > 0`.
unsafe fn primitive_from_raw<T: vortex::dtype::NativePType>(
    ptr: *const T,
    len: usize,
    validity: &vx_validity,
    error: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error, || {
        let slice = if ptr.is_null() {
            unsafe { std::slice::from_raw_parts(NonNull::dangling().as_ptr(), len) }
        } else {
            unsafe { std::slice::from_raw_parts(ptr, len) }
        };
        let buffer = Buffer::copy_from(slice);
        let array = PrimitiveArray::try_new(buffer, validity.into())?;
        Ok(vx_array::new(array.into_array()))
    })
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
        vx_ptype::PTYPE_U8 => unsafe { primitive_from_raw(ptr as *const u8, len, validity, error) },
        vx_ptype::PTYPE_U16 => unsafe {
            primitive_from_raw(ptr as *const u16, len, validity, error)
        },
        vx_ptype::PTYPE_U32 => unsafe {
            primitive_from_raw(ptr as *const u32, len, validity, error)
        },
        vx_ptype::PTYPE_U64 => unsafe {
            primitive_from_raw(ptr as *const u64, len, validity, error)
        },
        vx_ptype::PTYPE_I8 => unsafe { primitive_from_raw(ptr as *const i8, len, validity, error) },
        vx_ptype::PTYPE_I16 => unsafe {
            primitive_from_raw(ptr as *const i16, len, validity, error)
        },
        vx_ptype::PTYPE_I32 => unsafe {
            primitive_from_raw(ptr as *const i32, len, validity, error)
        },
        vx_ptype::PTYPE_I64 => unsafe {
            primitive_from_raw(ptr as *const i64, len, validity, error)
        },
        vx_ptype::PTYPE_F16 => unsafe {
            primitive_from_raw(ptr as *const f16, len, validity, error)
        },
        vx_ptype::PTYPE_F32 => unsafe {
            primitive_from_raw(ptr as *const f32, len, validity, error)
        },
        vx_ptype::PTYPE_F64 => unsafe {
            primitive_from_raw(ptr as *const f64, len, validity, error)
        },
    }
}

/// Create a new Bool array from vx_bool_view.
///
/// Example:
///
/// a Bool array with 9 elements, first 8 are "true", last is "false".
///
/// const vx_error* error = NULL;
/// vx_validity validity = {};
/// validity.type = VX_VALIDITY_NON_NULLABLE;
///
/// uint8_t words[2] = {0xff, 0}; // 11111111 00000000
/// vx_bool_view view = {words, 9, 0};
///
/// const vx_array* array = vx_array_new_bool(&view, &validity, &error);
/// vx_array_free(array);
#[unsafe(no_mangle)]
pub extern "C-unwind" fn vx_array_new_bool(
    view: *const vx_bool_view,
    validity: *const vx_validity,
    error: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error, || {
        vortex_ensure!(!view.is_null());
        vortex_ensure!(!validity.is_null());
        let bits = unsafe { &*view };
        let validity = unsafe { &*validity };
        vortex_ensure!(bits.bit_offset < 8, "bit_offset must be in [0; 8)");
        let byte_len = bits.len();

        let slice = if bits.ptr.is_null() {
            unsafe { std::slice::from_raw_parts(NonNull::dangling().as_ptr(), byte_len) }
        } else {
            unsafe { std::slice::from_raw_parts(bits.ptr, byte_len) }
        };
        let buffer = Buffer::copy_from(slice);
        let bits = BitBuffer::new_with_offset(buffer, bits.elements, bits.bit_offset);
        let array = BoolArray::try_new(bits, validity.into())?;
        Ok(vx_array::new(array.into_array()))
    })
}

/// Create a Vortex array by importing an Arrow array via the Arrow C Data Interface.
///
/// `array` and `schema` together describe a single Arrow array (the standard Arrow C Data
/// Interface pair, e.g. as produced by exporting a record batch). Both are *consumed*: their
/// `release` callbacks are invoked by this function and the caller must not use or release them
/// afterwards.
///
/// `nullable` controls the top-level nullability of the resulting array's dtype. For an Arrow
/// record batch (which has no top-level validity) pass `false`.
///
/// On error, returns NULL and sets `error_out`.
///
/// Example:
///
/// // export an Arrow record batch into (array, schema), then:
/// vx_error* error = NULL;
/// const vx_array* vx = vx_array_from_arrow(&array, &schema, false, &error);
/// // ... push it to a sink or write it ...
/// vx_array_free(vx);
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_from_arrow(
    array: *mut FFI_ArrowArray,
    schema: *mut FFI_ArrowSchema,
    nullable: bool,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error_out, || {
        vortex_ensure!(!array.is_null(), "null arrow array");
        vortex_ensure!(!schema.is_null(), "null arrow schema");
        let ffi_array = unsafe { ptr::replace(array, FFI_ArrowArray::empty()) };
        let ffi_schema = unsafe { ptr::replace(schema, FFI_ArrowSchema::empty()) };
        let array_data = unsafe { from_ffi(ffi_array, &ffi_schema) }?;
        drop(ffi_schema);
        let arrow_array = make_array(array_data);
        let vortex_array = ArrayRef::from_arrow(arrow_array.as_ref(), nullable)?;
        Ok(vx_array::new(vortex_array))
    })
}

/// SAFETY: "array" must be null or a valid "vx_array"
unsafe fn varbinview_at(
    array: *const vx_array,
    index: usize,
    want_utf8: bool,
    error_out: *mut *mut vx_error,
) -> vx_view {
    try_or(error_out, vx_view::null(), || {
        let array = unsafe { vx_array_ref(array) }?;
        vortex_ensure!(index < array.len(), "index {index} out of bounds");
        let dtype_matches = if want_utf8 {
            matches!(array.dtype(), DType::Utf8(_))
        } else {
            matches!(array.dtype(), DType::Binary(_))
        };
        vortex_ensure!(
            dtype_matches,
            "expected a {} array, got {}",
            if want_utf8 { "Utf8" } else { "Binary" },
            array.dtype()
        );
        let Some(views) = array.as_opt::<VarBinView>() else {
            vortex_bail!("expected a canonical array, got {}", array.encoding_id());
        };
        Ok(vx_view::from_bytes(views.bytes_at(index).as_slice()))
    })
}

/// Return UTF-8 string at "index" in a canonical Utf8 array.
///
/// For invalid elements the returned value is unspecified, check validity via
/// vx_array_get_validity.
/// Returned view is valid as long as "array" is valid.
/// Errors if index is out of bounds or array is not a canonical Utf8 array.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_utf8_at(
    array: *const vx_array,
    index: usize,
    error_out: *mut *mut vx_error,
) -> vx_view {
    unsafe { varbinview_at(array, index, true, error_out) }
}

/// Return a binary string at "index" in a canonical Binary array.
///
/// For invalid elements the returned value is unspecified, check validity via
/// vx_array_get_validity.
/// Returned view is valid as long as "array" is valid.
/// Errors if index is out of bounds or array is not a canonical Binary array.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_binary_at(
    array: *const vx_array,
    index: usize,
    error_out: *mut *mut vx_error,
) -> vx_view {
    unsafe { varbinview_at(array, index, false, error_out) }
}

/// For a canonical Bool array, return bool at "index".
/// For invalid elements returned value is unspecified, check validity via
/// vx_array_get_validity.
///
/// Panics if "array" is not canonical - call vx_array_canonicalize first.
/// Panics if "array" is not a Bool array.
/// Panics if "index" is out of bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_get_bool(array: *const vx_array, index: usize) -> bool {
    let array = vx_array::as_ref(array);
    let bool_array = array
        .as_opt::<Bool>()
        .vortex_expect("vx_array_get_bool requires a canonical Bool array");
    let bits = bool_array.to_bit_buffer();
    if index >= bits.len() {
        vortex_panic!(
            "index {index} out of bounds for array of length {}",
            bits.len()
        );
    }
    bits.value(index)
}

/// Get array's element at position "index".
///
/// If element at index is invalid, returns a Null vx_scalar.
///
/// This operation executes the array to extract a scalar and thus is
/// expensive. If you need bulk access, use
/// vx_array_data_ptr_primitive or vx_data_ptr_bool.
///
/// Errors if "index" is out of bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_get_scalar(
    session: *const vx_session,
    array: *const vx_array,
    index: usize,
    error_out: *mut *mut vx_error,
) -> *const vx_scalar {
    try_or_default(error_out, || {
        let session = vx_session::as_ref(session);
        let array = vx_array::as_ref(array);
        let scalar = array.execute_scalar(index, &mut session.create_execution_ctx())?;
        Ok(vx_scalar::new(scalar))
    })
}

/// Decode array into its canonical form.
///
/// On error returns NULL and "sets error_out".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_canonicalize(
    session: *const vx_session,
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error_out, || {
        let session = vx_session::as_ref(session);
        let array = vx_array::as_ref(array);
        let mut ctx = session.create_execution_ctx();
        let canonical = array.clone().execute::<Canonical>(&mut ctx)?;
        Ok(vx_array::new(canonical.into_array()))
    })
}

/// Return a pointer to the values buffer of a canonical Primitive array.
/// Pointer is valid as long as "array" is valid.
///
/// Errors if array is not a canonical Primitive.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_data_ptr_primitive(
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) -> *const c_void {
    try_or(error_out, ptr::null(), || {
        let array = vx_array::as_ref(array);
        let primitive = array.as_opt::<Primitive>().ok_or_else(|| {
            vortex_err!(
                "vx_array_data_ptr_primitive requires a canonical Primitive array, got {}",
                array.encoding_id()
            )
        })?;
        let bytes = primitive
            .buffer_handle()
            .as_host_opt()
            .ok_or_else(|| vortex_err!("array buffer is not in host memory"))?;
        Ok(bytes.as_ptr().cast())
    })
}

/// Return vx_bool_view for a canonical Bool array.
/// View is valid as long as "array" is valid.
///
/// Errors if array is not a canonical Bool.
///
/// Example:
///
/// vx_validity validity = {};
/// validity.type = VX_VALIDITY_NON_NULLABLE;
///
/// uint8_t words[2] = {0xff, 0}; // 11111111 00000000
/// vx_bool_view view = {words, 9, 0};
///
/// const vx_array* array = vx_array_new_bool(&view, &validity, &error);
/// vx_bool_view other = vx_array_data_ptr_bool(array, &error);
///
/// vx_array_free(array);
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_data_ptr_bool(
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) -> vx_bool_view {
    try_or(error_out, vx_bool_view::null(), || {
        let array = unsafe { vx_array_ref(array) }?;
        let array = array.as_opt::<Bool>().ok_or_else(|| {
            let id = array.encoding_id();
            vortex_err!("vx_array_data_ptr_bool requires a canonical Bool array, got {id}")
        })?;
        let bits = array.to_bit_buffer();
        let ptr = bits.inner().as_ptr().cast();
        let elements = bits.len();
        let bit_offset = bits.offset();
        Ok(vx_bool_view {
            ptr,
            elements,
            bit_offset,
        })
    })
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
        Ok(vx_array::new(array.clone().apply(expression)?))
    })
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use std::slice::from_raw_parts;
    use std::sync::Arc;

    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::array_session;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::arrays::bool::BoolArrayExt;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::expr::eq;
    use vortex::expr::lit;
    use vortex::expr::root;

    use crate::array::*;
    use crate::dtype::vx_dtype_free;
    use crate::dtype::vx_dtype_get_variant;
    use crate::dtype::vx_dtype_variant;
    use crate::error::vx_error_free;
    use crate::expression::vx_expression_free;
    use crate::scalar::*;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;
    use crate::tests::assert_error;
    use crate::tests::assert_no_error;

    unsafe fn get_i32(session: *const vx_session, array: *const vx_array, index: usize) -> i32 {
        let mut error = ptr::null_mut();
        let scalar = unsafe { vx_array_get_scalar(session, array, index, &raw mut error) };
        assert_no_error(error);
        let value = unsafe { vx_scalar_get_i32(scalar) };
        unsafe { vx_scalar_free(scalar.cast_mut()) };
        value
    }

    unsafe fn get_u8(session: *const vx_session, array: *const vx_array, index: usize) -> u8 {
        let mut error = ptr::null_mut();
        let scalar = unsafe { vx_array_get_scalar(session, array, index, &raw mut error) };
        assert_no_error(error);
        let value = unsafe { vx_scalar_get_u8(scalar) };
        unsafe { vx_scalar_free(scalar.cast_mut()) };
        value
    }

    #[test]
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
    fn test_simple() {
        unsafe {
            let session = vx_session_new();
            let primitive = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let ffi_array = vx_array::new(primitive.into_array());

            assert_eq!(vx_array_len(ffi_array), 3);

            let array_dtype = vx_array_dtype(ffi_array);
            assert_eq!(
                vx_dtype_get_variant(array_dtype),
                vx_dtype_variant::DTYPE_PRIMITIVE
            );

            let mut error = ptr::null_mut();

            let scalar = vx_array_get_scalar(session, ffi_array, 0, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_scalar_get_i32(scalar), 1);
            vx_scalar_free(scalar);

            let scalar = vx_array_get_scalar(session, ffi_array, 1, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_scalar_get_i32(scalar), 2);
            vx_scalar_free(scalar);

            let scalar = vx_array_get_scalar(session, ffi_array, 2, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_scalar_get_i32(scalar), 3);
            vx_scalar_free(scalar);

            vx_dtype_free(array_dtype);
            vx_array_free(ffi_array);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_simple_is() {
        unsafe {
            let primitive =
                PrimitiveArray::new(buffer![1i32, 2i32, 3i32, 4i32, 5i32], Validity::NonNullable);
            let array = vx_array::new(primitive.into_array());
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
            let session = vx_session_new();
            let primitive =
                PrimitiveArray::new(buffer![1i32, 2i32, 3i32, 4i32, 5i32], Validity::NonNullable);
            let ffi_array = vx_array::new(primitive.into_array());

            let mut error = ptr::null_mut();
            let sliced = vx_array_slice(ffi_array, 1, 4, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_array_len(sliced), 3);
            assert_eq!(get_i32(session, sliced, 0), 2);
            assert_eq!(get_i32(session, sliced, 2), 4);

            vx_array_free(sliced);
            vx_array_free(ffi_array);
            vx_session_free(session);
        }
    }

    #[test]
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
    fn test_null_operations() {
        unsafe {
            let session = vx_session_new();
            let primitive = PrimitiveArray::new(
                buffer![1i32, 2i32, 3i32],
                Validity::from_iter([true, false, true]),
            );
            let ffi_array = vx_array::new(primitive.into_array());

            let mut error = ptr::null_mut();
            assert!(!vx_array_element_is_invalid(
                session,
                ffi_array,
                0,
                &raw mut error
            ));
            assert_no_error(error);
            assert!(vx_array_element_is_invalid(
                session,
                ffi_array,
                1,
                &raw mut error
            ));
            assert_no_error(error);
            assert!(!vx_array_element_is_invalid(
                session,
                ffi_array,
                2,
                &raw mut error
            ));
            assert_no_error(error);

            let null_count = vx_array_invalid_count(ffi_array, &raw mut error);
            assert_no_error(error);
            assert_eq!(null_count, 1);

            vx_array_free(ffi_array);
            vx_session_free(session);
        }
    }

    #[test]
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
    fn test_get_field() {
        unsafe {
            let session = vx_session_new();
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
            assert_no_error(error);
            assert_eq!(vx_array_len(field0), 3);

            let field1 = vx_array_get_field(ffi_array, 1, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_array_len(field1), 3);
            assert_eq!(get_u8(session, field1, 0), 30);
            assert_eq!(get_u8(session, field1, 1), 25);
            assert_eq!(get_u8(session, field1, 2), 35);

            // Test out of bounds
            let field_oob = vx_array_get_field(ffi_array, 2, &raw mut error);
            assert!(!error.is_null());
            assert!(field_oob.is_null());
            vx_error_free(error);

            vx_array_free(field0);
            vx_array_free(field1);
            vx_array_free(ffi_array);
            vx_session_free(session);
        }
    }
    #[test]
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
    fn test_utf8_binary_at() {
        unsafe {
            let long = "a string that is longer than twelve bytes";
            let utf8_array =
                VarBinViewArray::from_iter_nullable_str([Some("hello"), None, Some(long)]);
            let ffi_array = vx_array::new(utf8_array.into_array());

            let mut error = ptr::null_mut();
            let inlined = vx_array_utf8_at(ffi_array, 0, &raw mut error);
            assert!(error.is_null());
            assert_eq!(inlined.as_str().unwrap(), "hello");

            vx_array_utf8_at(ffi_array, 1, &raw mut error);
            assert!(error.is_null());

            let buffered = vx_array_utf8_at(ffi_array, 2, &raw mut error);
            assert!(error.is_null());
            assert_eq!(buffered.as_str().unwrap(), long);

            vx_array_utf8_at(ffi_array, 3, &raw mut error);
            assert_error(error);

            vx_array_free(ffi_array);

            let numbers =
                PrimitiveArray::new(buffer![1i32, 2i32], Validity::NonNullable).into_array();
            let ffi_array = vx_array::new(numbers);
            let value = vx_array_utf8_at(ffi_array, 0, &raw mut error);
            assert!(value.ptr.is_null());
            assert_error(error);
            vx_array_free(ffi_array);

            let binary_array = VarBinViewArray::from_iter_bin(vec![vec![0x01, 0x02, 0x03]]);
            let ffi_array = vx_array::new(binary_array.into_array());
            let bin = vx_array_binary_at(ffi_array, 0, &raw mut error);
            assert!(error.is_null());
            assert_eq!(bin.as_bytes().unwrap(), &[0x01, 0x02, 0x03]);
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
                let mut ctx = array_session().create_execution_ctx();
                let bool_array = res.clone().execute::<BoolArray>(&mut ctx).unwrap();
                let buffer = bool_array.to_bit_buffer();
                let expected = BoolArray::from_iter(vec![false, false, true, true]);
                assert_eq!(buffer, expected.to_bit_buffer());
            }
            vx_array_free(res);

            vx_expression_free(expression);
            vx_array_free(array);
        }
    }

    // TODO: re-enable under miri once parking_lot_core fixes strict-provenance violations
    #[cfg_attr(miri, ignore)]
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
        assert!(unsafe { vx_array_has_dtype(vx_arr, vx_dtype_variant::DTYPE_STRUCT) });

        let dtype_ptr = unsafe { vx_array_dtype(vx_arr) };
        let variant = unsafe { vx_dtype_get_variant(dtype_ptr) };
        assert_eq!(variant, vx_dtype_variant::DTYPE_STRUCT);

        unsafe { vx_array_free(vx_arr) };
        unsafe {
            vx_dtype_free(dtype_ptr);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_arrow_roundtrip() {
        use arrow_array::Array as ArrowArrayTrait;
        use arrow_array::Int32Array;
        use arrow_array::RecordBatch;
        use arrow_array::StringArray;
        use arrow_array::ffi::to_ffi;
        use arrow_schema::DataType;
        use arrow_schema::Field;
        use arrow_schema::Schema as ArrowSchema;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("x"), None, Some("z")])),
            ],
        )
        .unwrap();

        let data = ArrowArrayTrait::into_data(arrow_array::StructArray::from(batch));
        let (mut ffi_array, mut ffi_schema) = to_ffi(&data).unwrap();

        let mut error = ptr::null_mut();
        let vx = unsafe {
            vx_array_from_arrow(
                &raw mut ffi_array,
                &raw mut ffi_schema,
                false,
                &raw mut error,
            )
        };
        assert_no_error(error);
        assert!(!vx.is_null());

        unsafe {
            let session = vx_session_new();

            assert!(vx_array_has_dtype(vx, vx_dtype_variant::DTYPE_STRUCT));
            assert_eq!(vx_array_len(vx), 3);
            assert!(!vx_array_is_nullable(vx));

            let a = vx_array_get_field(vx, 0, &raw mut error);
            assert_no_error(error);
            assert!(vx_array_is_primitive(a, vx_ptype::PTYPE_I32));
            assert_eq!(get_i32(session, a, 0), 1);
            assert_eq!(get_i32(session, a, 2), 3);
            vx_array_free(a);

            let b = vx_array_get_field(vx, 1, &raw mut error);
            assert_no_error(error);
            assert!(vx_array_has_dtype(b, vx_dtype_variant::DTYPE_UTF8));
            assert!(vx_array_element_is_invalid(session, b, 1, &raw mut error));
            assert_no_error(error);
            vx_array_free(b);

            vx_array_free(vx);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_canonicalize_nullable() {
        let primitive = PrimitiveArray::new(
            buffer![10i32, 20i32, 30i32],
            Validity::from_iter([true, false, true]),
        );

        unsafe {
            let session = vx_session_new();
            let mut error = ptr::null_mut();

            let array = vx_array::new(primitive.into_array());
            let canonical = vx_array_canonicalize(session, array, &raw mut error);
            assert_no_error(error);
            let data = vx_array_data_ptr_primitive(canonical, &raw mut error);
            assert_no_error(error);

            assert_eq!(from_raw_parts(data.cast::<i32>(), 3), [10, 20, 30]);
            let mut validity = vx_validity {
                r#type: vx_validity_type::VX_VALIDITY_NON_NULLABLE,
                array: ptr::null(),
            };

            vx_array_get_validity(canonical, &raw mut validity, &raw mut error);
            assert_no_error(error);
            assert!(matches!(
                validity.r#type,
                vx_validity_type::VX_VALIDITY_ARRAY
            ));
            let validity_bools = vx_array_canonicalize(session, validity.array, &raw mut error);
            assert_no_error(error);
            let view = vx_array_data_ptr_bool(validity_bools, &raw mut error);
            assert_no_error(error);
            let byte = *view.ptr.add(view.bit_offset / 8);
            assert_eq!((byte >> (view.bit_offset % 8)) & 0b111, 0b101);

            vx_array_free(validity_bools);
            vx_array_free(validity.array);
            vx_array_free(canonical);
            vx_array_free(array);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_canonicalize() {
        let primitive = PrimitiveArray::new(buffer![1u8, 2u8], Validity::NonNullable);

        unsafe {
            let session = vx_session_new();
            let mut error = ptr::null_mut();
            let mut validity = vx_validity {
                r#type: vx_validity_type::VX_VALIDITY_NON_NULLABLE,
                array: ptr::null(),
            };

            let array = vx_array::new(primitive.into_array());
            vx_array_get_validity(array, &raw mut validity, &raw mut error);
            assert_no_error(error);
            assert!(matches!(
                validity.r#type,
                vx_validity_type::VX_VALIDITY_NON_NULLABLE
            ));
            assert!(validity.array.is_null());
            vx_array_free(array);

            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_bool() {
        unsafe {
            let words: [u8; 2] = [u8::MAX, 0];
            let validity = vx_validity {
                r#type: vx_validity_type::VX_VALIDITY_NON_NULLABLE,
                array: ptr::null(),
            };

            let view = vx_bool_view {
                ptr: words.as_ptr(),
                elements: 9,
                bit_offset: 0,
            };

            let mut error = ptr::null_mut();
            let array = vx_array_new_bool(&raw const view, &raw const validity, &raw mut error);
            assert_no_error(error);
            assert!(!array.is_null());
            assert!(vx_array_has_dtype(array, vx_dtype_variant::DTYPE_BOOL));
            assert_eq!(vx_array_len(array), 9);

            for i in 0..8 {
                assert!(vx_array_get_bool(array, i));
            }
            assert!(!vx_array_get_bool(array, 8));

            let other_view = vx_array_data_ptr_bool(array, &raw mut error);
            assert_no_error(error);
            assert_eq!(other_view.elements, 9);
            assert_eq!(other_view.bit_offset, 0);
            assert_eq!(*other_view.ptr, words[0]);

            vx_array_free(array);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_bool_offset() {
        // 6 elements starting at bit 2, first 5 true, last false.
        let word: u8 = 0b01111100;

        unsafe {
            let validity = vx_validity {
                r#type: vx_validity_type::VX_VALIDITY_NON_NULLABLE,
                array: ptr::null(),
            };
            let view = vx_bool_view {
                ptr: &raw const word,
                elements: 6,
                bit_offset: 2,
            };

            let mut error = ptr::null_mut();
            let array = vx_array_new_bool(&raw const view, &raw const validity, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_array_len(array), 6);

            for i in 0..5 {
                assert!(vx_array_get_bool(array, i), "index {i}");
            }
            assert!(!vx_array_get_bool(array, 5));

            vx_array_free(array);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_bool_roundtrip() {
        let expected = [
            true, true, false, true, false, true, true, false, true, true,
        ];

        unsafe {
            let session = vx_session_new();
            let mut error = ptr::null_mut();
            let validity = vx_validity {
                r#type: vx_validity_type::VX_VALIDITY_NON_NULLABLE,
                array: ptr::null(),
            };

            let mut words = vec![0u8; expected.len().div_ceil(8)];
            for (i, value) in expected.iter().enumerate() {
                if *value {
                    words[i / 8] |= 1 << (i % 8);
                }
            }

            let view = vx_bool_view {
                ptr: words.as_ptr(),
                elements: expected.len(),
                bit_offset: 0,
            };
            let array = vx_array_new_bool(&raw const view, &raw const validity, &raw mut error);
            assert_no_error(error);

            let canonical = vx_array_canonicalize(session, array, &raw mut error);
            assert_no_error(error);

            for (i, value) in expected.iter().enumerate() {
                assert_eq!(vx_array_get_bool(canonical, i), *value, "index {i}");
            }

            let bits = vx_array_data_ptr_bool(canonical, &raw mut error);
            assert_no_error(error);
            for (i, value) in expected.iter().enumerate() {
                let bit = bits.bit_offset + i;
                let actual = (*bits.ptr.add(bit / 8) >> (bit % 8)) & 1 == 1;
                assert_eq!(actual, *value, "index {i}");
            }

            vx_array_free(canonical);
            vx_array_free(array);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_data_ptr_bool() {
        let bools = BoolArray::from_iter([
            true, true, false, true, false, true, true, false, true, true,
        ]);

        unsafe {
            let session = vx_session_new();
            let mut error = ptr::null_mut();

            let array = vx_array::new(bools.into_array());
            let sliced = vx_array_slice(array, 3, 10, &raw mut error);
            assert_no_error(error);

            let canonical = vx_array_canonicalize(session, sliced, &raw mut error);
            assert_no_error(error);

            let bits = vx_array_data_ptr_bool(canonical, &raw mut error);
            assert_no_error(error);
            assert!(bits.bit_offset < 8);
            assert_eq!(bits.elements, 7);
            for (i, expected) in [true, false, true, true, false, true, true]
                .into_iter()
                .enumerate()
            {
                let bit = bits.bit_offset + i;
                let actual = (*bits.ptr.add(bit / 8) >> (bit % 8)) & 1 == 1;
                assert_eq!(actual, expected, "bit {i}");
            }
            vx_array_free(canonical);
            vx_array_free(sliced);
            vx_array_free(array);

            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_data_ptr_error() {
        let strings = VarBinViewArray::from_iter_str(["a", "b"]);

        unsafe {
            let mut error = ptr::null_mut();

            let array = vx_array::new(strings.into_array());
            let data = vx_array_data_ptr_primitive(array, &raw mut error);
            assert!(data.is_null());
            assert_error(error);

            let bits = vx_array_data_ptr_bool(array, &raw mut error);
            assert!(bits.ptr.is_null());
            assert_error(error);

            vx_array_free(array);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_get_bool() {
        let bools = BoolArray::from_iter([true, false, true]);
        unsafe {
            let array = vx_array::new(bools.into_array());
            assert!(vx_array_get_bool(array, 0));
            assert!(!vx_array_get_bool(array, 1));
            assert!(vx_array_get_bool(array, 2));
            vx_array_free(array);
        }
    }
}
