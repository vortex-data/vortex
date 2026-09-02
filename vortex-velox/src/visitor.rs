// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_char;
use std::ffi::c_void;
use std::mem::size_of;
use std::slice;
use std::sync::Arc;

use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::primitive::PrimitiveArrayExt;
use vortex::buffer::ByteBuffer;
use vortex::dtype::PType;
use vortex::mask::Mask;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_ffi::try_or;
use vortex_ffi::vx_array;
use vortex_ffi::vx_array_ref;
use vortex_ffi::vx_error;
use vortex_ffi::vx_session;
use vortex_ffi::vx_session_ref;

/// A fixed-width primitive value identifier in a semantic visitor block.
pub type vx_velox_primitive_type = u32;
/// Unsigned 8-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_U8: vx_velox_primitive_type = 0;
/// Unsigned 16-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_U16: vx_velox_primitive_type = 1;
/// Unsigned 32-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_U32: vx_velox_primitive_type = 2;
/// Unsigned 64-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_U64: vx_velox_primitive_type = 3;
/// Signed 8-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_I8: vx_velox_primitive_type = 4;
/// Signed 16-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_I16: vx_velox_primitive_type = 5;
/// Signed 32-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_I32: vx_velox_primitive_type = 6;
/// Signed 64-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_I64: vx_velox_primitive_type = 7;
/// IEEE 754 binary16 primitive identifier.
pub const VX_VELOX_PRIMITIVE_F16: vx_velox_primitive_type = 8;
/// IEEE 754 binary32 primitive identifier.
pub const VX_VELOX_PRIMITIVE_F32: vx_velox_primitive_type = 9;
/// IEEE 754 binary64 primitive identifier.
pub const VX_VELOX_PRIMITIVE_F64: vx_velox_primitive_type = 10;

fn primitive_type_id(value: PType) -> vx_velox_primitive_type {
    match value {
        PType::U8 => VX_VELOX_PRIMITIVE_U8,
        PType::U16 => VX_VELOX_PRIMITIVE_U16,
        PType::U32 => VX_VELOX_PRIMITIVE_U32,
        PType::U64 => VX_VELOX_PRIMITIVE_U64,
        PType::I8 => VX_VELOX_PRIMITIVE_I8,
        PType::I16 => VX_VELOX_PRIMITIVE_I16,
        PType::I32 => VX_VELOX_PRIMITIVE_I32,
        PType::I64 => VX_VELOX_PRIMITIVE_I64,
        PType::F16 => VX_VELOX_PRIMITIVE_F16,
        PType::F32 => VX_VELOX_PRIMITIVE_F32,
        PType::F64 => VX_VELOX_PRIMITIVE_F64,
    }
}

/// A fixed-width validity representation identifier for one visitor block.
pub type vx_velox_validity_kind = u32;
/// The type is not nullable.
pub const VX_VELOX_VALIDITY_NON_NULLABLE: vx_velox_validity_kind = 0;
/// Every value is valid.
pub const VX_VELOX_VALIDITY_ALL_VALID: vx_velox_validity_kind = 1;
/// Every value is null.
pub const VX_VELOX_VALIDITY_ALL_INVALID: vx_velox_validity_kind = 2;
/// A packed bitmap contains one valid bit per value.
pub const VX_VELOX_VALIDITY_BITMAP: vx_velox_validity_kind = 3;

/// A retained owner for buffers in a visitor block.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_buffer_owner {
    /// Set this field to `sizeof(vx_velox_buffer_owner)`.
    pub struct_size: usize,
    /// An opaque retained object.
    pub owner: *const c_void,
    /// Add one owner reference before the callback returns.
    pub retain: Option<unsafe extern "C" fn(owner: *const c_void)>,
    /// Release one retained owner reference.
    pub release: Option<unsafe extern "C" fn(owner: *const c_void)>,
    /// The allocated number of payload bytes retained by this compact owner.
    pub retained_bytes: usize,
}

/// A canonical primitive block delivered to Velox.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_primitive_view {
    /// Set this field to `sizeof(vx_velox_primitive_view)`.
    pub struct_size: usize,
    /// The physical type of each value.
    pub primitive_type: vx_velox_primitive_type,
    /// The number of logical values in the block.
    pub length: usize,
    /// The first value byte.
    pub values: *const u8,
    /// The number of value bytes.
    pub values_length: usize,
    /// The validity representation.
    pub validity_kind: vx_velox_validity_kind,
    /// The first validity byte when `validity_kind` is `Bitmap`.
    pub validity: *const u8,
    /// The number of validity bytes.
    pub validity_length: usize,
    /// The first logical validity bit within `validity`.
    pub validity_bit_offset: usize,
    /// Retains all pointers in this view.
    pub buffers: vx_velox_buffer_owner,
    /// The guaranteed byte alignment of a non-empty values buffer.
    pub values_alignment: usize,
    /// The guaranteed byte alignment of a non-empty validity buffer.
    pub validity_alignment: usize,
}

/// A single-shot subset request for the semantic visitor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_visit_request {
    /// Set this field to `sizeof(vx_velox_visit_request)`.
    pub struct_size: usize,
    /// Unique, increasing source positions. Null selects every row.
    pub rows: *const u64,
    /// The number of source positions.
    pub row_count: usize,
}

/// Host callbacks for Vortex array traversal.
///
/// One array visit calls the primitive callback synchronously. Shared tables can receive concurrent
/// callbacks from simultaneous visits. `last_error` must return the calling thread's most recent
/// error. The string must remain valid until the next callback on that thread. Callbacks must catch
/// foreign exceptions and must not unwind across this ABI. The host owns the context.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct vx_velox_visitor {
    /// Set this field to `sizeof(vx_velox_visitor)`.
    pub struct_size: usize,
    /// Set this field to [`crate::VX_VELOX_ABI_VERSION`].
    pub abi_version: u32,
    /// An opaque callback context.
    pub context: *mut c_void,
    /// Consume one canonical primitive block. Zero means success.
    pub visit_primitive: Option<
        unsafe extern "C" fn(context: *mut c_void, view: *const vx_velox_primitive_view) -> i32,
    >,
    /// Return the last callback error as a null-terminated string.
    pub last_error: Option<unsafe extern "C" fn(context: *mut c_void) -> *const c_char>,
}

struct PrimitiveOwner {
    values: Box<[u64]>,
    values_length: usize,
    validity: Option<Box<[u8]>>,
    retained_bytes: usize,
}

impl PrimitiveOwner {
    fn try_new(
        host_values: &ByteBuffer,
        validity: Option<&vortex::buffer::BitBuffer>,
        length: usize,
    ) -> VortexResult<Self> {
        let values_length = host_values.len();
        let mut values = vec![0_u64; values_length.div_ceil(size_of::<u64>())].into_boxed_slice();
        let values_allocation = values
            .len()
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| vortex_err!("Primitive visitor value byte count overflow"))?;
        if values_length != 0 {
            // SAFETY: The byte view spans the complete initialized `u64` allocation.
            let values_bytes = unsafe {
                slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), values_allocation)
            };
            values_bytes[..values_length].copy_from_slice(host_values.as_slice());
        }
        let validity = validity.map(|validity| {
            let mut compact = vec![0_u8; length.div_ceil(8)].into_boxed_slice();
            for (index, is_valid) in validity.into_iter().take(length).enumerate() {
                if is_valid {
                    compact[index / 8] |= 1 << (index % 8);
                }
            }
            compact
        });
        let retained_bytes = values_allocation
            .checked_add(validity.as_ref().map_or(0, |validity| validity.len()))
            .ok_or_else(|| vortex_err!("Primitive visitor retained byte count overflow"))?;
        Ok(Self {
            values,
            values_length,
            validity,
            retained_bytes,
        })
    }

    fn values(&self) -> *const u8 {
        if self.values_length == 0 {
            std::ptr::null()
        } else {
            self.values.as_ptr().cast()
        }
    }

    fn validity(&self) -> *const u8 {
        self.validity
            .as_ref()
            .filter(|validity| !validity.is_empty())
            .map_or(std::ptr::null(), |validity| validity.as_ptr())
    }

    fn values_length(&self) -> usize {
        self.values_length
    }

    fn validity_length(&self) -> usize {
        self.validity.as_ref().map_or(0, |validity| validity.len())
    }

    fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
}

fn pointer_alignment(pointer: *const u8) -> usize {
    if pointer.is_null() {
        return 0;
    }
    1usize << pointer.addr().trailing_zeros()
}

unsafe extern "C" fn retain_primitive_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<PrimitiveOwner>()) };
}

unsafe extern "C" fn release_primitive_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<PrimitiveOwner>()) });
}

fn validate_visitor(visitor: &vx_velox_visitor) -> VortexResult<()> {
    if visitor.struct_size < size_of::<vx_velox_visitor>() {
        vortex_bail!(
            "Vortex Velox visitor structure is too small: expected at least {}, got {}",
            size_of::<vx_velox_visitor>(),
            visitor.struct_size
        );
    }
    if visitor.abi_version != crate::VX_VELOX_ABI_VERSION {
        vortex_bail!(
            "Unsupported Vortex Velox ABI version: expected {}, got {}",
            crate::VX_VELOX_ABI_VERSION,
            visitor.abi_version
        );
    }
    if visitor.visit_primitive.is_none() {
        vortex_bail!("Vortex Velox visitor requires a primitive callback");
    }
    Ok(())
}

fn callback_error(visitor: &vx_velox_visitor, status: i32) -> String {
    let Some(last_error) = visitor.last_error else {
        return format!("Velox primitive visitor failed with status {status}");
    };
    // SAFETY: The callback contract returns null or a valid null-terminated string.
    let message = unsafe { last_error(visitor.context) };
    if message.is_null() {
        return format!("Velox primitive visitor failed with status {status}");
    }
    // SAFETY: The callback keeps the string valid until the next callback.
    unsafe { std::ffi::CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

fn selected_array(
    array: &vortex::array::ArrayRef,
    request: &vx_velox_visit_request,
) -> VortexResult<vortex::array::ArrayRef> {
    if request.rows.is_null() {
        if request.row_count != 0 {
            vortex_bail!("A null visitor row pointer requires a zero row count");
        }
        return Ok(array.clone());
    }
    // SAFETY: The caller supplies `row_count` readable positions.
    let rows = unsafe { slice::from_raw_parts(request.rows, request.row_count) };
    let mut previous = None;
    for row in rows {
        let position = usize::try_from(*row)
            .map_err(|_| vortex_err!("Visitor row does not fit usize: {}", row))?;
        if position >= array.len() {
            vortex_bail!(
                "Visitor row is out of bounds: row {}, array length {}",
                row,
                array.len()
            );
        }
        if previous.is_some_and(|previous| previous >= *row) {
            vortex_bail!("Visitor rows must be unique and increasing");
        }
        previous = Some(*row);
    }
    let dense = rows.len() == array.len()
        && rows
            .iter()
            .enumerate()
            .all(|(position, row)| *row == position as u64);
    if dense {
        return Ok(array.clone());
    }
    array.take(PrimitiveArray::from_iter(rows.iter().copied()).into_array())
}

fn visit_primitive(
    array: vortex::array::ArrayRef,
    session: &vortex::session::VortexSession,
    visitor: &vx_velox_visitor,
) -> VortexResult<()> {
    let mut execution = session.create_execution_ctx();
    let Canonical::Primitive(primitive) = array.execute::<Canonical>(&mut execution)? else {
        vortex_bail!("Primitive visitor received a non-primitive array");
    };
    let values = primitive.buffer_handle().clone();
    let host_values = values.try_to_host_sync()?;
    let mask = primitive
        .validity()?
        .execute_mask(primitive.len(), &mut execution)?;
    let (validity_kind, validity) = if !primitive.dtype().is_nullable() {
        (VX_VELOX_VALIDITY_NON_NULLABLE, None)
    } else {
        match mask {
            Mask::AllTrue(_) => (VX_VELOX_VALIDITY_ALL_VALID, None),
            Mask::AllFalse(_) => (VX_VELOX_VALIDITY_ALL_INVALID, None),
            Mask::Values(values) => (VX_VELOX_VALIDITY_BITMAP, Some(values.bit_buffer().clone())),
        }
    };
    let owner = Arc::new(PrimitiveOwner::try_new(
        &host_values,
        validity.as_ref(),
        primitive.len(),
    )?);
    let values_length = owner.values_length();
    let validity_length = owner.validity_length();
    let values = owner.values();
    let validity = owner.validity();
    let view = vx_velox_primitive_view {
        struct_size: size_of::<vx_velox_primitive_view>(),
        primitive_type: primitive_type_id(primitive.ptype()),
        length: primitive.len(),
        values,
        values_length,
        validity_kind,
        validity,
        validity_length,
        validity_bit_offset: 0,
        buffers: vx_velox_buffer_owner {
            struct_size: size_of::<vx_velox_buffer_owner>(),
            owner: Arc::as_ptr(&owner).cast(),
            retain: Some(retain_primitive_owner),
            release: Some(release_primitive_owner),
            retained_bytes: owner.retained_bytes(),
        },
        values_alignment: pointer_alignment(values),
        validity_alignment: pointer_alignment(validity),
    };
    let callback = visitor
        .visit_primitive
        .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a primitive callback"))?;
    // SAFETY: The view and its local owner stay live until the callback returns.
    let status = unsafe { callback(visitor.context, &raw const view) };
    if status != 0 {
        vortex_bail!("{}", callback_error(visitor, status));
    }
    Ok(())
}

/// Visit one Vortex array through host semantic callbacks.
///
/// The request selects source positions once. Callback block positions are compact and follow the
/// request order.
///
/// # Safety
///
/// Every pointer must be null or valid for the documented access. The array and session handles
/// must remain live until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_array_visit(
    session: *const vx_session,
    array: *const vx_array,
    request: *const vx_velox_visit_request,
    visitor: *const vx_velox_visitor,
    error_out: *mut *mut vx_error,
) -> i32 {
    try_or(error_out, 1, || {
        let session = unsafe { vx_session_ref(session)? };
        let array = unsafe { vx_array_ref(array)? };
        let request = unsafe {
            request
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox visit request must not be null"))?
        };
        if request.struct_size < size_of::<vx_velox_visit_request>() {
            vortex_bail!(
                "Vortex Velox visit request is too small: expected at least {}, got {}",
                size_of::<vx_velox_visit_request>(),
                request.struct_size
            );
        }
        let visitor = unsafe {
            visitor
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox visitor must not be null"))?
        };
        validate_visitor(visitor)?;
        visit_primitive(selected_array(array, request)?, session, visitor)?;
        Ok(0)
    })
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;
    use vortex_error::vortex_ensure;
    use vortex_ffi::vx_array_new_with;
    use vortex_ffi::vx_session_free;
    use vortex_ffi::vx_session_new_with;

    use super::*;
    use crate::api::vx_velox_array_free;

    #[rstest]
    #[case(PType::U8, VX_VELOX_PRIMITIVE_U8)]
    #[case(PType::U16, VX_VELOX_PRIMITIVE_U16)]
    #[case(PType::U32, VX_VELOX_PRIMITIVE_U32)]
    #[case(PType::U64, VX_VELOX_PRIMITIVE_U64)]
    #[case(PType::I8, VX_VELOX_PRIMITIVE_I8)]
    #[case(PType::I16, VX_VELOX_PRIMITIVE_I16)]
    #[case(PType::I32, VX_VELOX_PRIMITIVE_I32)]
    #[case(PType::I64, VX_VELOX_PRIMITIVE_I64)]
    #[case(PType::F16, VX_VELOX_PRIMITIVE_F16)]
    #[case(PType::F32, VX_VELOX_PRIMITIVE_F32)]
    #[case(PType::F64, VX_VELOX_PRIMITIVE_F64)]
    fn maps_primitive_types(#[case] input: PType, #[case] expected: vx_velox_primitive_type) {
        assert_eq!(primitive_type_id(input), expected);
    }

    #[derive(Default)]
    struct Capture {
        primitive_type: Option<vx_velox_primitive_type>,
        length: usize,
        values: *const u8,
        values_length: usize,
        values_alignment: usize,
        validity: *const u8,
        validity_length: usize,
        validity_bit_offset: usize,
        validity_alignment: usize,
        retained_bytes: usize,
        validity_kind: Option<vx_velox_validity_kind>,
        owner: Option<vx_velox_buffer_owner>,
    }

    unsafe extern "C" fn capture_primitive(
        context: *mut c_void,
        view: *const vx_velox_primitive_view,
    ) -> i32 {
        if context.is_null() || view.is_null() {
            return 1;
        }
        // SAFETY: The test passes pointers to live `Capture` and view objects.
        let (capture, view) = unsafe { (&mut *context.cast::<Capture>(), &*view) };
        let Some(retain) = view.buffers.retain else {
            return 2;
        };
        // SAFETY: The visitor owner is live for the callback.
        unsafe { retain(view.buffers.owner) };
        capture.primitive_type = Some(view.primitive_type);
        capture.length = view.length;
        capture.values = view.values;
        capture.values_length = view.values_length;
        capture.values_alignment = view.values_alignment;
        capture.validity = view.validity;
        capture.validity_length = view.validity_length;
        capture.validity_bit_offset = view.validity_bit_offset;
        capture.validity_alignment = view.validity_alignment;
        capture.retained_bytes = view.buffers.retained_bytes;
        capture.validity_kind = Some(view.validity_kind);
        capture.owner = Some(view.buffers);
        0
    }

    #[test]
    fn visits_sparse_nullable_values_with_retained_buffers() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let array = vx_array_new_with(
            PrimitiveArray::from_option_iter([Some(10_i64), None, Some(30), Some(40)]).into_array(),
        );
        let rows = [1_u64, 3];
        let request = vx_velox_visit_request {
            struct_size: size_of::<vx_velox_visit_request>(),
            rows: rows.as_ptr(),
            row_count: rows.len(),
        };
        let mut capture = Capture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: Some(capture_primitive),
            last_error: None,
        };
        let mut error = ptr::null_mut();
        // SAFETY: Every handle and callback object stays live for this call.
        let status = unsafe {
            vx_velox_array_visit(
                session,
                array,
                &raw const request,
                &raw const visitor,
                &raw mut error,
            )
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "visitor returned an error");
        assert_eq!(capture.primitive_type, Some(VX_VELOX_PRIMITIVE_I64));
        assert_eq!(capture.length, 2);
        assert_eq!(capture.values_length, 2 * size_of::<i64>());
        assert!(capture.values_alignment.is_power_of_two());
        assert_eq!(capture.values.addr() % capture.values_alignment, 0);
        assert_eq!(capture.validity_kind, Some(VX_VELOX_VALIDITY_BITMAP));
        assert_eq!(capture.validity_length, 1);
        assert_eq!(capture.validity_bit_offset, 0);
        assert!(capture.validity_alignment.is_power_of_two());
        assert_eq!(capture.validity.addr() % capture.validity_alignment, 0);
        assert_eq!(capture.retained_bytes, capture.values_length + 1);
        // SAFETY: The callback retained the owner before storing these pointers.
        let values = unsafe { slice::from_raw_parts(capture.values.cast::<i64>(), 2) };
        assert_eq!(values, [0, 40]);
        // SAFETY: The retained validity pointer has one readable byte.
        let validity = unsafe { *capture.validity };
        assert_eq!(validity & 0b11, 0b10);

        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
        // SAFETY: This release matches the retain in `capture_primitive`.
        unsafe { release(owner.owner) };
        // SAFETY: Each owned handle is freed exactly once.
        unsafe {
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        Ok(())
    }

    #[test]
    fn copies_sliced_values_and_reports_compact_allocation() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let source = PrimitiveArray::from_iter(0_i32..16);
        let source_values = source.buffer_handle().try_to_host_sync()?;
        // SAFETY: The source contains sixteen i32 values. The fifth value is in bounds.
        let source_slice = unsafe { source_values.as_ptr().add(5 * size_of::<i32>()) };
        let array = vx_array_new_with(source.into_array().slice(5..8)?);
        let request = vx_velox_visit_request {
            struct_size: size_of::<vx_velox_visit_request>(),
            rows: ptr::null(),
            row_count: 0,
        };
        let mut capture = Capture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: Some(capture_primitive),
            last_error: None,
        };
        let mut error = ptr::null_mut();
        let status = unsafe {
            vx_velox_array_visit(
                session,
                array,
                &raw const request,
                &raw const visitor,
                &raw mut error,
            )
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "visitor returned an error");
        assert_eq!(capture.values_length, 3 * size_of::<i32>());
        assert_eq!(capture.retained_bytes, 2 * size_of::<u64>());
        assert_ne!(capture.values, source_slice);
        // SAFETY: The retained compact values contain three i32 values.
        let values = unsafe { slice::from_raw_parts(capture.values.cast::<i32>(), 3) };
        assert_eq!(values, [5, 6, 7]);
        assert!(capture.values_alignment.is_power_of_two());
        assert_eq!(capture.values.addr() % capture.values_alignment, 0);
        assert_eq!(capture.validity_alignment, 0);

        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
        unsafe { release(owner.owner) };
        unsafe {
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        Ok(())
    }

    #[test]
    fn copies_validity_into_compact_storage() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let session_ref = unsafe { vx_session_ref(session)? };
        let primitive = PrimitiveArray::from_option_iter([Some(1_i32), None, Some(3)]);
        let mut execution = session_ref.create_execution_ctx();
        let Mask::Values(mask) = primitive
            .validity()?
            .execute_mask(primitive.len(), &mut execution)?
        else {
            vortex_bail!("Expected bitmap validity");
        };
        let expected_validity = mask.bit_buffer().inner().as_ptr();
        let array = vx_array_new_with(primitive.into_array());
        let request = vx_velox_visit_request {
            struct_size: size_of::<vx_velox_visit_request>(),
            rows: ptr::null(),
            row_count: 0,
        };
        let mut capture = Capture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: Some(capture_primitive),
            last_error: None,
        };
        let mut error = ptr::null_mut();
        let status = unsafe {
            vx_velox_array_visit(
                session,
                array,
                &raw const request,
                &raw const visitor,
                &raw mut error,
            )
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "visitor returned an error");
        assert_ne!(capture.validity, expected_validity);
        assert_eq!(capture.validity_bit_offset, 0);
        assert_eq!(capture.retained_bytes, 2 * size_of::<u64>() + 1);

        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
        unsafe { release(owner.owner) };
        unsafe {
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        Ok(())
    }

    #[test]
    fn rejects_unsorted_rows() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([1_i64, 2, 3]).into_array();
        let rows = [2_u64, 1];
        let request = vx_velox_visit_request {
            struct_size: size_of::<vx_velox_visit_request>(),
            rows: rows.as_ptr(),
            row_count: rows.len(),
        };
        match selected_array(&array, &request) {
            Ok(_) => vortex_bail!("unsorted rows unexpectedly succeeded"),
            Err(error) => assert!(error.to_string().contains("unique and increasing")),
        }
        Ok(())
    }
}
