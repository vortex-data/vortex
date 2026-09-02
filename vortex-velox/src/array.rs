// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_char;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;

use arrow_array::Array;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_buffer::BooleanBuffer;
use arrow_buffer::Buffer;
use arrow_buffer::NullBuffer;
use arrow_data::ArrayData;
use arrow_data::ArrayDataBuilder;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_ffi::try_or;
use vortex_ffi::vx_array;
use vortex_ffi::vx_array_new_with;
use vortex_ffi::vx_array_ref;
use vortex_ffi::vx_error;
use vortex_ffi::vx_session;
use vortex_ffi::vx_session_ref;

/// Host memory callbacks for one Arrow C Data export.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct vx_velox_arrow_memory_callbacks {
    /// Set this field to `sizeof(vx_velox_arrow_memory_callbacks)`.
    pub struct_size: usize,
    /// Set this field to [`crate::VX_VELOX_ABI_VERSION`].
    pub abi_version: u32,
    /// An opaque host context.
    pub context: *mut c_void,
    /// Retain the host context until the Arrow array release callback runs.
    pub retain_context: Option<unsafe extern "C" fn(context: *mut c_void)>,
    /// Release one host context reference.
    pub release_context: Option<unsafe extern "C" fn(context: *mut c_void)>,
    /// Reserve Arrow payload bytes before conversion. Zero means success.
    pub report_allocation:
        Option<unsafe extern "C" fn(context: *mut c_void, retained_bytes: usize) -> i32>,
    /// Free retained Arrow payload bytes.
    pub report_free: Option<unsafe extern "C" fn(context: *mut c_void, retained_bytes: usize)>,
    /// Return the last callback error as a null-terminated string.
    pub last_error: Option<unsafe extern "C" fn(context: *mut c_void) -> *const c_char>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ArrowMemoryCallbacksPrefix {
    struct_size: usize,
    abi_version: u32,
    context: *mut c_void,
    retain_context: Option<unsafe extern "C" fn(context: *mut c_void)>,
    release_context: Option<unsafe extern "C" fn(context: *mut c_void)>,
    report_allocation:
        Option<unsafe extern "C" fn(context: *mut c_void, retained_bytes: usize) -> i32>,
    report_free: Option<unsafe extern "C" fn(context: *mut c_void, retained_bytes: usize)>,
}

struct ArrowMemoryOwner {
    original_private_data: *mut c_void,
    original_release: unsafe extern "C" fn(array: *mut FFI_ArrowArray),
    callbacks: vx_velox_arrow_memory_callbacks,
    retained_bytes: usize,
}

struct ArrowMemoryReservation {
    callbacks: vx_velox_arrow_memory_callbacks,
    retained_bytes: usize,
    active: bool,
}

// SAFETY: The callback contract permits the retained context to move between threads.
unsafe impl Send for ArrowMemoryOwner {}
// SAFETY: Release accesses the immutable callback table once through exclusive ownership.
unsafe impl Sync for ArrowMemoryOwner {}

unsafe extern "C" fn release_accounted_arrow(array: *mut FFI_ArrowArray) {
    if array.is_null() {
        return;
    }
    // SAFETY: Arrow calls this function once with the exported top-level array.
    let array = unsafe { &mut *array };
    // SAFETY: `private_data` was created from this exact box in `attach_memory_owner`.
    let owner = unsafe { Box::from_raw(array.private_data.cast::<ArrowMemoryOwner>()) };
    array.private_data = owner.original_private_data;
    array.release = Some(owner.original_release);
    // SAFETY: The original release function owns the restored original private data.
    unsafe { (owner.original_release)(array) };
    if let Some(report_free) = owner.callbacks.report_free {
        // SAFETY: The retained callback context stays live until this release function ends.
        unsafe { report_free(owner.callbacks.context, owner.retained_bytes) };
    }
    if let Some(release_context) = owner.callbacks.release_context {
        // SAFETY: This release matches the retain before export.
        unsafe { release_context(owner.callbacks.context) };
    }
}

impl ArrowMemoryReservation {
    fn try_new(
        callbacks: vx_velox_arrow_memory_callbacks,
        retained_bytes: usize,
    ) -> VortexResult<Self> {
        let retain_context = callbacks
            .retain_context
            .ok_or_else(|| vortex_err!("Missing Arrow context retain callback"))?;
        let release_context = callbacks
            .release_context
            .ok_or_else(|| vortex_err!("Missing Arrow context release callback"))?;
        let report_allocation = callbacks
            .report_allocation
            .ok_or_else(|| vortex_err!("Missing Arrow allocation callback"))?;
        // SAFETY: The callback contract accepts one retained context reference.
        unsafe { retain_context(callbacks.context) };
        // SAFETY: The retained context stays live for this callback.
        let status = unsafe { report_allocation(callbacks.context, retained_bytes) };
        if status != 0 {
            let message = callback_error(&callbacks, status);
            // SAFETY: The reservation failed, so release the temporary context reference.
            unsafe { release_context(callbacks.context) };
            vortex_bail!("{}", message);
        }
        Ok(Self {
            callbacks,
            retained_bytes,
            active: true,
        })
    }

    fn reconcile(&mut self, actual_retained_bytes: usize) -> VortexResult<()> {
        match actual_retained_bytes.cmp(&self.retained_bytes) {
            std::cmp::Ordering::Less => {
                let released = self.retained_bytes - actual_retained_bytes;
                let report_free = self
                    .callbacks
                    .report_free
                    .ok_or_else(|| vortex_err!("Missing Arrow free callback"))?;
                // SAFETY: The retained callback context stays live for this callback.
                unsafe { report_free(self.callbacks.context, released) };
            }
            std::cmp::Ordering::Greater => {
                let additional = actual_retained_bytes - self.retained_bytes;
                let report_allocation = self
                    .callbacks
                    .report_allocation
                    .ok_or_else(|| vortex_err!("Missing Arrow allocation callback"))?;
                // SAFETY: The retained callback context stays live for this callback.
                let status = unsafe { report_allocation(self.callbacks.context, additional) };
                if status != 0 {
                    vortex_bail!("{}", callback_error(&self.callbacks, status));
                }
            }
            std::cmp::Ordering::Equal => {}
        }
        self.retained_bytes = actual_retained_bytes;
        Ok(())
    }
}

impl Drop for ArrowMemoryReservation {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Some(report_free) = self.callbacks.report_free {
            // SAFETY: The retained callback context stays live for this callback.
            unsafe { report_free(self.callbacks.context, self.retained_bytes) };
        }
        if let Some(release_context) = self.callbacks.release_context {
            // SAFETY: This release matches the reservation retain.
            unsafe { release_context(self.callbacks.context) };
        }
    }
}

unsafe fn parse_memory_callbacks(
    callbacks: *const vx_velox_arrow_memory_callbacks,
) -> VortexResult<vx_velox_arrow_memory_callbacks> {
    if callbacks.is_null() {
        vortex_bail!("Arrow memory callbacks must not be null");
    }
    // SAFETY: The caller guarantees that the pointer identifies at least `struct_size` bytes.
    let struct_size = unsafe { ptr::read(callbacks.cast::<usize>()) };
    if struct_size < size_of::<ArrowMemoryCallbacksPrefix>() {
        vortex_bail!(
            "Vortex Velox Arrow memory callback structure is too small: expected at least {}, got {}",
            size_of::<ArrowMemoryCallbacksPrefix>(),
            struct_size
        );
    }
    // SAFETY: The checked size covers the required callback prefix.
    let prefix = unsafe { ptr::read(callbacks.cast::<ArrowMemoryCallbacksPrefix>()) };
    if prefix.abi_version != crate::VX_VELOX_ABI_VERSION {
        vortex_bail!(
            "Unsupported Vortex Velox ABI version: expected {}, got {}",
            crate::VX_VELOX_ABI_VERSION,
            prefix.abi_version
        );
    }
    if prefix.retain_context.is_none()
        || prefix.release_context.is_none()
        || prefix.report_allocation.is_none()
        || prefix.report_free.is_none()
    {
        vortex_bail!("Vortex Velox Arrow memory callbacks are incomplete");
    }
    let last_error = if struct_size >= size_of::<vx_velox_arrow_memory_callbacks>() {
        // SAFETY: The checked size covers the optional tail field.
        unsafe { ptr::read(ptr::addr_of!((*callbacks).last_error)) }
    } else {
        None
    };
    Ok(vx_velox_arrow_memory_callbacks {
        struct_size,
        abi_version: prefix.abi_version,
        context: prefix.context,
        retain_context: prefix.retain_context,
        release_context: prefix.release_context,
        report_allocation: prefix.report_allocation,
        report_free: prefix.report_free,
        last_error,
    })
}

fn callback_error(callbacks: &vx_velox_arrow_memory_callbacks, status: i32) -> String {
    let Some(last_error) = callbacks.last_error else {
        return format!("Velox Arrow allocation callback failed with status {status}");
    };
    // SAFETY: The callback contract returns null or a valid null-terminated string.
    let message = unsafe { last_error(callbacks.context) };
    if message.is_null() {
        return format!("Velox Arrow allocation callback failed with status {status}");
    }
    // SAFETY: The callback keeps the string valid until the next callback.
    unsafe { std::ffi::CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

fn copy_nulls(nulls: &NullBuffer, data_offset: usize) -> VortexResult<NullBuffer> {
    let bit_length = data_offset
        .checked_add(nulls.len())
        .ok_or_else(|| vortex_err!("Arrow validity bit count overflow"))?;
    let mut bytes = vec![0_u8; bit_length.div_ceil(8)];
    for index in 0..nulls.len() {
        if nulls.is_valid(index) {
            let bit = data_offset + index;
            bytes[bit / 8] |= 1 << (bit % 8);
        }
    }
    Ok(NullBuffer::new(BooleanBuffer::new(
        Buffer::from(bytes),
        data_offset,
        nulls.len(),
    )))
}

fn copy_arrow_data(data: &ArrayData) -> VortexResult<ArrayData> {
    let buffers = data
        .buffers()
        .iter()
        .map(|buffer| Buffer::from_slice_ref(buffer.as_slice()))
        .collect();
    let children = data
        .child_data()
        .iter()
        .map(copy_arrow_data)
        .collect::<VortexResult<Vec<_>>>()?;
    let nulls = data
        .nulls()
        .map(|nulls| copy_nulls(nulls, data.offset()))
        .transpose()?;
    Ok(ArrayDataBuilder::new(data.data_type().clone())
        .len(data.len())
        .offset(data.offset())
        .buffers(buffers)
        .child_data(children)
        .nulls(nulls)
        .build()?)
}

fn variadic_ffi_buffer_bytes(data: &ArrayData) -> usize {
    let own_bytes = if arrow_data::layout(data.data_type()).variadic {
        let mut lengths = Vec::<i64>::new();
        #[expect(
            clippy::same_item_push,
            reason = "Match the Arrow FFI vector growth to account its exact retained capacity"
        )]
        for _ in data.buffers().iter().skip(1) {
            lengths.push(0);
        }
        lengths.capacity() * size_of::<i64>()
    } else {
        0
    };
    own_bytes
        + data
            .child_data()
            .iter()
            .map(variadic_ffi_buffer_bytes)
            .sum::<usize>()
}

fn attach_memory_owner(
    array: &mut FFI_ArrowArray,
    mut reservation: ArrowMemoryReservation,
) -> VortexResult<()> {
    let original_release = array
        .release
        .ok_or_else(|| vortex_err!("Exported Arrow array has no release callback"))?;
    let owner = Box::new(ArrowMemoryOwner {
        original_private_data: array.private_data,
        original_release,
        callbacks: reservation.callbacks,
        retained_bytes: reservation.retained_bytes,
    });
    reservation.active = false;
    array.private_data = Box::into_raw(owner).cast();
    array.release = Some(release_accounted_arrow);
    Ok(())
}

const ARROW_RESERVATION_OVERHEAD: usize = 64 * 1024;

fn conservative_arrow_reservation(
    array: &vortex::array::ArrayRef,
    execution: &mut vortex::array::ExecutionCtx,
) -> VortexResult<usize> {
    uncompressed_size_in_bytes(array, execution)?
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(ARROW_RESERVATION_OVERHEAD))
        .ok_or_else(|| vortex_err!("Arrow reservation size overflow"))
}

/// Return one struct field with the supplied session.
///
/// # Safety
///
/// The session and array pointers must identify live handles. `error_out` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_array_get_field(
    session: *const vx_session,
    array: *const vx_array,
    index: usize,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    try_or(error_out, ptr::null(), || {
        let session = unsafe { vx_session_ref(session)? };
        let array = unsafe { vx_array_ref(array)? };
        let mut execution = session.create_execution_ctx();
        let struct_array = array.clone().execute::<StructArray>(&mut execution)?;
        let field = struct_array
            .unmasked_field_opt(index)
            .ok_or_else(|| vortex_err!("Field index out of bounds: {index}"))?
            .clone();
        Ok(vx_array_new_with(field))
    })
}

/// Return the invalid value count with the supplied session.
///
/// # Safety
///
/// The session and array pointers must identify live handles. `error_out` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_array_invalid_count(
    session: *const vx_session,
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) -> usize {
    try_or(error_out, 0, || {
        let session = unsafe { vx_session_ref(session)? };
        let array = unsafe { vx_array_ref(array)? };
        array.invalid_count(&mut session.create_execution_ctx())
    })
}

/// Export one Vortex array through the Arrow C Data Interface.
///
/// The caller owns both outputs and must call their release callbacks. The memory callbacks reserve
/// a conservative payload charge before Arrow conversion. The adapter refunds the difference after
/// it knows the retained payload capacities. It requests a deficit before it returns the outputs.
/// The charge excludes schema and small FFI metadata.
///
/// # Safety
///
/// The session and array pointers must identify live handles. `memory_callbacks` must identify its
/// declared `struct_size` bytes for this call. Its callback context must remain valid through every
/// retained reference. Its callbacks and returned error strings must satisfy the header contract
/// and must not unwind. Both output pointers must identify uninitialized writable structures.
/// `error_out` must be null or identify writable storage for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_array_export_arrow(
    session: *const vx_session,
    array: *const vx_array,
    memory_callbacks: *const vx_velox_arrow_memory_callbacks,
    schema_out: *mut FFI_ArrowSchema,
    array_out: *mut FFI_ArrowArray,
    error_out: *mut *mut vx_error,
) -> i32 {
    try_or(error_out, 1, || {
        let session = unsafe { vx_session_ref(session)? };
        let array = unsafe { vx_array_ref(array)? };
        // SAFETY: The caller provides the callback table described by this function's contract.
        let memory_callbacks = unsafe { parse_memory_callbacks(memory_callbacks)? };
        if schema_out.is_null() {
            return Err(vortex_err!("Arrow schema output must not be null"));
        }
        if array_out.is_null() {
            return Err(vortex_err!("Arrow array output must not be null"));
        }

        let mut execution = session.create_execution_ctx();
        let reserved_bytes = conservative_arrow_reservation(array, &mut execution)?;
        let mut reservation = ArrowMemoryReservation::try_new(memory_callbacks, reserved_bytes)?;
        let mut arrow = session
            .arrow()
            .execute_arrow(array.clone(), None, &mut execution)?;
        if arrow.offset() != 0 {
            let length = u64::try_from(array.len())
                .map_err(|_| vortex_err!("Array length does not fit u64: {}", array.len()))?;
            let compact = array.take(PrimitiveArray::from_iter(0..length).into_array())?;
            arrow = session
                .arrow()
                .execute_arrow(compact, None, &mut execution)?;
        }
        let schema = FFI_ArrowSchema::try_from(arrow.data_type())?;
        let data = copy_arrow_data(&arrow.to_data())?;
        let retained_bytes = data
            .get_buffer_memory_size()
            .checked_add(variadic_ffi_buffer_bytes(&data))
            .ok_or_else(|| vortex_err!("Arrow retained memory size overflow"))?;
        let mut array = FFI_ArrowArray::new(&data);
        drop(data);
        drop(arrow);
        reservation.reconcile(retained_bytes)?;
        attach_memory_owner(&mut array, reservation)?;
        unsafe {
            ptr::write(schema_out, schema);
            ptr::write(array_out, array);
        }
        Ok(0)
    })
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use arrow_array::Int32Array;
    use arrow_array::StructArray as ArrowStructArray;
    use arrow_array::array::make_array;
    use arrow_array::ffi::from_ffi;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_ffi::vx_array_new_with;
    use vortex_ffi::vx_error_free;
    use vortex_ffi::vx_session_free;
    use vortex_ffi::vx_session_new_with;

    use super::*;
    use crate::api::vx_velox_array_free;

    #[derive(Default)]
    struct MemoryCapture {
        allocated: AtomicUsize,
        freed: AtomicUsize,
        first_allocation: AtomicUsize,
        allocation_calls: AtomicUsize,
        allocation_attempts: AtomicUsize,
        retained_contexts: AtomicUsize,
        released_contexts: AtomicUsize,
        reject_allocation: AtomicBool,
    }

    unsafe extern "C" fn retain_context(context: *mut c_void) {
        // SAFETY: The test context is an `Arc<MemoryCapture>` pointer.
        unsafe { Arc::increment_strong_count(context.cast::<MemoryCapture>()) };
        // SAFETY: The test keeps one independent `Arc` reference live.
        let capture = unsafe { &*context.cast::<MemoryCapture>() };
        capture.retained_contexts.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "C" fn release_context(context: *mut c_void) {
        // SAFETY: The test keeps one independent `Arc` reference live.
        let capture = unsafe { &*context.cast::<MemoryCapture>() };
        capture.released_contexts.fetch_add(1, Ordering::Relaxed);
        // SAFETY: This release matches one `retain_context` call.
        drop(unsafe { Arc::from_raw(context.cast::<MemoryCapture>()) });
    }

    unsafe extern "C" fn report_allocation(context: *mut c_void, bytes: usize) -> i32 {
        // SAFETY: The test context is a live `MemoryCapture`.
        let capture = unsafe { &*context.cast::<MemoryCapture>() };
        capture.allocation_attempts.fetch_add(1, Ordering::Relaxed);
        if capture.reject_allocation.load(Ordering::Relaxed) {
            return 7;
        }
        capture
            .first_allocation
            .compare_exchange(0, bytes, Ordering::Relaxed, Ordering::Relaxed)
            .ok();
        capture.allocation_calls.fetch_add(1, Ordering::Relaxed);
        capture.allocated.fetch_add(bytes, Ordering::Relaxed);
        0
    }

    unsafe extern "C" fn report_free(context: *mut c_void, bytes: usize) {
        // SAFETY: The test context is a live `MemoryCapture`.
        let capture = unsafe { &*context.cast::<MemoryCapture>() };
        capture.freed.fetch_add(bytes, Ordering::Relaxed);
    }

    unsafe extern "C" fn last_error(_context: *mut c_void) -> *const c_char {
        c"allocation rejected".as_ptr()
    }

    fn memory_callbacks(capture: &Arc<MemoryCapture>) -> vx_velox_arrow_memory_callbacks {
        vx_velox_arrow_memory_callbacks {
            struct_size: size_of::<vx_velox_arrow_memory_callbacks>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: Arc::as_ptr(capture).cast_mut().cast(),
            retain_context: Some(retain_context),
            release_context: Some(release_context),
            report_allocation: Some(report_allocation),
            report_free: Some(report_free),
            last_error: Some(last_error),
        }
    }

    #[test]
    fn accepts_required_memory_callback_prefix() -> VortexResult<()> {
        let capture = Arc::new(MemoryCapture::default());
        let callbacks = ArrowMemoryCallbacksPrefix {
            struct_size: size_of::<ArrowMemoryCallbacksPrefix>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: Arc::as_ptr(&capture).cast_mut().cast(),
            retain_context: Some(retain_context),
            release_context: Some(release_context),
            report_allocation: Some(report_allocation),
            report_free: Some(report_free),
        };
        // SAFETY: The test prefix reports its exact initialized size.
        let normalized = unsafe {
            parse_memory_callbacks(
                (&raw const callbacks).cast::<vx_velox_arrow_memory_callbacks>(),
            )?
        };
        assert!(normalized.last_error.is_none());
        Ok(())
    }

    #[test]
    fn exports_one_array_to_arrow() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let array = vx_array_new_with(PrimitiveArray::from_iter([1_i32, 2, 3]).into_array());
        let mut schema = FFI_ArrowSchema::empty();
        let mut arrow_array = FFI_ArrowArray::empty();
        let mut error = ptr::null_mut();
        let capture = Arc::new(MemoryCapture::default());
        let callbacks = memory_callbacks(&capture);

        let status = unsafe {
            vx_velox_array_export_arrow(
                session,
                array,
                &raw const callbacks,
                &raw mut schema,
                &raw mut arrow_array,
                &raw mut error,
            )
        };
        if !error.is_null() {
            unsafe { vx_error_free(error) };
        }
        assert_eq!(status, 0);
        assert!(error.is_null());

        let data = unsafe { from_ffi(arrow_array, &schema)? };
        let arrow = make_array(data);
        let values = arrow
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| vortex_err!("Expected an Arrow Int32 array"))?;
        assert_eq!(values.values(), &[1, 2, 3]);
        assert!(capture.allocated.load(Ordering::Relaxed) > 0);
        assert!(capture.first_allocation.load(Ordering::Relaxed) >= ARROW_RESERVATION_OVERHEAD);
        assert_eq!(capture.allocation_calls.load(Ordering::Relaxed), 1);
        assert!(capture.freed.load(Ordering::Relaxed) < capture.allocated.load(Ordering::Relaxed));
        drop(arrow);
        assert_eq!(
            capture.freed.load(Ordering::Relaxed),
            capture.allocated.load(Ordering::Relaxed)
        );
        assert_eq!(capture.retained_contexts.load(Ordering::Relaxed), 1);
        assert_eq!(capture.released_contexts.load(Ordering::Relaxed), 1);

        unsafe {
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        Ok(())
    }

    #[test]
    fn exports_sliced_array_with_zero_arrow_offset() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let source = PrimitiveArray::from_iter([1_i32, 2, 3, 4]).into_array();
        let array = vx_array_new_with(source.slice(1..3)?);
        let mut error = ptr::null_mut();

        let mut schema = FFI_ArrowSchema::empty();
        let mut arrow_array = FFI_ArrowArray::empty();
        let capture = Arc::new(MemoryCapture::default());
        let callbacks = memory_callbacks(&capture);
        let status = unsafe {
            vx_velox_array_export_arrow(
                session,
                array,
                &raw const callbacks,
                &raw mut schema,
                &raw mut arrow_array,
                &raw mut error,
            )
        };
        if !error.is_null() {
            unsafe { vx_error_free(error) };
        }
        assert_eq!(status, 0);
        assert!(error.is_null());

        let data = unsafe { from_ffi(arrow_array, &schema)? };
        assert_eq!(data.offset(), 0);
        let arrow = make_array(data);
        let values = arrow
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| vortex_err!("Expected an Arrow Int32 array"))?;
        assert_eq!(values.values(), &[2, 3]);

        unsafe {
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        Ok(())
    }

    #[test]
    fn reserves_and_reconciles_nested_arrow_fallback() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let inner = StructArray::try_new(
            ["value"].into(),
            vec![PrimitiveArray::from_option_iter([Some(1_i32), None, Some(3)]).into_array()],
            3,
            Validity::NonNullable,
        )?
        .into_array();
        let outer = StructArray::try_new(["nested"].into(), vec![inner], 3, Validity::NonNullable)?
            .into_array();
        let array = vx_array_new_with(outer);
        let capture = Arc::new(MemoryCapture::default());
        let callbacks = memory_callbacks(&capture);
        let mut schema = FFI_ArrowSchema::empty();
        let mut arrow_array = FFI_ArrowArray::empty();
        let mut error = ptr::null_mut();

        let status = unsafe {
            vx_velox_array_export_arrow(
                session,
                array,
                &raw const callbacks,
                &raw mut schema,
                &raw mut arrow_array,
                &raw mut error,
            )
        };
        assert_eq!(status, 0);
        assert!(error.is_null());
        assert_eq!(capture.allocation_calls.load(Ordering::Relaxed), 1);
        assert!(capture.first_allocation.load(Ordering::Relaxed) >= ARROW_RESERVATION_OVERHEAD);

        let data = unsafe { from_ffi(arrow_array, &schema)? };
        let arrow = make_array(data);
        let outer = arrow
            .as_any()
            .downcast_ref::<ArrowStructArray>()
            .ok_or_else(|| vortex_err!("Expected an outer Arrow struct"))?;
        let inner = outer
            .column(0)
            .as_any()
            .downcast_ref::<ArrowStructArray>()
            .ok_or_else(|| vortex_err!("Expected a nested Arrow struct"))?;
        let values = inner
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| vortex_err!("Expected nested Arrow i32 values"))?;
        assert_eq!(values.value(0), 1);
        assert!(values.is_null(1));
        assert_eq!(values.value(2), 3);
        drop(arrow);
        assert_eq!(
            capture.freed.load(Ordering::Relaxed),
            capture.allocated.load(Ordering::Relaxed)
        );

        unsafe {
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        Ok(())
    }

    #[test]
    fn copies_external_arrow_buffers_before_accounting() -> VortexResult<()> {
        struct ExternalBytes {
            bytes: Box<[u8]>,
            drops: Arc<AtomicUsize>,
        }

        impl AsRef<[u8]> for ExternalBytes {
            fn as_ref(&self) -> &[u8] {
                &self.bytes
            }
        }

        impl Drop for ExternalBytes {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::Relaxed);
            }
        }

        let values = [1_i32, 2, 3];
        // SAFETY: `values` is live and readable for its byte size.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(&values))
        };
        let drops = Arc::new(AtomicUsize::new(0));
        let external = Buffer::from(bytes::Bytes::from_owner(ExternalBytes {
            bytes: bytes.into(),
            drops: Arc::clone(&drops),
        }));
        let data_type = Int32Array::from(values.to_vec()).data_type().clone();
        let source = ArrayDataBuilder::new(data_type)
            .len(values.len())
            .add_buffer(external)
            .build()?;

        let copied = copy_arrow_data(&source)?;
        assert!(copied.buffers()[0].capacity() >= size_of_val(&values));
        assert_ne!(copied.buffers()[0].as_ptr(), source.buffers()[0].as_ptr());
        drop(source);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(copied.buffers()[0].as_slice(), bytes);
        Ok(())
    }

    #[test]
    fn rejects_arrow_allocation_and_releases_context() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let array = vx_array_new_with(PrimitiveArray::from_iter([1_i32, 2]).into_array());
        let capture = Arc::new(MemoryCapture::default());
        capture.reject_allocation.store(true, Ordering::Relaxed);
        let callbacks = memory_callbacks(&capture);
        let mut schema = FFI_ArrowSchema::empty();
        let mut arrow_array = FFI_ArrowArray::empty();
        let mut error = ptr::null_mut();

        let status = unsafe {
            vx_velox_array_export_arrow(
                session,
                array,
                &raw const callbacks,
                &raw mut schema,
                &raw mut arrow_array,
                &raw mut error,
            )
        };
        assert_eq!(status, 1);
        assert!(!error.is_null());
        assert!(arrow_array.release.is_none());
        assert_eq!(capture.allocated.load(Ordering::Relaxed), 0);
        assert_eq!(capture.freed.load(Ordering::Relaxed), 0);
        assert_eq!(capture.allocation_calls.load(Ordering::Relaxed), 0);
        assert_eq!(capture.allocation_attempts.load(Ordering::Relaxed), 1);
        assert!(capture.first_allocation.load(Ordering::Relaxed) == 0);
        assert_eq!(capture.retained_contexts.load(Ordering::Relaxed), 1);
        assert_eq!(capture.released_contexts.load(Ordering::Relaxed), 1);

        unsafe {
            vx_error_free(error);
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        Ok(())
    }

    #[test]
    fn accesses_fields_and_invalid_counts_with_session() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let values = PrimitiveArray::from_option_iter([Some(1_i32), None, Some(3)]).into_array();
        let structure =
            StructArray::try_new(["value"].into(), vec![values], 3, Validity::NonNullable)?
                .into_array();
        let array = vx_array_new_with(structure);
        let mut error = ptr::null_mut();

        let field = unsafe { vx_velox_array_get_field(session, array, 0, &raw mut error) };
        assert!(!field.is_null());
        assert!(error.is_null());
        assert_eq!(unsafe { vx_array_ref(field)? }.len(), 3);
        assert_eq!(
            unsafe { vx_velox_array_invalid_count(session, field, &raw mut error) },
            1
        );
        assert!(error.is_null());

        let missing = unsafe { vx_velox_array_get_field(session, array, 1, &raw mut error) };
        assert!(missing.is_null());
        assert!(!error.is_null());
        unsafe { vx_error_free(error) };
        error = ptr::null_mut();
        assert_eq!(
            unsafe { vx_velox_array_invalid_count(ptr::null(), field, &raw mut error) },
            0
        );
        assert!(!error.is_null());

        unsafe {
            vx_error_free(error);
            vx_velox_array_free(field);
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        Ok(())
    }
}
