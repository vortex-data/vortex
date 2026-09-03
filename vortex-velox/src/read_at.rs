// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_char;
use std::ffi::c_void;
use std::mem::size_of;
use std::slice;
use std::sync::Arc;

use bytes::Bytes;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::ReadAtRequest;
use vortex_io::ReadAtStream;
use vortex_io::VortexReadAt;
use vortex_io::runtime::BlockingRuntime;

use crate::ffi::ffi_runtime;
use crate::ffi::try_or;
use crate::ffi::vx_velox_error;

/// A positional read request passed to the Velox callback.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct vx_velox_read_request {
    /// Set this field to `sizeof(vx_velox_read_request)`.
    pub struct_size: usize,
    /// The file offset in bytes.
    pub offset: u64,
    /// The exact requested length in bytes.
    pub length: usize,
    /// The required buffer alignment in bytes.
    pub alignment: usize,
}

/// A retained host buffer returned by the Velox callback.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_buffer {
    /// Set this field to `sizeof(vx_velox_buffer)`.
    pub struct_size: usize,
    /// The first byte of the returned range.
    pub data: *const u8,
    /// The number of returned bytes.
    pub length: usize,
    /// An opaque owner passed to `release`.
    pub owner: *mut c_void,
    /// Release the owner after Vortex no longer needs the bytes.
    pub release: Option<unsafe extern "C" fn(owner: *mut c_void)>,
}

impl Default for vx_velox_buffer {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>(),
            data: std::ptr::null(),
            length: 0,
            owner: std::ptr::null_mut(),
            release: None,
        }
    }
}

/// Velox callbacks that provide a Vortex positional reader.
///
/// Vortex can call these functions concurrently. The context and every callback must be
/// thread-safe. `concurrency` limits one callback batch and gives Vortex a scheduling hint. It does
/// not provide synchronization. `last_error` must return the calling thread's most recent callback
/// error. Its string must remain valid until the next callback on that thread. Every callback must
/// catch foreign exceptions and must not unwind across this ABI.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct vx_velox_read_at_callbacks {
    /// Set this field to `sizeof(vx_velox_read_at_callbacks)`.
    pub struct_size: usize,
    /// Set this field to [`crate::VX_VELOX_ABI_VERSION`].
    pub abi_version: u32,
    /// An opaque callback context.
    pub context: *mut c_void,
    /// Return the file size through `size_out`. Zero means success.
    pub size: Option<unsafe extern "C" fn(context: *mut c_void, size_out: *mut u64) -> i32>,
    /// Read every request and populate the matching output. Zero means success.
    pub read_ranges: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            requests: *const vx_velox_read_request,
            request_count: usize,
            outputs: *mut vx_velox_buffer,
        ) -> i32,
    >,
    /// Return the last callback error as a null-terminated string.
    pub last_error: Option<unsafe extern "C" fn(context: *mut c_void) -> *const c_char>,
    /// Release the callback context.
    pub release_context: Option<unsafe extern "C" fn(context: *mut c_void)>,
    /// Return a non-zero value after the host cancels the scan.
    pub is_cancelled: Option<unsafe extern "C" fn(context: *mut c_void) -> i32>,
    /// Limit one callback batch and give Vortex a preferred concurrency value.
    pub concurrency: usize,
}

struct CallbackState {
    callbacks: vx_velox_read_at_callbacks,
}

// SAFETY: The public ABI requires the callback context and functions to permit concurrent calls.
unsafe impl Send for CallbackState {}
// SAFETY: The public ABI requires the callback context and functions to permit concurrent calls.
unsafe impl Sync for CallbackState {}

impl Drop for CallbackState {
    fn drop(&mut self) {
        if let Some(release_context) = self.callbacks.release_context {
            // SAFETY: The callback contract keeps `context` valid until this call.
            unsafe { release_context(self.callbacks.context) };
        }
    }
}

#[derive(Clone)]
pub(crate) struct CallbackReadAt {
    state: Arc<CallbackState>,
}

impl CallbackReadAt {
    fn try_new(callbacks: vx_velox_read_at_callbacks) -> VortexResult<Self> {
        if callbacks.struct_size < size_of::<vx_velox_read_at_callbacks>() {
            vortex_bail!(
                "Velox read callback structure is too small: expected at least {}, got {}",
                size_of::<vx_velox_read_at_callbacks>(),
                callbacks.struct_size
            );
        }
        if callbacks.abi_version != crate::VX_VELOX_ABI_VERSION {
            vortex_bail!(
                "Unsupported Vortex Velox ABI version: expected {}, got {}",
                crate::VX_VELOX_ABI_VERSION,
                callbacks.abi_version
            );
        }
        if callbacks.size.is_none() {
            vortex_bail!("Velox read callbacks require a size function");
        }
        if callbacks.read_ranges.is_none() {
            vortex_bail!("Velox read callbacks require a read_ranges function");
        }
        if callbacks.is_cancelled.is_none() {
            vortex_bail!("Velox read callbacks require an is_cancelled function");
        }

        Ok(Self {
            state: Arc::new(CallbackState { callbacks }),
        })
    }

    fn ensure_not_cancelled(&self) -> VortexResult<()> {
        let is_cancelled = self
            .state
            .callbacks
            .is_cancelled
            .vortex_expect("is_cancelled is validated when the reader is created");
        // SAFETY: The callback context stays live while the reader owns its callback state.
        if unsafe { is_cancelled(self.state.callbacks.context) } != 0 {
            vortex_bail!("Velox cancelled the Vortex read");
        }
        Ok(())
    }

    fn last_error(&self, operation: &str, status: i32) -> String {
        let Some(last_error) = self.state.callbacks.last_error else {
            return format!("Velox {operation} callback failed with status {status}");
        };

        // SAFETY: The callback contract returns null or a valid null-terminated string.
        let message = unsafe { last_error(self.state.callbacks.context) };
        if message.is_null() {
            return format!("Velox {operation} callback failed with status {status}");
        }

        // SAFETY: The callback contract keeps the string valid until the next callback call.
        unsafe { std::ffi::CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    }

    fn read_batch(&self, requests: Arc<[ReadAtRequest]>) -> Vec<VortexResult<BufferHandle>> {
        if requests.is_empty() {
            return Vec::new();
        }
        if let Err(error) = self.ensure_not_cancelled() {
            let message = error.to_string();
            return requests
                .iter()
                .map(|_| Err(vortex_err!("{}", message)))
                .collect();
        }

        let raw_requests = requests
            .iter()
            .map(|request| vx_velox_read_request {
                struct_size: size_of::<vx_velox_read_request>(),
                offset: request.offset,
                length: request.length,
                alignment: usize::from(request.alignment),
            })
            .collect::<Vec<_>>();
        let mut outputs = vec![vx_velox_buffer::default(); requests.len()];
        let read_ranges = self
            .state
            .callbacks
            .read_ranges
            .vortex_expect("read_ranges is validated when the reader is created");

        // SAFETY: The slices remain valid for the duration of the callback.
        let status = unsafe {
            read_ranges(
                self.state.callbacks.context,
                raw_requests.as_ptr(),
                raw_requests.len(),
                outputs.as_mut_ptr(),
            )
        };
        if status != 0 {
            let message = self.last_error("read_ranges", status);
            release_outputs(&mut outputs);
            return requests
                .iter()
                .map(|_| Err(vortex_err!("{}", message)))
                .collect();
        }

        outputs
            .into_iter()
            .zip(requests.iter())
            .map(|(output, request)| output.into_handle(request))
            .collect()
    }
}

impl VortexReadAt for CallbackReadAt {
    fn concurrency(&self) -> usize {
        self.state.callbacks.concurrency.max(1)
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let reader = self.clone();
        async move {
            reader.ensure_not_cancelled()?;
            let mut size = 0;
            let size_callback = reader
                .state
                .callbacks
                .size
                .vortex_expect("size is validated when the reader is created");
            // SAFETY: `size` remains valid for the duration of the callback.
            let status = unsafe { size_callback(reader.state.callbacks.context, &raw mut size) };
            if status != 0 {
                vortex_bail!("{}", reader.last_error("size", status));
            }
            Ok(size)
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let reader = self.clone();
        async move {
            let request = ReadAtRequest::new(offset, length, alignment);
            reader
                .read_batch(Arc::from([request]))
                .pop()
                .ok_or_else(|| vortex_err!("Velox read callback returned no result"))?
        }
        .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        let pairs = requests
            .chunks(self.concurrency())
            .flat_map(|requests| {
                let requests: Arc<[ReadAtRequest]> = Arc::from(requests);
                let results = self.read_batch(Arc::clone(&requests));
                requests.iter().copied().zip(results).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        stream::iter(pairs).boxed()
    }
}

impl vx_velox_read_at {
    pub(crate) fn reader(&self) -> CallbackReadAt {
        self.0.clone()
    }
}

struct ForeignBuffer {
    data: *const u8,
    length: usize,
    owner: *mut c_void,
    release: unsafe extern "C" fn(owner: *mut c_void),
}

// SAFETY: The buffer contract keeps immutable bytes valid until `release` runs.
unsafe impl Send for ForeignBuffer {}
// SAFETY: The buffer contract permits immutable byte access from concurrent threads.
unsafe impl Sync for ForeignBuffer {}

impl AsRef<[u8]> for ForeignBuffer {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: The buffer contract guarantees a valid immutable range for this lifetime.
        unsafe { slice::from_raw_parts(self.data, self.length) }
    }
}

impl Drop for ForeignBuffer {
    fn drop(&mut self) {
        // SAFETY: The owner is released exactly once when the final `Bytes` reference drops.
        unsafe { (self.release)(self.owner) };
    }
}

impl vx_velox_buffer {
    fn into_handle(self, request: &ReadAtRequest) -> VortexResult<BufferHandle> {
        if self.struct_size < size_of::<vx_velox_buffer>() {
            self.release_if_present();
            vortex_bail!(
                "Velox read callback returned a buffer structure that is too small: expected at least {}, got {}",
                size_of::<vx_velox_buffer>(),
                self.struct_size
            );
        }
        if self.length != request.length {
            self.release_if_present();
            vortex_bail!(
                "Velox read callback returned {} bytes for a {} byte request at offset {}",
                self.length,
                request.length,
                request.offset
            );
        }
        if self.length == 0 {
            self.release_if_present();
            return Ok(BufferHandle::new_host(ByteBuffer::empty_aligned(
                request.alignment,
            )));
        }
        if self.data.is_null() {
            self.release_if_present();
            vortex_bail!(
                "Velox read callback returned a null buffer for {} bytes at offset {}",
                request.length,
                request.offset
            );
        }
        let Some(release) = self.release else {
            vortex_bail!(
                "Velox read callback returned bytes without an owner release callback at offset {}",
                request.offset
            );
        };

        let owner = ForeignBuffer {
            data: self.data,
            length: self.length,
            owner: self.owner,
            release,
        };
        let bytes = Bytes::from_owner(owner);
        Ok(BufferHandle::new_host(
            ByteBuffer::from(bytes).aligned(request.alignment),
        ))
    }

    fn release_if_present(self) {
        if let Some(release) = self.release {
            // SAFETY: Failed validation still transfers the returned owner to this adapter.
            unsafe { release(self.owner) };
        }
    }
}

fn release_outputs(outputs: &mut [vx_velox_buffer]) {
    for output in outputs {
        let owned = std::mem::take(output);
        owned.release_if_present();
    }
}

/// An opaque Vortex positional reader backed by Velox callbacks.
pub struct vx_velox_read_at(CallbackReadAt);

/// Create a Vortex positional reader from Velox callbacks.
///
/// # Safety
///
/// `callbacks` must point to a valid callback structure. Every callback and its context must be
/// thread-safe and must not unwind. `error_out` must be null or valid for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_read_at_new(
    callbacks: *const vx_velox_read_at_callbacks,
    error_out: *mut *mut vx_velox_error,
) -> *mut vx_velox_read_at {
    try_or(error_out, std::ptr::null_mut(), || {
        let callbacks = unsafe {
            callbacks
                .as_ref()
                .ok_or_else(|| vortex_err!("Velox read callbacks must not be null"))?
        };
        let reader = CallbackReadAt::try_new(*callbacks)?;
        Ok(Box::into_raw(Box::new(vx_velox_read_at(reader))))
    })
}

/// Free a Vortex positional reader.
///
/// # Safety
///
/// `reader` must be null or a pointer returned by [`vx_velox_read_at_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_read_at_free(reader: *mut vx_velox_read_at) {
    if !reader.is_null() {
        // SAFETY: The caller transfers the unique pointer returned by the constructor.
        drop(unsafe { Box::from_raw(reader) });
    }
}

/// Return the size of a callback-backed source.
///
/// This entry point validates the host callback contract before file-reader code consumes the
/// source.
///
/// # Safety
///
/// `reader` must point to a live reader. `error_out` must be null or valid for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_read_at_size(
    reader: *const vx_velox_read_at,
    error_out: *mut *mut vx_velox_error,
) -> u64 {
    try_or(error_out, 0, || {
        let reader = unsafe {
            reader
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox reader must not be null"))?
        };
        ffi_runtime().block_on(reader.0.size())
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::ffi::CString;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use arrow_array::ffi::FFI_ArrowSchema;
    use arrow_schema::Schema;
    use futures::executor::block_on;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::validity::Validity;
    use vortex::expr::and;
    use vortex::expr::col;
    use vortex::expr::gt;
    use vortex::expr::lit;
    use vortex::expr::lt;
    use vortex::file::WriteOptionsSessionExt;
    use vortex_buffer::Alignment;
    use vortex_error::vortex_ensure;

    use super::*;
    use crate::api::vx_velox_array_free;
    use crate::api::vx_velox_data_source_free;
    use crate::api::vx_velox_data_source_scan;
    use crate::api::vx_velox_expression_free;
    use crate::api::vx_velox_partition_free;
    use crate::api::vx_velox_partition_next;
    use crate::api::vx_velox_scan_free;
    use crate::api::vx_velox_scan_next_partition;
    use crate::api::vx_velox_scan_options;
    use crate::api::vx_velox_scan_selection;
    use crate::ffi::vx_array_ref;
    use crate::ffi::vx_expression_new_with;
    use crate::ffi::vx_session_free;
    use crate::ffi::vx_session_new_with;
    use crate::ffi::vx_session_ref;
    use crate::schema::vx_velox_source_export_schema;
    use crate::source::vx_velox_natural_split;
    use crate::source::vx_velox_source_data_source;
    use crate::source::vx_velox_source_file_size;
    use crate::source::vx_velox_source_free;
    use crate::source::vx_velox_source_natural_split_at;
    use crate::source::vx_velox_source_natural_split_count;
    use crate::source::vx_velox_source_new;
    use crate::source::vx_velox_source_prune_natural_splits;
    use crate::source::vx_velox_source_row_count;

    struct TestContext {
        bytes: Arc<[u8]>,
        error: CString,
        calls: AtomicUsize,
        cancelled: AtomicBool,
        fail_after_first_output: AtomicBool,
        releases: Arc<AtomicUsize>,
        context_releases: Arc<AtomicUsize>,
    }

    struct TestOwner {
        bytes: Arc<[u8]>,
        releases: Arc<AtomicUsize>,
    }

    struct ConcurrentErrorContext {
        barrier: Barrier,
        context_releases: Arc<AtomicUsize>,
    }

    thread_local! {
        static CONCURRENT_ERROR: Cell<*const c_char> = const { Cell::new(std::ptr::null()) };
    }

    unsafe extern "C" fn concurrent_size(_context: *mut c_void, size_out: *mut u64) -> i32 {
        // SAFETY: The callback contract supplies a valid output pointer.
        unsafe { size_out.write(2) };
        0
    }

    unsafe extern "C" fn concurrent_read_ranges(
        context: *mut c_void,
        requests: *const vx_velox_read_request,
        request_count: usize,
        _outputs: *mut vx_velox_buffer,
    ) -> i32 {
        // SAFETY: The test passes a live context and one readable request.
        let context = unsafe { &*context.cast::<ConcurrentErrorContext>() };
        // SAFETY: The callback contract supplies `request_count` readable requests.
        let requests = unsafe { slice::from_raw_parts(requests, request_count) };
        let message = match requests.first().map(|request| request.offset) {
            Some(0) => c"read zero failed".as_ptr(),
            Some(1) => c"read one failed".as_ptr(),
            _ => c"unexpected read failed".as_ptr(),
        };
        CONCURRENT_ERROR.with(|error| error.set(message));
        context.barrier.wait();
        1
    }

    unsafe extern "C" fn concurrent_last_error(_context: *mut c_void) -> *const c_char {
        CONCURRENT_ERROR.with(|error| error.get())
    }

    unsafe extern "C" fn concurrent_release_context(context: *mut c_void) {
        // SAFETY: The test created this context with `Box::into_raw`.
        let context = unsafe { Box::from_raw(context.cast::<ConcurrentErrorContext>()) };
        context.context_releases.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "C" fn never_cancelled(_context: *mut c_void) -> i32 {
        0
    }

    unsafe extern "C" fn test_size(context: *mut c_void, size_out: *mut u64) -> i32 {
        // SAFETY: Tests pass a `TestContext` and a valid output pointer.
        let context = unsafe { &*context.cast::<TestContext>() };
        // SAFETY: The callback contract supplies a valid output pointer.
        unsafe { size_out.write(context.bytes.len() as u64) };
        0
    }

    unsafe extern "C" fn test_read_ranges(
        context: *mut c_void,
        requests: *const vx_velox_read_request,
        request_count: usize,
        outputs: *mut vx_velox_buffer,
    ) -> i32 {
        // SAFETY: Tests pass valid callback arguments.
        let context = unsafe { &*context.cast::<TestContext>() };
        context.calls.fetch_add(1, Ordering::Relaxed);
        // SAFETY: The callback contract supplies arrays with `request_count` entries.
        let requests = unsafe { slice::from_raw_parts(requests, request_count) };
        // SAFETY: The callback contract supplies writable outputs with matching length.
        let outputs = unsafe { slice::from_raw_parts_mut(outputs, request_count) };
        for (request, output) in requests.iter().zip(outputs) {
            let Ok(start) = usize::try_from(request.offset) else {
                return 1;
            };
            let Some(end) = start.checked_add(request.length) else {
                return 1;
            };
            if end > context.bytes.len() {
                return 1;
            }
            let owner = Box::new(TestOwner {
                bytes: Arc::clone(&context.bytes),
                releases: Arc::clone(&context.releases),
            });
            output.data = owner.bytes[start..end].as_ptr();
            output.length = request.length;
            output.owner = Box::into_raw(owner).cast();
            output.release = Some(test_release_buffer);
            if context.fail_after_first_output.load(Ordering::Relaxed) {
                return 1;
            }
        }
        0
    }

    unsafe extern "C" fn test_release_buffer(owner: *mut c_void) {
        // SAFETY: The test callback created this owner with `Box::into_raw`.
        let owner = unsafe { Box::from_raw(owner.cast::<TestOwner>()) };
        owner.releases.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "C" fn test_last_error(context: *mut c_void) -> *const c_char {
        // SAFETY: Tests pass a `TestContext`.
        let context = unsafe { &*context.cast::<TestContext>() };
        context.error.as_ptr()
    }

    unsafe extern "C" fn test_release_context(context: *mut c_void) {
        // SAFETY: The test created this context with `Box::into_raw`.
        let context = unsafe { Box::from_raw(context.cast::<TestContext>()) };
        context.context_releases.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "C" fn test_is_cancelled(context: *mut c_void) -> i32 {
        // SAFETY: Tests pass a `TestContext`.
        let context = unsafe { &*context.cast::<TestContext>() };
        i32::from(context.cancelled.load(Ordering::Relaxed))
    }

    fn callbacks(
        bytes: &[u8],
        releases: Arc<AtomicUsize>,
        context_releases: Arc<AtomicUsize>,
    ) -> vx_velox_read_at_callbacks {
        let context = Box::new(TestContext {
            bytes: Arc::from(bytes),
            error: c"test read failed".to_owned(),
            calls: AtomicUsize::new(0),
            cancelled: AtomicBool::new(false),
            fail_after_first_output: AtomicBool::new(false),
            releases,
            context_releases,
        });
        vx_velox_read_at_callbacks {
            struct_size: size_of::<vx_velox_read_at_callbacks>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: Box::into_raw(context).cast(),
            size: Some(test_size),
            read_ranges: Some(test_read_ranges),
            last_error: Some(test_last_error),
            release_context: Some(test_release_context),
            is_cancelled: Some(test_is_cancelled),
            concurrency: 8,
        }
    }

    #[test]
    fn batches_reads_and_releases_owners() -> VortexResult<()> {
        let releases = Arc::new(AtomicUsize::new(0));
        let context_releases = Arc::new(AtomicUsize::new(0));
        let read_at = CallbackReadAt::try_new(callbacks(
            b"abcdefgh",
            Arc::clone(&releases),
            Arc::clone(&context_releases),
        ))?;

        assert_eq!(block_on(read_at.size())?, 8);
        let requests: Arc<[ReadAtRequest]> = Arc::from([
            ReadAtRequest::new(1, 3, Alignment::none()),
            ReadAtRequest::new(5, 2, Alignment::none()),
        ]);
        let results = block_on(read_at.read_ranges(requests).collect::<Vec<_>>());
        assert_eq!(results.len(), 2);
        let mut results = results.into_iter();
        let (_, first) = results
            .next()
            .vortex_expect("the first read result is present");
        let (_, second) = results
            .next()
            .vortex_expect("the second read result is present");
        let first = first?;
        let second = second?;
        assert_eq!(first.to_host_sync().as_ref(), b"bcd");
        assert_eq!(second.to_host_sync().as_ref(), b"fg");
        assert_eq!(read_at.state.callbacks.concurrency, 8);
        assert_eq!(releases.load(Ordering::Relaxed), 0);

        drop((first, second));
        assert_eq!(releases.load(Ordering::Relaxed), 2);
        drop(read_at);
        assert_eq!(context_releases.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn respects_callback_batch_limit() -> VortexResult<()> {
        let releases = Arc::new(AtomicUsize::new(0));
        let context_releases = Arc::new(AtomicUsize::new(0));
        let mut callbacks = callbacks(
            b"abcdefgh",
            Arc::clone(&releases),
            Arc::clone(&context_releases),
        );
        callbacks.concurrency = 1;
        // SAFETY: The callback context stays owned by `callbacks` until reader destruction.
        let context = unsafe { &*callbacks.context.cast::<TestContext>() };
        let read_at = CallbackReadAt::try_new(callbacks)?;
        let requests: Arc<[ReadAtRequest]> = Arc::from([
            ReadAtRequest::new(0, 2, Alignment::none()),
            ReadAtRequest::new(2, 2, Alignment::none()),
        ]);
        let results = block_on(read_at.read_ranges(requests).collect::<Vec<_>>());
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(_, result)| result.is_ok()));
        assert_eq!(context.calls.load(Ordering::Relaxed), 2);
        drop(results);
        assert_eq!(releases.load(Ordering::Relaxed), 2);
        drop(read_at);
        assert_eq!(context_releases.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn rejects_incomplete_callback_table() {
        let releases = Arc::new(AtomicUsize::new(0));
        let context_releases = Arc::new(AtomicUsize::new(0));
        let mut callbacks = callbacks(b"abc", releases, Arc::clone(&context_releases));
        callbacks.struct_size -= 1;
        let result = CallbackReadAt::try_new(callbacks);
        assert!(result.is_err());

        // The constructor did not take ownership after validation failed.
        unsafe { test_release_context(callbacks.context) };
        assert_eq!(context_releases.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn releases_partial_outputs_after_callback_failure() -> VortexResult<()> {
        let releases = Arc::new(AtomicUsize::new(0));
        let context_releases = Arc::new(AtomicUsize::new(0));
        let callbacks = callbacks(
            b"abcdefgh",
            Arc::clone(&releases),
            Arc::clone(&context_releases),
        );
        // SAFETY: The callback context stays owned by `callbacks` until reader destruction.
        unsafe { &*callbacks.context.cast::<TestContext>() }
            .fail_after_first_output
            .store(true, Ordering::Relaxed);
        let read_at = CallbackReadAt::try_new(callbacks)?;
        let requests: Arc<[ReadAtRequest]> = Arc::from([
            ReadAtRequest::new(0, 2, Alignment::none()),
            ReadAtRequest::new(2, 2, Alignment::none()),
        ]);
        let results = block_on(read_at.read_ranges(requests).collect::<Vec<_>>());
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(_, result)| result.is_err()));
        assert_eq!(releases.load(Ordering::Relaxed), 1);
        drop(read_at);
        assert_eq!(context_releases.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn observes_host_cancellation_before_callbacks() -> VortexResult<()> {
        let releases = Arc::new(AtomicUsize::new(0));
        let context_releases = Arc::new(AtomicUsize::new(0));
        let callbacks = callbacks(
            b"abcdefgh",
            Arc::clone(&releases),
            Arc::clone(&context_releases),
        );
        // SAFETY: The callback context stays owned by `callbacks` until reader destruction.
        let context = unsafe { &*callbacks.context.cast::<TestContext>() };
        context.cancelled.store(true, Ordering::Relaxed);
        let read_at = CallbackReadAt::try_new(callbacks)?;

        let error = block_on(read_at.size()).expect_err("cancelled size must fail");
        assert!(error.to_string().contains("cancelled"));
        assert_eq!(context.calls.load(Ordering::Relaxed), 0);
        assert_eq!(releases.load(Ordering::Relaxed), 0);
        drop(read_at);
        assert_eq!(context_releases.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn preserves_per_thread_errors_across_concurrent_callbacks() -> VortexResult<()> {
        let context_releases = Arc::new(AtomicUsize::new(0));
        let context = Box::new(ConcurrentErrorContext {
            barrier: Barrier::new(2),
            context_releases: Arc::clone(&context_releases),
        });
        let reader = CallbackReadAt::try_new(vx_velox_read_at_callbacks {
            struct_size: size_of::<vx_velox_read_at_callbacks>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: Box::into_raw(context).cast(),
            size: Some(concurrent_size),
            read_ranges: Some(concurrent_read_ranges),
            last_error: Some(concurrent_last_error),
            release_context: Some(concurrent_release_context),
            is_cancelled: Some(never_cancelled),
            concurrency: 2,
        })?;

        let first = reader.clone();
        let first = std::thread::spawn(move || {
            block_on(first.read_at(0, 1, Alignment::none()))
                .expect_err("the first callback must fail")
                .to_string()
        });
        let second = reader.clone();
        let second = std::thread::spawn(move || {
            block_on(second.read_at(1, 1, Alignment::none()))
                .expect_err("the second callback must fail")
                .to_string()
        });

        let first = first
            .join()
            .map_err(|_| vortex_err!("The first callback thread panicked"))?;
        let second = second
            .join()
            .map_err(|_| vortex_err!("The second callback thread panicked"))?;
        assert!(first.contains("zero"));
        assert!(second.contains("one"));
        drop(reader);
        assert_eq!(context_releases.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn opens_source_and_reports_natural_splits() -> VortexResult<()> {
        let session_handle = vx_session_new_with(|session| session);
        // SAFETY: The test owns the live session handle.
        let session = unsafe { vx_session_ref(session_handle)? }.clone();
        const ROW_COUNT: u64 = 300_000;
        const ROWS_PER_NATURAL_SPLIT: u64 = 100_000;
        let values = PrimitiveArray::from_iter(0_i64..i64::try_from(ROW_COUNT)?).into_array();
        let array = StructArray::try_new(
            ["value"].into(),
            vec![values],
            usize::try_from(ROW_COUNT)?,
            Validity::NonNullable,
        )?
        .into_array();
        let mut bytes = Vec::new();
        session
            .write_options()
            .blocking(ffi_runtime())
            .write(&mut bytes, array.to_array_iterator())?;

        let releases = Arc::new(AtomicUsize::new(0));
        let context_releases = Arc::new(AtomicUsize::new(0));
        let reader = CallbackReadAt::try_new(callbacks(
            &bytes,
            Arc::clone(&releases),
            Arc::clone(&context_releases),
        ))?;
        let reader_handle = Box::into_raw(Box::new(vx_velox_read_at(reader)));
        let mut error = std::ptr::null_mut();
        // SAFETY: The test owns all handles and output pointers.
        let source = unsafe { vx_velox_source_new(session_handle, reader_handle, &raw mut error) };
        vortex_ensure!(error.is_null(), "source open returned an error");
        vortex_ensure!(!source.is_null(), "source open returned null");

        // SAFETY: The source stays live for all calls.
        unsafe {
            assert_eq!(vx_velox_source_row_count(source), ROW_COUNT);
            assert_eq!(vx_velox_source_file_size(source), bytes.len() as u64);
        }
        let mut schema = FFI_ArrowSchema::empty();
        // SAFETY: The source and outputs stay live for this call.
        let status =
            unsafe { vx_velox_source_export_schema(source, &raw mut schema, &raw mut error) };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "source schema returned an error");
        let schema = Schema::try_from(&schema)?;
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "value");
        // SAFETY: The source stays live for this call.
        let split_count = unsafe { vx_velox_source_natural_split_count(source) };
        assert!(split_count > 0);
        let mut previous_end = 0;
        let mut previous_assignment_byte = 0;
        for index in 0..split_count {
            let mut split = vx_velox_natural_split {
                struct_size: size_of::<vx_velox_natural_split>(),
                ..Default::default()
            };
            // SAFETY: The source and outputs stay live for this call.
            let status = unsafe {
                vx_velox_source_natural_split_at(source, index, &raw mut split, &raw mut error)
            };
            assert_eq!(status, 0);
            vortex_ensure!(error.is_null(), "natural split lookup returned an error");
            assert_eq!(split.row_begin, previous_end);
            assert!(split.row_end > split.row_begin);
            assert!(split.assignment_byte >= previous_assignment_byte);
            assert!(split.assignment_byte < bytes.len() as u64);
            if index == 0 {
                assert_eq!(split.assignment_byte, 0);
            }
            previous_end = split.row_end;
            previous_assignment_byte = split.assignment_byte;
        }
        assert_eq!(previous_end, ROW_COUNT);

        let expression = vx_expression_new_with(and(
            gt(
                col("value"),
                lit(i64::try_from(ROWS_PER_NATURAL_SPLIT - 1)?),
            ),
            lt(
                col("value"),
                lit(i64::try_from(2 * ROWS_PER_NATURAL_SPLIT)?),
            ),
        ));
        let mut pruned = vec![0; split_count];
        // SAFETY: The source, expression, output, and error pointer stay live for this call.
        let status = unsafe {
            vx_velox_source_prune_natural_splits(
                source,
                expression,
                0,
                split_count,
                pruned.as_mut_ptr(),
                &raw mut error,
            )
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "natural split pruning returned an error");
        assert_eq!(pruned.first(), Some(&1));
        assert_eq!(pruned.last(), Some(&1));
        assert!(pruned[1..pruned.len() - 1].contains(&0));
        // SAFETY: The test owns this expression handle.
        unsafe { vx_velox_expression_free(expression) };

        // SAFETY: The source and error output stay live for this call.
        let data_source = unsafe { vx_velox_source_data_source(source, &raw mut error) };
        vortex_ensure!(error.is_null(), "data source conversion returned an error");
        vortex_ensure!(
            !data_source.is_null(),
            "data source conversion returned null"
        );

        let scan_options = vx_velox_scan_options {
            struct_size: size_of::<vx_velox_scan_options>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            projection: std::ptr::null(),
            filter: std::ptr::null(),
            row_range_begin: 0,
            row_range_end: 0,
            selection: vx_velox_scan_selection::default(),
            limit: 0,
            ordered: false,
        };
        // SAFETY: The data source and scan options stay live for this call.
        let scan = unsafe {
            vx_velox_data_source_scan(data_source, &raw const scan_options, &raw mut error)
        };
        vortex_ensure!(error.is_null(), "data source scan returned an error");
        vortex_ensure!(!scan.is_null(), "data source scan returned null");
        let mut scanned_rows = 0;
        loop {
            // SAFETY: The scan stays live and is consumed from one thread.
            let partition = unsafe { vx_velox_scan_next_partition(scan, &raw mut error) };
            vortex_ensure!(error.is_null(), "partition lookup returned an error");
            if partition.is_null() {
                break;
            }
            loop {
                // SAFETY: The partition stays live and is consumed from one thread.
                let array = unsafe { vx_velox_partition_next(partition, &raw mut error) };
                vortex_ensure!(error.is_null(), "partition scan returned an error");
                if array.is_null() {
                    break;
                }
                // SAFETY: The returned array stays live until the matching free call.
                scanned_rows += unsafe { vx_array_ref(array)? }.len();
                // SAFETY: The scan returned this owned array handle.
                unsafe { vx_velox_array_free(array) };
            }
            // SAFETY: The scan returned this owned partition handle.
            unsafe { vx_velox_partition_free(partition) };
        }
        assert_eq!(scanned_rows, usize::try_from(ROW_COUNT)?);

        // SAFETY: Each owned handle is freed exactly once.
        unsafe {
            vx_velox_scan_free(scan);
            vx_velox_data_source_free(data_source);
            vx_velox_source_free(source);
            vx_velox_read_at_free(reader_handle);
            vx_session_free(session_handle);
        }
        assert_eq!(context_releases.load(Ordering::Relaxed), 1);
        assert!(releases.load(Ordering::Relaxed) > 0);
        Ok(())
    }
}
