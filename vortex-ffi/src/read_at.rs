// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`VortexReadAt`] backed by caller-supplied C callbacks, so a host with its
//! own configured I/O stack can serve the bytes instead of duplicating that
//! configuration in Vortex. C form of the Java `dev.vortex.io.NativeReadable`,
//! see <https://github.com/vortex-data/vortex/issues/8433>.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::LazyLock;

use async_lock::Semaphore;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Handle;

use crate::string::vx_view;

/// Concurrency used when the caller passes 0. Matches the object-store default,
/// since a host that delegates its I/O is usually talking to remote storage.
const DEFAULT_CONCURRENCY: usize = 192;

/// Ceiling on callbacks in flight across every source in the process, since
/// per-source limits otherwise multiply. The Java bindings cap upcalls likewise.
static READ_LIMITER: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(DEFAULT_CONCURRENCY)));

/// A random-access byte source implemented by the caller.
///
/// "read_at" must tolerate concurrent calls from arbitrary threads. The struct
/// is read only during the call it is passed to, but "ctx" and the callbacks
/// must stay valid until "release" runs.
#[repr(C)]
pub struct vx_readat {
    /// Opaque caller state passed to every callback. May be NULL.
    pub ctx: *mut c_void,
    /// Total length of the source in bytes. Must be exact: the footer is read
    /// relative to the end, so a wrong length surfaces as a corrupt file.
    pub len: u64,
    /// Maximum number of concurrent "read_at" calls for this source. 0 selects a
    /// default. A process-wide ceiling applies across all sources as well.
    pub concurrency: usize,
    /// Optional name, typically the URI, used for cache keys and error messages;
    /// it should be stable and unique. Copied. Zero-length means anonymous.
    pub name: vx_view,
    /// Required. Writes "length" bytes at "offset" into "dst" and returns the
    /// count written; a short count or a negative value fails the read.
    pub read_at: Option<
        unsafe extern "C" fn(ctx: *mut c_void, offset: u64, dst: *mut u8, length: usize) -> i64,
    >,
    /// Optional. Called once, before the call that drops the source returns -
    /// on that thread, or on a worker thread if any are configured.
    pub release: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
}

/// The callbacks and context, owned for as long as Vortex holds the source.
struct CReadAtInner {
    ctx: *mut c_void,
    len: u64,
    concurrency: usize,
    name: Option<Arc<str>>,
    read_at: unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> i64,
    release: Option<unsafe extern "C" fn(*mut c_void)>,
}

// SAFETY: `vx_readat` requires `read_at` to be callable from any thread and `ctx`
// to stay valid until `release`, which runs only once the last `Arc` drops.
unsafe impl Send for CReadAtInner {}
unsafe impl Sync for CReadAtInner {}

impl Drop for CReadAtInner {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            // SAFETY: last owner, so every read has finished - reads hold this `Arc`.
            unsafe { release(self.ctx) };
        }
    }
}

/// A [`VortexReadAt`] that forwards positional reads to C callbacks.
///
/// Reads run on the blocking pool: the callback blocks, and the FFI runtime is
/// current-thread, so calling it from a future would stall every other task.
struct CReadAt {
    inner: Arc<CReadAtInner>,
    handle: Handle,
}

impl VortexReadAt for CReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        self.inner.name.as_ref()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        // Host I/O behind the callback is usually remote: prefer fewer, larger reads.
        Some(CoalesceConfig::object_storage())
    }

    fn concurrency(&self) -> usize {
        self.inner.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.inner.len;

        async move { Ok(len) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let inner = Arc::clone(&self.inner);
        let handle = self.handle.clone();

        async move {
            // Take a permit before occupying a blocking thread. The lock-free path
            // barges ahead of waiters: the limiter must not become the bottleneck.
            let permit = match READ_LIMITER.try_acquire_arc() {
                Some(permit) => permit,
                None => READ_LIMITER.acquire_arc().await,
            };

            handle
                .spawn_blocking(move || {
                    // Cancelling the read drops the task handle but cannot interrupt a
                    // callback already running, so the permit stays with the work.
                    let _permit = permit;
                    let end = offset
                        .checked_add(length as u64)
                        .ok_or_else(|| vortex_err!("read {offset}+{length} overflows u64"))?;
                    if end > inner.len {
                        vortex_bail!(
                            "read {offset}..{end} out of bounds for source of length {}",
                            inner.len
                        );
                    }

                    let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);

                    if length > 0 {
                        // SAFETY: spare capacity covers `length` bytes; the pointer is
                        // not retained past the call.
                        let written = unsafe {
                            (inner.read_at)(
                                inner.ctx,
                                offset,
                                buffer.spare_capacity_mut().as_mut_ptr().cast(),
                                length,
                            )
                        };
                        if written < 0 {
                            vortex_bail!(
                                "read_at callback failed with code {written} for {offset}..{end}"
                            );
                        }
                        // A host returning short without saying so would leave the tail of
                        // `dst` uninitialized for the scan to read.
                        if written as u64 != length as u64 {
                            vortex_bail!(
                                "read_at callback wrote {written} of {length} bytes for {offset}..{end}"
                            );
                        }
                    }

                    // SAFETY: the callback reported writing all `length` bytes, checked above.
                    unsafe { buffer.set_len(length) };

                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
                .await
        }
        .boxed()
    }
}

/// Build an owned [`VortexReadAt`] from a caller-supplied `vx_readat`.
///
/// On error the caller keeps `ctx`: `release` is wired up only once the source
/// exists, so a rejected descriptor never double-frees.
pub(crate) unsafe fn read_at_from_ffi(
    reader: *const vx_readat,
) -> VortexResult<Arc<dyn VortexReadAt>> {
    vortex_ensure!(!reader.is_null(), "null vx_readat");

    let reader = unsafe { &*reader };
    let read_at = reader
        .read_at
        .ok_or_else(|| vortex_err!("vx_readat.read_at is required"))?;

    // `as_str` rejects a null pointer with a non-zero length, which a manual
    // null check here would silently accept as anonymous.
    let name = unsafe { reader.name.as_str() }?;
    let name = (!name.is_empty()).then(|| Arc::from(name));

    let concurrency = if reader.concurrency == 0 {
        DEFAULT_CONCURRENCY
    } else {
        reader.concurrency
    };

    Ok(Arc::new(CReadAt {
        inner: Arc::new(CReadAtInner {
            ctx: reader.ctx,
            len: reader.len,
            concurrency,
            name,
            read_at,
            release: reader.release,
        }),
        handle: crate::RUNTIME.handle(),
    }))
}
