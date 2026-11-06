// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Host memory allocation, copies, and device allocations with stream-ordered access tracking.
//!
//! Every allocation tracks which streams accessed it, so work on one stream never uses memory
//! before another stream's earlier write to it completes, and memory is never freed or
//! rewritten before other streams' earlier accesses complete. Event tracking depends on the
//! access kind and stream:
//!
//! - A write records one event right after it is enqueued, taken from the context's event
//!   pool, because later readers on other streams must wait for exactly that write rather than
//!   for whatever their writer enqueued afterwards.
//! - A read waits for the last write only if that write was on another stream, and only once
//!   per stream and write.
//! - A read on another stream than the allocation's records a per-stream event, reused for
//!   that stream's later reads, so later writes and the free can wait for it.
//! - Reads on the allocation stream, and the allocation itself, record nothing. If a later
//!   access on another stream must follow them, it records the allocation stream at that time
//!   and waits for that, which also covers them because they were enqueued earlier.
//! - The free runs on the allocation stream after waiting only for other streams' accesses.
//!
//! Callers must use a pointer only on the stream they passed when obtaining it, and keep the
//! returned [`SyncOnDrop`] alive until the work using it is enqueued, then finish it explicitly
//! to receive tracking errors.

#![expect(
    clippy::same_name_method,
    reason = "inherent accessors mirror the DeviceSlice ones so callers need not import the trait"
)]
#![expect(
    clippy::expect_used,
    reason = "out-of-range views and overflowing sizes are caller bugs, as with std slices"
)]

use std::ffi::c_uint;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Bound;
use std::ops::RangeBounds;
use std::sync::Arc;

use parking_lot::Mutex;
use smallvec::SmallVec;

use crate::DriverError;
use crate::context::CudaContext;
use crate::error::log_error;
use crate::event::CudaEvent;
use crate::repr::DeviceRepr;
use crate::repr::ValidAsZeroBits;
use crate::stream::CudaStream;
use crate::stream::StreamId;
use crate::sys;

/// Allocates `bytes` of page-locked host memory with the given `cuMemHostAlloc` flags.
///
/// # Safety
///
/// A CUDA context must be current; the memory is uninitialized.
pub unsafe fn malloc_host(bytes: usize, flags: c_uint) -> Result<*mut c_void, DriverError> {
    let mut pointer = MaybeUninit::uninit();
    // SAFETY: The driver writes the allocation into `pointer` on success.
    unsafe { sys::cuMemHostAlloc(pointer.as_mut_ptr(), bytes, flags) }.result()?;
    // SAFETY: Initialized by the successful call.
    Ok(unsafe { pointer.assume_init() })
}

/// Frees memory allocated by [`malloc_host`].
///
/// # Safety
///
/// `pointer` must come from [`malloc_host`] and no pending copy may still use it.
pub unsafe fn free_host(pointer: *mut c_void) -> Result<(), DriverError> {
    // SAFETY: Under the caller's guarantees.
    unsafe { sys::cuMemFreeHost(pointer) }.result()
}

/// Synchronously copies `target.len()` elements from device memory at `source`.
///
/// # Safety
///
/// `source` must address at least `size_of_val(target)` readable device bytes.
pub unsafe fn memcpy_dtoh_sync<T>(
    target: &mut [T],
    source: sys::CUdeviceptr,
) -> Result<(), DriverError> {
    // SAFETY: Under the caller's guarantees; `target` is writable for its whole size.
    unsafe { sys::cuMemcpyDtoH_v2(target.as_mut_ptr().cast(), source, size_of_val(target)) }
        .result()
}

/// An event recorded on a stream after an access, owned by the access state.
#[derive(Clone, Copy)]
struct Recorded {
    stream: StreamId,
    event: sys::CUevent,
}

#[derive(Default)]
struct AccessState {
    /// The last write and the event recorded right after it.
    write: Option<Recorded>,
    /// Streams that already waited for `write`, besides its own.
    waited: SmallVec<[StreamId; 4]>,
    /// The latest read on each stream other than the allocation stream.
    reads: SmallVec<[Recorded; 2]>,
    /// Whether the allocation stream holds the allocation or reads newer than `snapshot`.
    allocation_stream_pending: bool,
    /// The allocation stream, recorded when another stream had to follow its pending work.
    snapshot: Option<sys::CUevent>,
}

// SAFETY: Event handles are plain driver handles, usable from any thread.
unsafe impl Send for AccessState {}

/// Stream ordering of one allocation's accesses, shared by its views.
pub(crate) struct Access {
    state: Mutex<AccessState>,
}

/// Whether an access reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AccessKind {
    Read,
    Write,
}

impl Access {
    fn new() -> Self {
        Self {
            state: Mutex::new(AccessState {
                allocation_stream_pending: true,
                ..AccessState::default()
            }),
        }
    }

    /// Makes `stream` wait for earlier accesses that an access of `kind` must follow.
    pub(crate) fn before(
        &self,
        owner: &CudaStream,
        stream: &CudaStream,
        kind: AccessKind,
    ) -> Result<(), DriverError> {
        let context = stream.context();
        // Access events come from the accessing stream's context and return to the pool of
        // whichever context later releases them, and events record only on their own context.
        assert_eq!(
            context.cu_ctx(),
            owner.context().cu_ctx(),
            "CUDA memory accessed from a stream of another context"
        );
        stream.bind()?;
        let mut state = self.state.lock();
        let id = stream.id();
        if let Some(write) = state.write
            && write.stream != id
            && !state.waited.contains(&id)
        {
            stream.wait_raw(write.event)?;
            state.waited.push(id);
        }
        // Unrecorded allocation-stream work: the allocation, which every access follows, and
        // reads, which writes follow. A recorded write already follows the allocation.
        let follow_owner = id != owner.id()
            && match kind {
                AccessKind::Read => state.write.is_none(),
                AccessKind::Write => true,
            };
        if follow_owner {
            if state.allocation_stream_pending {
                // Record everything enqueued there so far, which includes that work.
                let event = match state.snapshot {
                    Some(event) => event,
                    None => {
                        let event = context.take_event()?;
                        state.snapshot = Some(event);
                        event
                    }
                };
                owner.record_raw(event)?;
                state.allocation_stream_pending = false;
            }
            if let Some(snapshot) = state.snapshot {
                stream.wait_raw(snapshot)?;
            }
        }
        if kind == AccessKind::Write {
            for read in &state.reads {
                if read.stream != id {
                    stream.wait_raw(read.event)?;
                }
            }
        }
        Ok(())
    }

    /// Records an access of `kind` on `stream`, after it was enqueued.
    pub(crate) fn after(
        &self,
        owner: &CudaStream,
        stream: &CudaStream,
        kind: AccessKind,
    ) -> Result<(), DriverError> {
        let context = stream.context();
        stream.bind()?;
        let mut state = self.state.lock();
        let id = stream.id();
        match kind {
            AccessKind::Write => {
                // Readers and older writes are ordered before this write by `before`, so only
                // this write remains to be followed.
                let event = match state.write {
                    Some(previous) => previous.event,
                    None => context.take_event()?,
                };
                if let Err(error) = stream.record_raw(event) {
                    if state.write.is_none() {
                        context.give_event(event);
                    }
                    return Err(error);
                }
                state.write = Some(Recorded { stream: id, event });
                state.waited.clear();
                for read in state.reads.drain(..) {
                    context.give_event(read.event);
                }
                if id == owner.id() {
                    state.allocation_stream_pending = false;
                }
            }
            AccessKind::Read if id == owner.id() => {
                state.allocation_stream_pending = true;
            }
            AccessKind::Read => {
                let index = state.reads.iter().position(|read| read.stream == id);
                let event = match index {
                    Some(index) => state.reads[index].event,
                    None => context.take_event()?,
                };
                if let Err(error) = stream.record_raw(event) {
                    if index.is_none() {
                        context.give_event(event);
                    }
                    return Err(error);
                }
                if index.is_none() {
                    state.reads.push(Recorded { stream: id, event });
                }
            }
        }
        Ok(())
    }

    /// Like [`after`](Self::after), but if recording fails, waits for the access to complete
    /// instead, so later accesses and the free need not follow it.
    ///
    /// The access is already enqueued, so its memory must not be reused before it completes.
    pub(crate) fn after_or_wait(
        &self,
        owner: &CudaStream,
        stream: &CudaStream,
        kind: AccessKind,
    ) -> Result<(), DriverError> {
        let recorded = self.after(owner, stream, kind);
        // If recovery fails too, the context is broken and its work does not complete.
        if recorded.is_err()
            && let Err(error) = stream.synchronize()
        {
            log_error("wait after access recording failure", error);
        }
        recorded
    }

    /// Makes the allocation stream wait for every other stream's access before a free.
    fn before_free(&self, owner: &CudaStream) -> Result<(), DriverError> {
        owner.bind()?;
        let state = self.state.lock();
        let id = owner.id();
        if let Some(write) = state.write
            && write.stream != id
            && !state.waited.contains(&id)
        {
            owner.wait_raw(write.event)?;
        }
        for read in &state.reads {
            owner.wait_raw(read.event)?;
        }
        Ok(())
    }

    /// Returns the access events to the context's pool.
    fn release(&mut self, context: &CudaContext) {
        let state = self.state.get_mut();
        if let Some(write) = state.write.take() {
            context.give_event(write.event);
        }
        for read in state.reads.drain(..) {
            context.give_event(read.event);
        }
        if let Some(snapshot) = state.snapshot.take() {
            context.give_event(snapshot);
        }
    }
}

/// Records an access or waits for work when finished or dropped, after the work was enqueued.
///
/// Use [`finish`](Self::finish) to receive cleanup errors. Dropping the guard only logs them.
#[must_use]
pub struct SyncOnDrop<'a> {
    pending: Pending<'a>,
}

enum Pending<'a> {
    None,
    Access(&'a Access, &'a CudaStream, &'a CudaStream, AccessKind),
    Event(&'a CudaEvent, &'a CudaStream),
    Wait(&'a CudaStream),
    #[cfg(test)]
    Cleanup(&'a mut dyn FnMut() -> Result<(), DriverError>),
}

impl<'a> SyncOnDrop<'a> {
    pub(crate) fn new(
        access: &'a Access,
        owner: &'a CudaStream,
        stream: &'a CudaStream,
        kind: AccessKind,
    ) -> Self {
        Self {
            pending: Pending::Access(access, owner, stream, kind),
        }
    }

    /// A guard that records nothing.
    pub fn none() -> Self {
        Self {
            pending: Pending::None,
        }
    }

    /// A guard that records `stream`'s work into `event` when finished or dropped.
    pub fn record_event(event: &'a CudaEvent, stream: &'a CudaStream) -> Self {
        Self {
            pending: Pending::Event(event, stream),
        }
    }

    /// A guard that waits for `stream`'s work when finished or dropped.
    pub fn wait_for(stream: &'a CudaStream) -> Self {
        Self {
            pending: Pending::Wait(stream),
        }
    }

    /// Performs the pending cleanup, returning any error without logging it.
    ///
    /// The guard is disarmed before cleanup, so dropping it does not retry failed cleanup.
    pub fn finish(mut self) -> Result<(), DriverError> {
        self.complete()
    }

    /// Finishes every guard, returning the first error, if any.
    ///
    /// Later guards still finish after a failure, and only their errors are logged.
    pub fn finish_all(guards: impl IntoIterator<Item = Self>) -> Result<(), DriverError> {
        let mut result = Ok(());
        for guard in guards {
            let finished = guard.finish();
            if result.is_ok() {
                result = finished;
            } else if let Err(error) = finished {
                log_error("finish additional guard", error);
            }
        }
        result
    }

    fn complete(&mut self) -> Result<(), DriverError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::None => Ok(()),
            Pending::Access(access, owner, stream, kind) => {
                access.after_or_wait(owner, stream, kind)
            }
            Pending::Event(event, stream) => stream
                .bind()
                .and_then(|()| stream.record_raw(event.cu_event())),
            Pending::Wait(stream) => {
                // If this fails, the context is broken and its copies do not complete.
                stream.synchronize()
            }
            #[cfg(test)]
            Pending::Cleanup(cleanup) => cleanup(),
        }
    }

    /// Disarms a wait already satisfied by a successful synchronization of `stream`.
    /// Other streams' waits and access/event recording must still run on finish or drop.
    fn disarm_wait_for(&mut self, stream: &CudaStream) {
        if matches!(&self.pending, Pending::Wait(waiting) if waiting.id() == stream.id()) {
            self.pending = Pending::None;
        }
    }
}

impl Drop for SyncOnDrop<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.complete() {
            log_error("finish guard on drop", error);
        }
    }
}

impl std::fmt::Debug for SyncOnDrop<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pending = match &self.pending {
            Pending::None => "none",
            Pending::Access(..) => "access",
            Pending::Event(..) => "event",
            Pending::Wait(..) => "wait",
            #[cfg(test)]
            Pending::Cleanup(..) => "cleanup",
        };
        f.debug_struct("SyncOnDrop")
            .field("pending", &pending)
            .finish()
    }
}

/// A typed device allocation, freed in stream order on its allocation stream.
pub struct CudaSlice<T> {
    pointer: sys::CUdeviceptr,
    len: usize,
    access: Access,
    stream: Arc<CudaStream>,
    marker: PhantomData<*const T>,
}

// SAFETY: The allocation is device memory reached only through the driver, and the access
// state is synchronized.
unsafe impl<T> Send for CudaSlice<T> {}
// SAFETY: As above; shared references only read the handle and lock the access state.
unsafe impl<T> Sync for CudaSlice<T> {}

impl<T> std::fmt::Debug for CudaSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaSlice")
            .field("pointer", &self.pointer)
            .field("len", &self.len)
            .field("stream", &self.stream.id())
            .finish()
    }
}

impl<T> Drop for CudaSlice<T> {
    fn drop(&mut self) {
        let context = self.stream.context();
        // If ordering fails, other streams may still access the memory: leak it instead.
        match self.access.before_free(&self.stream) {
            Ok(()) => {
                if let Err(error) = self.stream.free_raw(self.pointer) {
                    log_error("free device allocation", error);
                }
            }
            Err(error) => log_error("order device allocation free", error),
        }
        self.access.release(context);
    }
}

impl<T> CudaSlice<T> {
    /// Wraps an allocation made on `stream`.
    ///
    /// # Safety
    ///
    /// `pointer` must be a stream-ordered allocation of at least `len` elements made on
    /// `stream`, owned by the returned slice.
    pub(crate) unsafe fn from_raw(
        pointer: sys::CUdeviceptr,
        len: usize,
        stream: Arc<CudaStream>,
    ) -> Self {
        Self {
            pointer,
            len,
            access: Access::new(),
            stream,
            marker: PhantomData,
        }
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The number of bytes.
    pub fn num_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Whether the slice holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The device ordinal.
    pub fn ordinal(&self) -> usize {
        self.stream.context().ordinal()
    }

    /// The context the memory belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.stream.context()
    }

    /// The stream that allocated the memory and will free it.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The device address, to inspect or compute pointers from without accessing the memory.
    ///
    /// Unlike [`DevicePtr::device_ptr`], this orders nothing: work using the address must be
    /// ordered after the memory's other accesses, for example through a device pointer guard.
    pub fn device_address(&self) -> sys::CUdeviceptr {
        self.pointer
    }

    /// The whole slice as a view.
    pub fn as_view(&self) -> CudaView<'_, T> {
        CudaView {
            pointer: self.pointer,
            len: self.len,
            access: &self.access,
            stream: &self.stream,
            marker: PhantomData,
        }
    }

    /// The whole slice as a mutable view.
    pub fn as_view_mut(&mut self) -> CudaViewMut<'_, T> {
        CudaViewMut {
            pointer: self.pointer,
            len: self.len,
            access: &self.access,
            stream: &self.stream,
            marker: PhantomData,
        }
    }

    /// A view of the elements in `bounds`.
    ///
    /// # Panics
    ///
    /// Panics if `bounds` is out of range.
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> CudaView<'_, T> {
        self.as_view().slice(bounds)
    }

    /// A mutable view of the elements in `bounds`.
    ///
    /// # Panics
    ///
    /// Panics if `bounds` is out of range.
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        self.as_view_mut().into_slice_mut(bounds)
    }

    /// Reinterprets the memory as `len` elements of `S`, if they fit.
    ///
    /// # Safety
    ///
    /// The bytes must be valid values of `S`, and the memory must be aligned for `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'_, S>> {
        fits::<S>(len, self.num_bytes()).then_some(CudaView {
            pointer: self.pointer,
            len,
            access: &self.access,
            stream: &self.stream,
            marker: PhantomData,
        })
    }

    /// Reinterprets the memory as `len` mutable elements of `S`, if they fit.
    ///
    /// # Safety
    ///
    /// The bytes must be valid values of `S`, the memory must be aligned for `S`, and `T` must
    /// accept any bytes written as `S`.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'_, S>> {
        fits::<S>(len, self.num_bytes()).then_some(CudaViewMut {
            pointer: self.pointer,
            len,
            access: &self.access,
            stream: &self.stream,
            marker: PhantomData,
        })
    }
}

impl<T: DeviceRepr> CudaSlice<T> {
    /// A device copy of the slice on its stream.
    pub fn try_clone(&self) -> Result<Self, DriverError> {
        self.stream.clone_dtod(self)
    }
}

/// Whether `len` elements of `S` fit in `bytes`; an overflowing size does not.
fn fits<S>(len: usize, bytes: usize) -> bool {
    len.checked_mul(size_of::<S>())
        .is_some_and(|needed| needed <= bytes)
}

/// Resolves `bounds` against `len`, if in range.
fn range(bounds: &impl RangeBounds<usize>, len: usize) -> Option<(usize, usize)> {
    let start = match bounds.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1)?,
        Bound::Unbounded => 0,
    };
    let end = match bounds.end_bound() {
        Bound::Included(&end) => end.checked_add(1)?,
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    (start <= end && end <= len).then_some((start, end))
}

/// A shared view of part of a [`CudaSlice`].
pub struct CudaView<'a, T> {
    pointer: sys::CUdeviceptr,
    len: usize,
    access: &'a Access,
    stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a [T]>,
}

impl<T> Clone for CudaView<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for CudaView<'_, T> {}

impl<T> std::fmt::Debug for CudaView<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaView")
            .field("pointer", &self.pointer)
            .field("len", &self.len)
            .finish()
    }
}

impl<'a, T> CudaView<'a, T> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of bytes.
    pub fn num_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// A view of the elements in `bounds`.
    ///
    /// # Panics
    ///
    /// Panics if `bounds` is out of range.
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> CudaView<'a, T> {
        self.try_slice(bounds)
            .expect("CUDA view range out of bounds")
    }

    /// A view of the elements in `bounds`, if in range.
    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<CudaView<'a, T>> {
        let (start, end) = range(&bounds, self.len)?;
        Some(CudaView {
            pointer: self.pointer + (start * size_of::<T>()) as u64,
            len: end - start,
            ..*self
        })
    }

    /// Reinterprets the view as `len` elements of `S`, if they fit.
    ///
    /// # Safety
    ///
    /// The bytes must be valid values of `S`, and the memory must be aligned for `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'a, S>> {
        fits::<S>(len, self.num_bytes()).then_some(CudaView {
            pointer: self.pointer,
            len,
            access: self.access,
            stream: self.stream,
            marker: PhantomData,
        })
    }
}

/// A mutable view of part of a [`CudaSlice`].
pub struct CudaViewMut<'a, T> {
    pointer: sys::CUdeviceptr,
    len: usize,
    access: &'a Access,
    stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a mut [T]>,
}

impl<T> std::fmt::Debug for CudaViewMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaViewMut")
            .field("pointer", &self.pointer)
            .field("len", &self.len)
            .finish()
    }
}

impl<'a, T> CudaViewMut<'a, T> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of bytes.
    pub fn num_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// The view as a shared view.
    pub fn as_view(&self) -> CudaView<'_, T> {
        CudaView {
            pointer: self.pointer,
            len: self.len,
            access: self.access,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// A shared view of the elements in `bounds`.
    ///
    /// # Panics
    ///
    /// Panics if `bounds` is out of range.
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> CudaView<'_, T> {
        self.as_view().slice(bounds)
    }

    /// A mutable view of the elements in `bounds`.
    ///
    /// # Panics
    ///
    /// Panics if `bounds` is out of range.
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        // The return lifetime keeps `self` mutably borrowed while the copied view lives.
        CudaViewMut { ..*self }.into_slice_mut(bounds)
    }

    fn into_slice_mut(self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'a, T> {
        let (start, end) = range(&bounds, self.len).expect("CUDA view range out of bounds");
        CudaViewMut {
            pointer: self.pointer + (start * size_of::<T>()) as u64,
            len: end - start,
            ..self
        }
    }

    /// Reinterprets the view as `len` mutable elements of `S`, if they fit.
    ///
    /// # Safety
    ///
    /// The bytes must be valid values of `S`, the memory must be aligned for `S`, and `T` must
    /// accept any bytes written as `S`.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'_, S>> {
        fits::<S>(len, self.num_bytes()).then_some(CudaViewMut {
            pointer: self.pointer,
            len,
            access: self.access,
            stream: self.stream,
            marker: PhantomData,
        })
    }
}

/// Orders `stream` after earlier conflicting accesses and returns the guard recording this one.
fn access<'a>(
    access: &'a Access,
    owner: &'a CudaStream,
    stream: &'a CudaStream,
    kind: AccessKind,
) -> Result<SyncOnDrop<'a>, DriverError> {
    access.before(owner, stream, kind)?;
    Ok(SyncOnDrop::new(access, owner, stream, kind))
}

/// Keeps the device memory traits implemented only here: copies trust their lengths and access
/// tracking for memory safety.
mod sealed {
    pub trait Sealed {}
}

impl<T> sealed::Sealed for CudaSlice<T> {}
impl<T> sealed::Sealed for CudaView<'_, T> {}
impl<T> sealed::Sealed for CudaViewMut<'_, T> {}

/// Device memory with a length and an allocation stream.
pub trait DeviceSlice<T>: sealed::Sealed {
    /// The number of elements.
    fn len(&self) -> usize;

    /// The number of bytes.
    fn num_bytes(&self) -> usize {
        self.len() * size_of::<T>()
    }

    /// Whether there are no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The stream that allocated the memory.
    fn stream(&self) -> &Arc<CudaStream>;
}

/// Device memory that can be read on a stream.
pub trait DevicePtr<T>: DeviceSlice<T> {
    /// The device pointer for reading on `stream`, which first waits for earlier writes.
    ///
    /// Finish the guard after the read is enqueued to receive cleanup errors, or drop it to
    /// log them. Ordering failures return before providing a pointer.
    ///
    /// # Panics
    ///
    /// Panics if `stream` belongs to a different underlying CUDA context than the allocation,
    /// even on the same device. Different streams and context wrappers sharing the same CUDA
    /// context are supported.
    fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(sys::CUdeviceptr, SyncOnDrop<'a>), DriverError>;
}

/// Device memory that can be written on a stream.
pub trait DevicePtrMut<T>: DeviceSlice<T> {
    /// The device pointer for writing on `stream`, which first waits for earlier accesses.
    ///
    /// Finish the guard after the write is enqueued to receive cleanup errors, or drop it to
    /// log them. Ordering failures return before providing a pointer.
    ///
    /// # Panics
    ///
    /// Panics if `stream` belongs to a different underlying CUDA context than the allocation,
    /// even on the same device. Different streams and context wrappers sharing the same CUDA
    /// context are supported.
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> Result<(sys::CUdeviceptr, SyncOnDrop<'a>), DriverError>;
}

impl<T> DeviceSlice<T> for CudaSlice<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T> DeviceSlice<T> for CudaView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

impl<T> DeviceSlice<T> for CudaViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

impl<T> DevicePtr<T> for CudaSlice<T> {
    fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(sys::CUdeviceptr, SyncOnDrop<'a>), DriverError> {
        let guard = access(&self.access, &self.stream, stream, AccessKind::Read)?;
        Ok((self.pointer, guard))
    }
}

impl<T> DevicePtr<T> for CudaView<'_, T> {
    fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(sys::CUdeviceptr, SyncOnDrop<'a>), DriverError> {
        let guard = access(self.access, self.stream, stream, AccessKind::Read)?;
        Ok((self.pointer, guard))
    }
}

impl<T> DevicePtr<T> for CudaViewMut<'_, T> {
    fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(sys::CUdeviceptr, SyncOnDrop<'a>), DriverError> {
        let guard = access(self.access, self.stream, stream, AccessKind::Read)?;
        Ok((self.pointer, guard))
    }
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> Result<(sys::CUdeviceptr, SyncOnDrop<'a>), DriverError> {
        let guard = access(&self.access, &self.stream, stream, AccessKind::Write)?;
        Ok((self.pointer, guard))
    }
}

impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> Result<(sys::CUdeviceptr, SyncOnDrop<'a>), DriverError> {
        let guard = access(self.access, self.stream, stream, AccessKind::Write)?;
        Ok((self.pointer, guard))
    }
}

/// The raw parts a kernel launch needs to order and record an argument's access.
pub(crate) struct ArgAccess<'a> {
    pub(crate) access: &'a Access,
    pub(crate) owner: &'a CudaStream,
    pub(crate) kind: AccessKind,
}

impl<T> CudaSlice<T> {
    pub(crate) fn arg_access(&self, kind: AccessKind) -> (ArgAccess<'_>, &sys::CUdeviceptr) {
        (
            ArgAccess {
                access: &self.access,
                owner: &self.stream,
                kind,
            },
            &self.pointer,
        )
    }
}

impl<'a, T> CudaView<'a, T> {
    pub(crate) fn arg_access(&self) -> (ArgAccess<'a>, &sys::CUdeviceptr) {
        (
            ArgAccess {
                access: self.access,
                owner: self.stream,
                kind: AccessKind::Read,
            },
            &self.pointer,
        )
    }
}

impl<'a, T> CudaViewMut<'a, T> {
    pub(crate) fn arg_access(&self) -> (ArgAccess<'a>, &sys::CUdeviceptr) {
        (
            ArgAccess {
                access: self.access,
                owner: self.stream,
                kind: AccessKind::Write,
            },
            &self.pointer,
        )
    }
}

/// Host memory that copies can read from or write into.
pub trait HostSlice<T> {
    /// The number of elements.
    fn len(&self) -> usize;

    /// Whether there are no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements for a copy on `stream`, with a guard to finish after the copy is enqueued.
    ///
    /// Binding or memory-probe failures return before providing a slice.
    ///
    /// # Safety
    ///
    /// The slice may be used only by work enqueued on `stream` before the guard drops.
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(&'a [T], SyncOnDrop<'a>), DriverError>;

    /// The elements for a copy into them on `stream`, with a guard as above.
    ///
    /// # Safety
    ///
    /// As for [`stream_synced_slice`](Self::stream_synced_slice).
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> Result<(&'a mut [T], SyncOnDrop<'a>), DriverError>;
}

/// The guard of a copy of a borrowed host slice on `stream`, which must not outlive the borrow.
///
/// Copies read and write page-locked memory asynchronously, so the guard waits for them. The
/// driver stages copies of pageable memory, which no longer read the slice once enqueued.
fn borrowed<'a, T>(slice: &[T], stream: &'a CudaStream) -> Result<SyncOnDrop<'a>, DriverError> {
    let mut memory_type: c_uint = 0;
    let mut attribute = sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    let mut data = (&raw mut memory_type).cast();

    stream.bind()?;
    // The plural API succeeds with a zero memory type for unregistered pageable pointers.
    // The singular API returns INVALID_VALUE, which Compute Sanitizer reports even if handled.
    // SAFETY: The context is current; attribute and data each hold one entry,
    // and data points to writable memory_type storage.
    unsafe {
        sys::cuPointerGetAttributes(
            1,
            &raw mut attribute,
            &raw mut data,
            slice.as_ptr() as sys::CUdeviceptr,
        )
    }
    .result()?;

    // CUDA accesses pinned host memory directly, without pageable-memory staging.
    // Wait before ending the borrow so the caller cannot free or modify memory still in use.
    Ok(if memory_type == sys::CU_MEMORYTYPE_HOST {
        SyncOnDrop::wait_for(stream)
    } else {
        SyncOnDrop::none()
    })
}

impl<T> HostSlice<T> for [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(&'a [T], SyncOnDrop<'a>), DriverError> {
        Ok((self, borrowed(self, stream)?))
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> Result<(&'a mut [T], SyncOnDrop<'a>), DriverError> {
        let guard = borrowed(self, stream)?;
        Ok((self, guard))
    }
}

impl<T, const N: usize> HostSlice<T> for [T; N] {
    fn len(&self) -> usize {
        N
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(&'a [T], SyncOnDrop<'a>), DriverError> {
        Ok((self, borrowed(self, stream)?))
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> Result<(&'a mut [T], SyncOnDrop<'a>), DriverError> {
        let guard = borrowed(self, stream)?;
        Ok((self, guard))
    }
}

impl<T> HostSlice<T> for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(&'a [T], SyncOnDrop<'a>), DriverError> {
        Ok((self, borrowed(self, stream)?))
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> Result<(&'a mut [T], SyncOnDrop<'a>), DriverError> {
        let guard = borrowed(self, stream)?;
        Ok((self, guard))
    }
}

impl CudaStream {
    /// Allocates `len` uninitialized elements on this stream.
    ///
    /// # Safety
    ///
    /// The memory is uninitialized.
    pub unsafe fn alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("CUDA allocation size overflow");
        let pointer = self.alloc_raw(bytes)?;
        // SAFETY: A fresh stream-ordered allocation of `len` elements on this stream.
        Ok(unsafe { CudaSlice::from_raw(pointer, len, Arc::clone(self)) })
    }

    /// Allocates `len` zeroed elements on this stream.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        // SAFETY: Zeroed below; zero bits are valid values of `T`.
        let mut slice = unsafe { self.alloc(len) }?;
        self.memset_zeros(&mut slice)?;
        Ok(slice)
    }

    /// Zeroes `target` on this stream.
    pub fn memset_zeros<T: DeviceRepr + ValidAsZeroBits, Target: DevicePtrMut<T>>(
        self: &Arc<Self>,
        target: &mut Target,
    ) -> Result<(), DriverError> {
        let bytes = target.num_bytes();
        let (pointer, record) = target.device_ptr_mut(self)?;
        let result = self.memset_d8_raw(pointer, 0, bytes);
        let finished = record.finish();
        result.inspect_err(|_| {
            if let Err(error) = finished {
                log_error("finish device access after memset", error);
            }
        })?;
        finished
    }

    /// Copies `source` into a new allocation on this stream.
    pub fn clone_htod<T: DeviceRepr, Source: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        source: &Source,
    ) -> Result<CudaSlice<T>, DriverError> {
        // SAFETY: Fully written by the copy below.
        let mut target = unsafe { self.alloc(source.len()) }?;
        self.memcpy_htod(source, &mut target)?;
        Ok(target)
    }

    /// Copies `source` into `target` on this stream.
    ///
    /// Returns once the copy no longer reads `source`, as the guard of its [`HostSlice`]
    /// implementation ensures: for slices, the driver stages pageable memory before this
    /// returns, and the guard waits for copies of page-locked memory.
    ///
    /// # Panics
    ///
    /// Panics if `target` is shorter than `source`.
    pub fn memcpy_htod<T: DeviceRepr, Source: HostSlice<T> + ?Sized, Target: DevicePtrMut<T>>(
        self: &Arc<Self>,
        source: &Source,
        target: &mut Target,
    ) -> Result<(), DriverError> {
        // SAFETY: The slice is used only by the copy below, enqueued on this stream.
        let (source, record_source) = unsafe { source.stream_synced_slice(self) }?;
        // SAFETY: The guard keeps the source borrowed until the copy no longer reads it.
        let result = unsafe { self.memcpy_htod_async(source, target) };
        let finished = record_source.finish();
        result.inspect_err(|_| {
            if let Err(error) = finished {
                log_error("finish host access after host-to-device copy", error);
            }
        })?;
        finished
    }

    /// Copies `source` into `target` on this stream, without waiting for copies of page-locked
    /// memory.
    ///
    /// # Safety
    ///
    /// `source` must stay alive and unmodified until the copy completes, such as until an event
    /// recorded after it completes.
    ///
    /// # Panics
    ///
    /// Panics if `target` is shorter than `source`.
    pub unsafe fn memcpy_htod_async<T: DeviceRepr, Target: DevicePtrMut<T>>(
        self: &Arc<Self>,
        source: &[T],
        target: &mut Target,
    ) -> Result<(), DriverError> {
        // The copied slice, not `HostSlice::len`, bounds the copy.
        assert!(target.len() >= source.len());
        let (pointer, record) = target.device_ptr_mut(self)?;
        let result = self.bind().and_then(|()| {
            // SAFETY: The target holds at least `size_of_val(source)` bytes, ordered for writing.
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    pointer,
                    source.as_ptr().cast(),
                    size_of_val(source),
                    self.cu_stream(),
                )
            }
            .result()
        });
        let finished = record.finish();
        result.inspect_err(|_| {
            if let Err(error) = finished {
                log_error("finish device access after host-to-device copy", error);
            }
        })?;
        finished
    }

    /// Copies `source` into a new host vector, returning once the copy completed.
    pub fn clone_dtoh<T: DeviceRepr, Source: DevicePtr<T>>(
        self: &Arc<Self>,
        source: &Source,
    ) -> Result<Vec<T>, DriverError> {
        let mut target: Vec<T> = Vec::with_capacity(source.len());
        let (pointer, record) = source.device_ptr(self)?;
        let result = self.bind().and_then(|()| {
            // SAFETY: The vector has capacity for `source.len()` elements, and the wait below
            // completes the copy before the vector can be accessed or freed.
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    target.as_mut_ptr().cast(),
                    pointer,
                    source.num_bytes(),
                    self.cu_stream(),
                )
            }
            .result()
        });
        let finished = record.finish();
        let waited = if result.is_ok() && finished.is_ok() {
            // SAFETY: The copy and access recording left this stream's context current.
            unsafe { sys::cuStreamSynchronize(self.cu_stream()) }.result()
        } else {
            self.synchronize()
        };
        result.inspect_err(|_| {
            if let Err(error) = finished {
                log_error("finish device access after device-to-host copy", error);
            }
            if let Err(error) = waited {
                log_error("wait for device-to-host copy", error);
            }
        })?;
        finished.inspect_err(|_| {
            if let Err(error) = waited {
                log_error("wait for device-to-host copy", error);
            }
        })?;
        waited?;
        // SAFETY: The successful copy and wait initialized all `source.len()` elements.
        unsafe { target.set_len(source.len()) };
        Ok(target)
    }

    /// Copies `source` into `target`, returning once the copy completed.
    ///
    /// # Panics
    ///
    /// Panics if `target` is shorter than `source`.
    pub fn memcpy_dtoh<T: DeviceRepr, Source: DevicePtr<T>, Target: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        source: &Source,
        target: &mut Target,
    ) -> Result<(), DriverError> {
        // SAFETY: The slice is used only by the copy below, enqueued on this stream.
        let (target, mut record_target) = unsafe { target.stream_synced_mut_slice(self) }?;
        // The copied slice, not `HostSlice::len`, bounds the copy.
        assert!(target.len() >= source.len());
        let (pointer, record_source) = source.device_ptr(self)?;
        let result = self.bind().and_then(|()| {
            // SAFETY: The target holds at least `source.num_bytes()` bytes; the wait below
            // completes the copy before this returns.
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    target.as_mut_ptr().cast(),
                    pointer,
                    source.num_bytes(),
                    self.cu_stream(),
                )
            }
            .result()
        });
        let waited = if result.is_ok() {
            // SAFETY: The copy left this stream's context current.
            unsafe { sys::cuStreamSynchronize(self.cu_stream()) }.result()
        } else {
            self.synchronize()
        };
        if waited.is_ok() {
            // Avoid a duplicate same-stream wait, but preserve other waits and recording guards.
            record_target.disarm_wait_for(self);
        }
        let finished = SyncOnDrop::finish_all([record_source, record_target]);
        result.inspect_err(|_| {
            if let Err(error) = waited {
                log_error("wait for device-to-host copy", error);
            }
            if let Err(error) = finished {
                log_error("finish accesses after device-to-host copy", error);
            }
        })?;
        waited.inspect_err(|_| {
            if let Err(error) = finished {
                log_error("finish accesses after device-to-host copy", error);
            }
        })?;
        finished
    }

    /// Copies `source` into `target` on this stream.
    ///
    /// # Panics
    ///
    /// Panics if `target` is shorter than `source`.
    pub fn memcpy_dtod<T, Source: DevicePtr<T>, Target: DevicePtrMut<T>>(
        self: &Arc<Self>,
        source: &Source,
        target: &mut Target,
    ) -> Result<(), DriverError> {
        assert!(target.len() >= source.len());
        let bytes = source.num_bytes();
        let (source, record_source) = source.device_ptr(self)?;
        let (target, record_target) = target.device_ptr_mut(self)?;
        let result = self.bind().and_then(|()| {
            // SAFETY: Both ranges hold `bytes` bytes, ordered for this stream.
            unsafe { sys::cuMemcpyDtoDAsync_v2(target, source, bytes, self.cu_stream()) }.result()
        });
        let finished = SyncOnDrop::finish_all([record_source, record_target]);
        result.inspect_err(|_| {
            if let Err(error) = finished {
                log_error("finish accesses after device-to-device copy", error);
            }
        })?;
        finished
    }

    /// Copies `source` into a new allocation on this stream.
    pub fn clone_dtod<T: DeviceRepr, Source: DevicePtr<T>>(
        self: &Arc<Self>,
        source: &Source,
    ) -> Result<CudaSlice<T>, DriverError> {
        // SAFETY: Fully written by the copy below.
        let mut target = unsafe { self.alloc(source.len()) }?;
        self.memcpy_dtod(source, &mut target)?;
        Ok(target)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::tests::capture_logs;

    #[test]
    fn finish_returns_error_without_logging_or_retrying_on_drop() {
        let calls = Cell::new(0);
        let error = DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE);
        let mut cleanup = || {
            calls.set(calls.get() + 1);
            Err(error)
        };
        let guard = SyncOnDrop {
            pending: Pending::Cleanup(&mut cleanup),
        };
        let (result, logs) = capture_logs(|| guard.finish());
        assert_eq!(result, Err(error));
        assert_eq!(calls.get(), 1);
        assert!(logs.is_empty(), "{logs}");
    }

    #[test]
    fn finish_all_finishes_every_guard_and_logs_only_later_errors() {
        let first = DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE);
        let second = DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY);
        let calls = &Cell::new(0);
        let mut cleanups = [Ok(()), Err(first), Err(second), Err(second), Ok(())].map(|result| {
            move || {
                calls.set(calls.get() + 1);
                result
            }
        });
        let mut visited = 0;
        let guards = cleanups
            .iter_mut()
            .map(|cleanup| SyncOnDrop {
                pending: Pending::Cleanup(cleanup),
            })
            .inspect(|_| visited += 1);
        let (result, logs) = capture_logs(|| SyncOnDrop::finish_all(guards));
        assert_eq!(result, Err(first));
        assert_eq!(visited, 5);
        assert_eq!(calls.get(), 5);
        assert_eq!(logs.lines().count(), 2, "{logs}");
        for line in logs.lines() {
            assert!(line.contains("CUDA operation failed"), "{logs}");
            assert!(
                line.contains("operation=\"finish additional guard\""),
                "{logs}"
            );
            assert!(line.contains("error="), "{logs}");
            assert!(!line.contains("location="), "{logs}");
        }
    }

    #[test]
    fn finish_all_returns_single_error_without_logging() {
        let error = DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE);
        let mut cleanup = || Err(error);
        let (result, logs) = capture_logs(|| {
            SyncOnDrop::finish_all([
                SyncOnDrop::none(),
                SyncOnDrop {
                    pending: Pending::Cleanup(&mut cleanup),
                },
                SyncOnDrop::none(),
            ])
        });
        assert_eq!(result, Err(error));
        assert!(logs.is_empty(), "{logs}");
    }

    #[test]
    fn drop_logs_failed_cleanup_once() {
        let calls = Cell::new(0);
        let mut cleanup = || {
            calls.set(calls.get() + 1);
            Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))
        };
        let ((), logs) = capture_logs(|| {
            drop(SyncOnDrop {
                pending: Pending::Cleanup(&mut cleanup),
            });
        });
        assert_eq!(calls.get(), 1);
        assert_eq!(logs.lines().count(), 1, "{logs}");
        assert!(logs.contains("CUDA operation failed"), "{logs}");
        assert!(
            logs.contains("operation=\"finish guard on drop\""),
            "{logs}"
        );
        assert!(logs.contains("error="), "{logs}");
        assert!(!logs.contains("location="), "{logs}");
    }

    #[test]
    fn successful_guard_completion_does_not_log() -> Result<(), DriverError> {
        let (result, logs) = capture_logs(|| {
            let mut cleanup = || Ok(());
            drop(SyncOnDrop {
                pending: Pending::Cleanup(&mut cleanup),
            });
            SyncOnDrop {
                pending: Pending::Cleanup(&mut cleanup),
            }
            .finish()?;
            SyncOnDrop::finish_all([
                SyncOnDrop::none(),
                SyncOnDrop {
                    pending: Pending::Cleanup(&mut cleanup),
                },
            ])
        });
        result?;
        assert!(logs.is_empty(), "{logs}");
        Ok(())
    }

    // Run under Compute Sanitizer to also catch API errors from pageable-pointer probing.
    #[test]
    fn borrowed_host_slices_wait_only_for_pinned_memory() -> Result<(), DriverError> {
        // SAFETY: Only probes for the driver library.
        if !unsafe { sys::is_culib_present() } {
            return Ok(());
        }
        let Ok(context) = CudaContext::new(0) else {
            return Ok(());
        };
        let stream = context.new_stream()?;
        let pageable = [0u8; 16];
        let guard = borrowed(&pageable, &stream)?;
        assert!(matches!(guard.pending, Pending::None));
        guard.finish()?;

        // SAFETY: The context is current, and the allocation is freed below.
        let pointer = unsafe { malloc_host(16, 0) }?;
        // SAFETY: The allocation holds 16 writable bytes, initialized before creating the slice.
        let pinned = unsafe {
            pointer.cast::<u8>().write_bytes(0, 16);
            std::slice::from_raw_parts(pointer.cast::<u8>(), 16)
        };
        let guard = borrowed(pinned, &stream)?;
        let waits = matches!(guard.pending, Pending::Wait(waiting) if waiting.id() == stream.id());
        guard.finish()?;
        // SAFETY: No work or guard accesses the allocation anymore.
        unsafe { free_host(pointer) }?;
        assert!(waits);
        Ok(())
    }

    #[test]
    fn disarms_only_waits_for_the_synchronized_stream() -> Result<(), DriverError> {
        // SAFETY: Only probes for the driver library.
        if !unsafe { sys::is_culib_present() } {
            return Ok(());
        }
        let Ok(context) = CudaContext::new(0) else {
            return Ok(());
        };
        let stream = context.new_stream()?;
        let other = context.new_stream()?;
        let event = context.new_event(None)?;
        let mut wait = SyncOnDrop::wait_for(&stream);
        let mut other_wait = SyncOnDrop::wait_for(&other);
        let mut record = SyncOnDrop::record_event(&event, &stream);

        stream.synchronize()?;
        wait.disarm_wait_for(&stream);
        other_wait.disarm_wait_for(&stream);
        record.disarm_wait_for(&stream);
        assert!(matches!(wait.pending, Pending::None));
        assert!(matches!(other_wait.pending, Pending::Wait(_)));
        assert!(matches!(record.pending, Pending::Event(..)));
        SyncOnDrop::finish_all([wait, other_wait, record])
    }
}
