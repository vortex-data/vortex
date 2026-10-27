// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Device allocations, their views, and stream ordering between their accesses.
//!
//! Every allocation tracks which streams accessed it, so work on one stream never uses memory
//! before another stream's earlier write to it completes, and memory is never freed or
//! rewritten before other streams' earlier accesses complete. Unlike tracking every access
//! with events, only accesses that cross streams cost driver calls:
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
//! returned [`SyncOnDrop`] alive until the work using it is enqueued.

#![expect(
    clippy::same_name_method,
    reason = "inherent accessors mirror the DeviceSlice ones so callers need not import the trait"
)]
#![expect(
    clippy::expect_used,
    reason = "out-of-range views and overflowing sizes are caller bugs, as with std slices"
)]

use std::marker::PhantomData;
use std::ops::Bound;
use std::ops::RangeBounds;
use std::sync::Arc;

use parking_lot::Mutex;
use smallvec::SmallVec;

use crate::context::CudaContext;
use crate::repr::DeviceRepr;
use crate::repr::ValidAsZeroBits;
use crate::result::DriverError;
use crate::stream::CudaEvent;
use crate::stream::CudaStream;
use crate::stream::StreamId;
use crate::sys;

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
        stream.make_current()?;
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
        stream.make_current()?;
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
        if recorded.is_err() {
            // If this fails too, the context is broken and its work does not complete.
            stream.context().record_err(stream.synchronize_in_place());
        }
        recorded
    }

    /// Makes the allocation stream wait for every other stream's access before a free.
    fn before_free(&self, owner: &CudaStream) -> Result<(), DriverError> {
        owner.make_current()?;
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

/// Records an access when dropped, after the work using its pointer was enqueued.
#[must_use]
pub struct SyncOnDrop<'a> {
    pending: Pending<'a>,
}

enum Pending<'a> {
    None,
    Access(&'a Access, &'a CudaStream, &'a CudaStream, AccessKind),
    Event(&'a CudaEvent, &'a CudaStream),
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

    /// A guard that records `stream`'s work into `event` when dropped.
    pub fn record_event(event: &'a CudaEvent, stream: &'a CudaStream) -> Self {
        Self {
            pending: Pending::Event(event, stream),
        }
    }
}

impl Drop for SyncOnDrop<'_> {
    fn drop(&mut self) {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::None => {}
            Pending::Access(access, owner, stream, kind) => {
                stream
                    .context()
                    .record_err(access.after_or_wait(owner, stream, kind));
            }
            Pending::Event(event, stream) => {
                stream.context().record_err(
                    stream
                        .make_current()
                        .and_then(|()| stream.record_raw(event.cu_event())),
                );
            }
        }
    }
}

impl std::fmt::Debug for SyncOnDrop<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pending = match &self.pending {
            Pending::None => "none",
            Pending::Access(..) => "access",
            Pending::Event(..) => "event",
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
        match self.access.before_free(&self.stream) {
            Ok(()) => context.record_err(self.stream.free_raw(self.pointer)),
            // Other streams may still access the memory without the waits: leak it instead.
            Err(error) => context.record_err::<()>(Err(error)),
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

    /// A view of the elements in `bounds`, if in range.
    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<CudaView<'_, T>> {
        self.as_view().try_slice(bounds)
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
    /// The bytes must be valid values of `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'_, S>> {
        (len * size_of::<S>() <= self.num_bytes()).then_some(CudaView {
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
    /// The bytes must be valid values of `S`, and `T` must accept any bytes written as `S`.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'_, S>> {
        (len * size_of::<S>() <= self.num_bytes()).then_some(CudaViewMut {
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
    /// The bytes must be valid values of `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'a, S>> {
        (len * size_of::<S>() <= self.num_bytes()).then_some(CudaView {
            pointer: self.pointer,
            len,
            access: self.access,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// Like [`DevicePtr::device_ptr`], with the guard living as long as the view.
    pub fn view_ptr(self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        let guard = access(self.access, self.stream, stream, AccessKind::Read);
        (self.pointer, guard)
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

    /// Converts into a shared view with the original lifetime.
    pub fn into_view(self) -> CudaView<'a, T> {
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
        let (start, end) = range(&bounds, self.len).expect("CUDA view range out of bounds");
        CudaViewMut {
            pointer: self.pointer + (start * size_of::<T>()) as u64,
            len: end - start,
            access: self.access,
            stream: self.stream,
            marker: PhantomData,
        }
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
    /// The bytes must be valid values of `S`, and `T` must accept any bytes written as `S`.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'_, S>> {
        (len * size_of::<S>() <= self.num_bytes()).then_some(CudaViewMut {
            pointer: self.pointer,
            len,
            access: self.access,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// Like [`DevicePtrMut::device_ptr_mut`], with the guard living as long as the view.
    pub fn view_ptr_mut(self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        let guard = access(self.access, self.stream, stream, AccessKind::Write);
        (self.pointer, guard)
    }
}

/// Orders `stream` after earlier conflicting accesses and returns the guard recording this one.
///
/// Callers enqueue work with the pointer even if ordering it fails, so a failure falls back to
/// completing all work enqueued in the context so far, which the access then follows too.
fn access<'a>(
    access: &'a Access,
    owner: &'a CudaStream,
    stream: &'a CudaStream,
    kind: AccessKind,
) -> SyncOnDrop<'a> {
    if let Err(error) = access.before(owner, stream, kind) {
        let context = stream.context();
        context.record_err::<()>(Err(error));
        // If this fails too, the context is broken and rejects the access's work as well.
        context.record_err(context.synchronize_in_place());
    }
    SyncOnDrop::new(access, owner, stream, kind)
}

/// Device memory with a length and an allocation stream.
pub trait DeviceSlice<T> {
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
    /// Drop the guard after the read is enqueued.
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>);
}

/// Device memory that can be written on a stream.
pub trait DevicePtrMut<T>: DeviceSlice<T> {
    /// The device pointer for writing on `stream`, which first waits for earlier accesses.
    ///
    /// Drop the guard after the write is enqueued.
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>);
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
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        let guard = access(&self.access, &self.stream, stream, AccessKind::Read);
        (self.pointer, guard)
    }
}

impl<T> DevicePtr<T> for CudaView<'_, T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        let guard = access(self.access, self.stream, stream, AccessKind::Read);
        (self.pointer, guard)
    }
}

impl<T> DevicePtr<T> for CudaViewMut<'_, T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        let guard = access(self.access, self.stream, stream, AccessKind::Read);
        (self.pointer, guard)
    }
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        let guard = access(&self.access, &self.stream, stream, AccessKind::Write);
        (self.pointer, guard)
    }
}

impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        let guard = access(self.access, self.stream, stream, AccessKind::Write);
        (self.pointer, guard)
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

    /// The elements for a copy on `stream`, with a guard to drop after the copy is enqueued.
    ///
    /// # Safety
    ///
    /// The slice may be used only by work enqueued on `stream` before the guard drops.
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>);

    /// The elements for a copy into them on `stream`, with a guard as above.
    ///
    /// # Safety
    ///
    /// As for [`stream_synced_slice`](Self::stream_synced_slice).
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>);
}

impl<T> HostSlice<T> for [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::none())
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::none())
    }
}

impl<T, const N: usize> HostSlice<T> for [T; N] {
    fn len(&self) -> usize {
        N
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::none())
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::none())
    }
}

impl<T> HostSlice<T> for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::none())
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::none())
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
        let (pointer, _record) = target.device_ptr_mut(self);
        self.memset_d8_raw(pointer, 0, bytes)
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
    /// Pageable sources are staged by the driver before this returns; page-locked sources must
    /// stay alive and unmodified until the copy completes.
    ///
    /// # Panics
    ///
    /// Panics if `target` is shorter than `source`.
    pub fn memcpy_htod<T: DeviceRepr, Source: HostSlice<T> + ?Sized, Target: DevicePtrMut<T>>(
        self: &Arc<Self>,
        source: &Source,
        target: &mut Target,
    ) -> Result<(), DriverError> {
        assert!(target.len() >= source.len());
        // SAFETY: The slice is used only by the copy below, enqueued on this stream.
        let (source, _record_source) = unsafe { source.stream_synced_slice(self) };
        let (pointer, _record) = target.device_ptr_mut(self);
        self.bind()?;
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
    }

    /// Copies `source` into a new host vector, returning once the copy completed.
    pub fn clone_dtoh<T: DeviceRepr, Source: DevicePtr<T>>(
        self: &Arc<Self>,
        source: &Source,
    ) -> Result<Vec<T>, DriverError> {
        let mut target: Vec<T> = Vec::with_capacity(source.len());
        let (pointer, _record) = source.device_ptr(self);
        self.bind()?;
        // SAFETY: The vector has capacity for `source.len()` elements. A copy into pageable
        // memory returns only after it completes, which the synchronization below also
        // guarantees for page-locked memory.
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                target.as_mut_ptr().cast(),
                pointer,
                source.num_bytes(),
                self.cu_stream(),
            )
            .result()?;
            sys::cuStreamSynchronize(self.cu_stream()).result()?;
            target.set_len(source.len());
        }
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
        assert!(target.len() >= source.len());
        // SAFETY: The slice is used only by the copy below, enqueued on this stream.
        let (target, _record_target) = unsafe { target.stream_synced_mut_slice(self) };
        let (pointer, _record) = source.device_ptr(self);
        self.bind()?;
        // SAFETY: The target holds at least `source.num_bytes()` bytes; the copy completes
        // before this returns.
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                target.as_mut_ptr().cast(),
                pointer,
                source.num_bytes(),
                self.cu_stream(),
            )
            .result()?;
            sys::cuStreamSynchronize(self.cu_stream()).result()
        }
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
        let (source, _record_source) = source.device_ptr(self);
        let (target, _record_target) = target.device_ptr_mut(self);
        self.bind()?;
        // SAFETY: Both ranges hold `bytes` bytes, ordered for this stream.
        unsafe { sys::cuMemcpyDtoDAsync_v2(target, source, bytes, self.cu_stream()) }.result()
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
