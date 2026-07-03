// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped memory allocation for host-side buffers.

use std::any::Any;
use std::fmt::Debug;
use std::mem::size_of;
use std::sync::Arc;

use bytes::Bytes;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::VortexSession;

/// Mutable host buffer contract used by [`WritableHostBuffer`].
pub trait HostBufferMut: Send + 'static {
    /// Returns the logical byte length of the buffer.
    fn len(&self) -> usize;

    /// Whether the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alignment of the buffer.
    fn alignment(&self) -> Alignment;

    /// Returns mutable access to the writable byte range.
    fn as_mut_slice(&mut self) -> &mut [u8];

    /// Freeze the buffer into an immutable [`ByteBuffer`].
    fn freeze(self: Box<Self>) -> ByteBuffer;
}

/// Exact-size writable host buffer returned by a [`HostAllocator`].
pub struct WritableHostBuffer {
    inner: Box<dyn HostBufferMut>,
}

impl WritableHostBuffer {
    /// Create a writable host buffer from an implementation of [`HostBufferMut`].
    pub fn new(inner: Box<dyn HostBufferMut>) -> Self {
        Self { inner }
    }

    /// Returns the logical byte length of the buffer.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true when the buffer has zero bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alignment of the buffer.
    pub fn alignment(&self) -> Alignment {
        self.inner.alignment()
    }

    /// Returns mutable access to the writable byte range.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner.as_mut_slice()
    }

    /// Returns mutable access to the buffer as a typed slice.
    pub fn as_mut_slice_typed<T>(&mut self) -> VortexResult<&mut [T]> {
        vortex_ensure!(
            size_of::<T>() != 0,
            InvalidArgument: "Cannot create typed mutable slice for zero-sized type {}",
            std::any::type_name::<T>()
        );
        vortex_ensure!(
            self.alignment().is_aligned_to(Alignment::of::<T>()),
            InvalidArgument: "Buffer is not sufficiently aligned for type {}",
            std::any::type_name::<T>()
        );

        let bytes = self.as_mut_slice();
        let byte_len = bytes.len();
        let ptr = bytes.as_mut_ptr();
        let type_size = size_of::<T>();

        vortex_ensure!(
            byte_len.is_multiple_of(type_size),
            InvalidArgument: "Buffer length {byte_len} is not a multiple of {} for {}",
            type_size,
            std::any::type_name::<T>()
        );

        // SAFETY: We checked size divisibility and pointer alignment for `T`,
        // and we have exclusive mutable access to the underlying bytes.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr.cast::<T>(), byte_len / type_size) })
    }

    /// Freeze the writable buffer into an immutable [`ByteBuffer`].
    pub fn freeze(self) -> ByteBuffer {
        self.inner.freeze()
    }

    /// Freeze the writable buffer into a typed immutable [`Buffer<T>`].
    pub fn freeze_typed<T>(self) -> VortexResult<Buffer<T>> {
        vortex_ensure!(
            size_of::<T>() != 0,
            InvalidArgument: "Cannot freeze typed buffer for zero-sized type {}",
            std::any::type_name::<T>()
        );

        let buffer = self.freeze();
        let byte_len = buffer.len();
        let type_size = size_of::<T>();
        let type_align = Alignment::of::<T>();

        vortex_ensure!(
            byte_len.is_multiple_of(type_size),
            InvalidArgument: "Buffer length {byte_len} is not a multiple of {} for {}",
            type_size,
            std::any::type_name::<T>()
        );
        vortex_ensure!(
            buffer.is_aligned(type_align),
            InvalidArgument: "Buffer pointer is not aligned to {} for {}",
            type_align,
            std::any::type_name::<T>()
        );

        Ok(Buffer::from_byte_buffer(buffer))
    }
}

impl Debug for WritableHostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WritableHostBuffer")
            .field("len", &self.len())
            .field("alignment", &self.alignment())
            .finish()
    }
}

/// Allocator for exact-size writable host buffers.
pub trait HostAllocator: Debug + Send + Sync + 'static {
    /// Allocate a writable host buffer with the requested byte length and alignment.
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer>;
}

/// Shared allocator reference used throughout session-scoped memory APIs.
pub type HostAllocatorRef = Arc<dyn HostAllocator>;

/// Extension methods for [`HostAllocator`]s.
pub trait HostAllocatorExt: HostAllocator {
    /// Allocate host memory for `len` elements of `T` using `Alignment::of::<T>()`.
    fn allocate_typed<T>(&self, len: usize) -> VortexResult<WritableHostBuffer> {
        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            vortex_err!(
                "Typed host allocation overflow for type {} and len {}",
                std::any::type_name::<T>(),
                len
            )
        })?;
        self.allocate(bytes, Alignment::of::<T>())
    }
}

impl<A: HostAllocator + ?Sized> HostAllocatorExt for A {}

/// Session-scoped memory configuration for Vortex arrays.
#[derive(Clone, Debug)]
pub struct MemorySession {
    allocator: HostAllocatorRef,
}

impl MemorySession {
    /// Creates a new session memory configuration using the provided allocator.
    pub fn new(allocator: HostAllocatorRef) -> Self {
        Self { allocator }
    }

    /// Returns the configured allocator.
    pub fn allocator(&self) -> HostAllocatorRef {
        Arc::clone(&self.allocator)
    }

    /// Updates the configured allocator.
    pub fn set_allocator(&mut self, allocator: HostAllocatorRef) {
        self.allocator = allocator;
    }
}

impl Default for MemorySession {
    fn default() -> Self {
        Self::new(Arc::new(DefaultHostAllocator))
    }
}

impl SessionVar for MemorySession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing session-scoped memory configuration.
pub trait MemorySessionExt: SessionExt {
    /// Returns the memory session for this execution/session context.
    fn memory(&self) -> SessionGuard<'_, MemorySession> {
        self.get::<MemorySession>()
    }

    /// Returns the configured host allocator for this execution/session context.
    fn allocator(&self) -> HostAllocatorRef {
        self.memory().allocator()
    }

    /// Configures the session to use `allocator` as its host allocator, mutating it in place and
    /// returning it for chaining.
    fn with_allocator(self, allocator: HostAllocatorRef) -> VortexSession {
        let session = self.session();
        session.get_mut::<MemorySession>().set_allocator(allocator);
        session
    }
}

impl<S: SessionExt> MemorySessionExt for S {}

/// Default host allocator.
#[derive(Debug, Default)]
pub struct DefaultHostAllocator;

impl HostAllocator for DefaultHostAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
        // SAFETY: We fully initialize this slice before freezing it.
        unsafe { buffer.set_len(len) };
        Ok(WritableHostBuffer::new(Box::new(
            DefaultWritableHostBuffer { buffer, alignment },
        )))
    }
}

#[derive(Debug)]
struct DefaultWritableHostBuffer {
    buffer: ByteBufferMut,
    alignment: Alignment,
}

#[derive(Debug)]
struct HostBufferOwner {
    buffer: ByteBufferMut,
}

impl AsRef<[u8]> for HostBufferOwner {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

impl HostBufferMut for DefaultWritableHostBuffer {
    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn alignment(&self) -> Alignment {
        self.alignment
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }

    fn freeze(self: Box<Self>) -> ByteBuffer {
        let Self { buffer, alignment } = *self;
        let bytes = Bytes::from_owner(HostBufferOwner { buffer });
        ByteBuffer::from_bytes_aligned(bytes, alignment)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use super::*;

    #[derive(Debug)]
    struct CountingAllocator {
        allocations: Arc<AtomicUsize>,
    }

    impl HostAllocator for CountingAllocator {
        fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            DefaultHostAllocator.allocate(len, alignment)
        }
    }

    #[test]
    fn writable_host_buffer_freeze_round_trip() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate(16, Alignment::new(8)).unwrap();
        for (idx, byte) in writable.as_mut_slice().iter_mut().enumerate() {
            *byte = u8::try_from(idx).unwrap();
        }

        let host = writable.freeze();
        assert_eq!(host.len(), 16);
        assert!(host.is_aligned(Alignment::new(8)));
        assert_eq!(host.as_slice(), (0u8..16).collect::<Vec<_>>().as_slice());
    }

    #[test]
    fn memory_session_replaces_allocator() {
        let allocations = Arc::new(AtomicUsize::new(0));
        let allocator = Arc::new(CountingAllocator {
            allocations: Arc::clone(&allocations),
        });
        let mut session = MemorySession::default();
        session.set_allocator(allocator);
        drop(session.allocator().allocate(4, Alignment::none()).unwrap());
        assert_eq!(allocations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn typed_allocation_uses_type_alignment() {
        let allocator = DefaultHostAllocator;
        let writable = allocator.allocate_typed::<u64>(4).unwrap();
        assert_eq!(writable.len(), 4 * size_of::<u64>());
        assert_eq!(writable.alignment(), Alignment::of::<u64>());
    }

    #[test]
    fn typed_mut_slice_round_trip() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate_typed::<u64>(4).unwrap();
        writable
            .as_mut_slice_typed::<u64>()
            .unwrap()
            .copy_from_slice(&[10, 20, 30, 40]);

        let frozen = writable.freeze();
        let values = unsafe {
            std::slice::from_raw_parts(
                frozen.as_slice().as_ptr().cast::<u64>(),
                frozen.len() / size_of::<u64>(),
            )
        };
        assert_eq!(values, [10, 20, 30, 40]);
    }

    #[test]
    fn typed_mut_slice_rejects_length_mismatch() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate(7, Alignment::none()).unwrap();
        assert!(writable.as_mut_slice_typed::<u32>().is_err());
    }

    #[test]
    fn freeze_typed_round_trip() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate_typed::<u64>(4).unwrap();
        writable
            .as_mut_slice_typed::<u64>()
            .unwrap()
            .copy_from_slice(&[1, 3, 5, 7]);

        let frozen = writable.freeze_typed::<u64>().unwrap();
        assert_eq!(frozen.as_slice(), [1, 3, 5, 7]);
    }

    #[test]
    fn freeze_typed_rejects_length_mismatch() {
        let allocator = DefaultHostAllocator;
        let writable = allocator.allocate(7, Alignment::none()).unwrap();
        let err = writable.freeze_typed::<u32>().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not a multiple of"));
    }
}
