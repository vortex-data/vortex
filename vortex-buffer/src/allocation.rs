// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Allocator-backed storage for Vortex buffers.

use std::alloc::Layout;
use std::fmt;
use std::fmt::Debug;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::LazyLock;

use allocator_api2::alloc::AllocError;
use allocator_api2::alloc::Allocator;
use allocator_api2::alloc::Global;
use allocator_api2::alloc::handle_alloc_error;

use crate::Alignment;
use crate::BufferMut;

/// An allocator that can back a Vortex buffer.
pub trait BufferAllocator: Allocator + Debug + Send + Sync + 'static {}

impl<A> BufferAllocator for A where A: Allocator + Debug + Send + Sync + 'static {}

/// A shared reference to a buffer allocator.
#[derive(Clone)]
pub struct BufferAllocatorRef(Arc<dyn BufferAllocator>);

impl BufferAllocatorRef {
    /// Wrap an allocator in a shared reference.
    pub fn new(allocator: impl BufferAllocator) -> Self {
        Self(Arc::new(allocator))
    }

    /// Return a shared reference to the static allocator.
    pub fn statically_allocated() -> Self {
        STATIC_ALLOCATOR.clone()
    }

    /// Create a mutable buffer with this allocator.
    pub fn with_capacity<T>(&self, capacity: usize) -> BufferMut<T> {
        BufferMut::with_capacity_in(capacity, self.clone())
    }

    /// Create an aligned mutable buffer with this allocator.
    pub fn with_capacity_aligned<T>(&self, capacity: usize, alignment: Alignment) -> BufferMut<T> {
        BufferMut::with_capacity_aligned_in(capacity, alignment, self.clone())
    }

    /// Create a zeroed mutable buffer with this allocator.
    pub fn zeroed<T>(&self, len: usize) -> BufferMut<T> {
        BufferMut::zeroed_in(len, self.clone())
    }

    /// Copy values into a mutable buffer made by this allocator.
    pub fn copy_from<T>(&self, values: impl AsRef<[T]>) -> BufferMut<T> {
        BufferMut::copy_from_in(values, self.clone())
    }
}

impl Debug for BufferAllocatorRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// SAFETY: all calls are forwarded to the same allocator value held by the Arc.
unsafe impl Allocator for BufferAllocatorRef {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.allocate_zeroed(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { self.0.deallocate(ptr, layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { self.0.grow(ptr, old_layout, new_layout) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { self.0.grow_zeroed(ptr, old_layout, new_layout) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { self.0.shrink(ptr, old_layout, new_layout) }
    }
}

/// The allocator used by buffer APIs that do not take an allocator.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticBufferAllocator;

impl StaticBufferAllocator {
    /// Create a mutable buffer with the static allocator.
    pub fn with_capacity<T>(capacity: usize) -> BufferMut<T> {
        BufferMut::with_capacity(capacity)
    }

    /// Create an aligned mutable buffer with the static allocator.
    pub fn with_capacity_aligned<T>(capacity: usize, alignment: Alignment) -> BufferMut<T> {
        BufferMut::with_capacity_aligned(capacity, alignment)
    }

    /// Create a zeroed mutable buffer with the static allocator.
    pub fn zeroed<T>(len: usize) -> BufferMut<T> {
        BufferMut::zeroed(len)
    }

    /// Copy values into a mutable buffer made by the static allocator.
    pub fn copy_from<T>(values: impl AsRef<[T]>) -> BufferMut<T> {
        BufferMut::copy_from(values)
    }
}

// SAFETY: Global satisfies the Allocator contract and this type only forwards to it.
unsafe impl Allocator for StaticBufferAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate_zeroed(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { Global.deallocate(ptr, layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { Global.grow(ptr, old_layout, new_layout) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { Global.grow_zeroed(ptr, old_layout, new_layout) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { Global.shrink(ptr, old_layout, new_layout) }
    }
}

static STATIC_ALLOCATOR: LazyLock<BufferAllocatorRef> =
    LazyLock::new(|| BufferAllocatorRef::new(StaticBufferAllocator));

pub(crate) struct Allocation {
    ptr: NonNull<u8>,
    layout: Layout,
    capacity: usize,
    allocator: BufferAllocatorRef,
}

// SAFETY: Allocation owns its memory, and its allocator is Send + Sync.
unsafe impl Send for Allocation {}
// SAFETY: shared access to Allocation never permits mutation of the allocation.
unsafe impl Sync for Allocation {}

impl Allocation {
    pub(crate) fn allocate(layout: Layout, allocator: BufferAllocatorRef) -> Self {
        Self::allocate_impl(layout, allocator, false)
    }

    pub(crate) fn allocate_zeroed(layout: Layout, allocator: BufferAllocatorRef) -> Self {
        Self::allocate_impl(layout, allocator, true)
    }

    fn allocate_impl(layout: Layout, allocator: BufferAllocatorRef, zeroed: bool) -> Self {
        if layout.size() == 0 {
            return Self {
                ptr: layout.dangling_ptr(),
                layout,
                capacity: 0,
                allocator,
            };
        }

        let allocation = if zeroed {
            allocator.allocate_zeroed(layout)
        } else {
            allocator.allocate(layout)
        }
        .unwrap_or_else(|_| handle_alloc_error(layout));

        Self {
            ptr: allocation.cast(),
            layout,
            capacity: allocation.len(),
            allocator,
        }
    }

    pub(crate) fn ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    pub(crate) fn size(&self) -> usize {
        self.capacity
    }

    pub(crate) fn alignment(&self) -> Alignment {
        Alignment::new(self.layout.align())
    }

    pub(crate) fn allocator(&self) -> &BufferAllocatorRef {
        &self.allocator
    }

    pub(crate) fn grow(&mut self, layout: Layout) {
        if self.layout.size() == 0 {
            *self = Self::allocate(layout, self.allocator.clone());
            return;
        }

        let allocation =
            // SAFETY: ptr and layout describe a live block allocated by self.allocator. The new
            // layout is at least as large as the old layout.
            unsafe { self.allocator.grow(self.ptr, self.layout, layout) }
                .unwrap_or_else(|_| handle_alloc_error(layout));
        self.ptr = allocation.cast();
        self.layout = layout;
        self.capacity = allocation.len();
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if self.layout.size() == 0 {
            return;
        }
        // SAFETY: ptr and layout describe a live block allocated by self.allocator.
        unsafe { self.allocator.deallocate(self.ptr, self.layout) }
    }
}

pub(crate) trait BufferOwner: Send + Sync + 'static {
    fn as_slice(&self) -> &[u8];
}

impl<T> BufferOwner for T
where
    T: AsRef<[u8]> + Send + Sync + 'static,
{
    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

pub(crate) enum BufferBacking {
    Owned(Allocation),
    External { _owner: Box<dyn BufferOwner> },
}

impl BufferBacking {
    pub(crate) fn allocator(&self) -> &BufferAllocatorRef {
        match self {
            Self::Owned(allocation) => allocation.allocator(),
            Self::External { .. } => LazyLock::force(&STATIC_ALLOCATOR),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;
    use std::ptr::NonNull;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use allocator_api2::alloc::AllocError;
    use allocator_api2::alloc::Allocator;
    use allocator_api2::alloc::Global;

    use crate::Alignment;
    use crate::BufferAllocatorRef;

    #[derive(Clone, Debug, Default)]
    struct TrackingAllocator {
        state: Arc<TrackingState>,
    }

    #[derive(Debug, Default)]
    struct TrackingState {
        allocations: AtomicUsize,
        deallocations: AtomicUsize,
        alignment: AtomicUsize,
    }

    // SAFETY: this forwards all memory operations to Global and only records call metadata.
    unsafe impl Allocator for TrackingAllocator {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            self.state.allocations.fetch_add(1, Ordering::Relaxed);
            self.state
                .alignment
                .store(layout.align(), Ordering::Relaxed);
            Global.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            self.state.deallocations.fetch_add(1, Ordering::Relaxed);
            // SAFETY: the caller passes the pointer and layout returned by Global.
            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    #[test]
    fn allocation_lives_until_last_view() {
        let allocator = TrackingAllocator::default();
        let state = Arc::clone(&allocator.state);
        let buffer = BufferAllocatorRef::new(allocator)
            .copy_from([1u32, 2, 3, 4])
            .freeze();
        let view = buffer.slice(0..2);

        assert_eq!(state.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(
            state.alignment.load(Ordering::Relaxed),
            *Alignment::DEFAULT_ALIGNMENT
        );
        drop(buffer);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 0);
        drop(view);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 1);
    }
}
