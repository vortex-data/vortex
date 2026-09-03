// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Allocator-backed storage for Vortex buffers.

use std::alloc::Layout;
use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use allocator_api2::alloc::AllocError;
use allocator_api2::alloc::Allocator;
use allocator_api2::alloc::Global;
use allocator_api2::alloc::handle_alloc_error;
use arcref::ArcRef;
use vortex_error::VortexExpect;

use crate::Alignment;
use crate::BufferMut;

/// An allocator that can back a Vortex buffer.
///
/// Vortex over-allocates raw storage and aligns the buffer within it.
pub trait BufferAllocator: Allocator + Debug + Send + Sync + 'static {}

impl<A> BufferAllocator for A where A: Allocator + Debug + Send + Sync + 'static {}

/// A shared reference to a buffer allocator.
///
/// Use [`ArcRef::new_ref`] for a static allocator or [`ArcRef::new_arc`] for an owned allocator.
pub type BufferAllocatorRef = ArcRef<dyn BufferAllocator>;

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

static STATIC_BUFFER_ALLOCATOR: StaticBufferAllocator = StaticBufferAllocator;
static GLOBAL_ALLOCATOR: Global = Global;
/// The allocator used by buffer APIs that do not take an allocator.
pub static DEFAULT_BUFFER_ALLOCATOR: BufferAllocatorRef = ArcRef::new_ref(&STATIC_BUFFER_ALLOCATOR);
pub(crate) static GLOBAL_ALLOCATOR_REF: BufferAllocatorRef = ArcRef::new_ref(&GLOBAL_ALLOCATOR);

pub(crate) struct Allocation {
    ptr: NonNull<u8>,
    layout: Layout,
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

    pub(crate) fn from_vec<T>(vec: Vec<T>) -> Self {
        assert!(!std::mem::needs_drop::<T>());

        let mut vec = ManuallyDrop::new(vec);
        let layout = Layout::array::<T>(vec.capacity())
            .unwrap_or_else(|_| unreachable!("a Vec capacity always has a valid layout"));
        let ptr = NonNull::new(vec.as_mut_ptr().cast())
            .vortex_expect("a Vec always has a non-null pointer");

        Self {
            ptr,
            layout,
            allocator: GLOBAL_ALLOCATOR_REF.clone(),
        }
    }

    fn allocate_impl(layout: Layout, allocator: BufferAllocatorRef, zeroed: bool) -> Self {
        if layout.size() == 0 {
            return Self {
                ptr: layout.dangling_ptr(),
                layout,
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
            allocator,
        }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn size(&self) -> usize {
        self.layout.size()
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn alignment(&self) -> usize {
        self.layout.align()
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn allocator(&self) -> &BufferAllocatorRef {
        &self.allocator
    }

    pub(crate) fn grow(&mut self, new_layout: Layout) {
        let allocation = if self.layout.size() == 0 {
            self.allocator.allocate(new_layout)
        } else {
            // SAFETY: ptr denotes a live block owned by allocator, old_layout fits the block, and
            // new_layout is larger. Allocator::grow permits a change in alignment.
            unsafe { self.allocator.grow(self.ptr, self.layout, new_layout) }
        }
        .unwrap_or_else(|_| handle_alloc_error(new_layout));
        self.ptr = allocation.cast();
        self.layout = new_layout;
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
    fn as_ptr(&self) -> *const u8;
}

impl<T> BufferOwner for T
where
    T: AsRef<[u8]> + Send + Sync + 'static,
{
    fn as_ptr(&self) -> *const u8 {
        self.as_ref().as_ptr()
    }
}

pub(crate) enum BufferBacking {
    Owned(Allocation),
    Bytes(bytes::Bytes),
    #[cfg(feature = "arrow")]
    Arrow(arrow_buffer::Buffer),
    External {
        _owner: Box<dyn BufferOwner>,
    },
}

impl BufferBacking {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn allocator(&self) -> &BufferAllocatorRef {
        match self {
            Self::Owned(allocation) => allocation.allocator(),
            Self::Bytes(_) | Self::External { .. } => &DEFAULT_BUFFER_ALLOCATOR,
            #[cfg(feature = "arrow")]
            Self::Arrow(_) => &DEFAULT_BUFFER_ALLOCATOR,
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

    use super::DEFAULT_BUFFER_ALLOCATOR;
    use super::GLOBAL_ALLOCATOR_REF;
    use crate::Alignment;
    use crate::BufferAllocatorRef;
    use crate::BufferMut;

    #[derive(Clone, Debug, Default)]
    struct TrackingAllocator {
        state: Arc<TrackingState>,
    }

    #[derive(Debug, Default)]
    struct TrackingState {
        allocations: AtomicUsize,
        deallocations: AtomicUsize,
        grows: AtomicUsize,
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

        unsafe fn grow(
            &self,
            ptr: NonNull<u8>,
            old_layout: Layout,
            new_layout: Layout,
        ) -> Result<NonNull<[u8]>, AllocError> {
            self.state.grows.fetch_add(1, Ordering::Relaxed);
            // SAFETY: the caller upholds the Allocator contract.
            unsafe { Global.grow(ptr, old_layout, new_layout) }
        }
    }

    #[test]
    fn allocator_identity() {
        let static_allocator = DEFAULT_BUFFER_ALLOCATOR.clone();
        assert!(std::ptr::eq(
            static_allocator.as_ref(),
            DEFAULT_BUFFER_ALLOCATOR.as_ref()
        ));

        let global_allocator = GLOBAL_ALLOCATOR_REF.clone();
        assert!(std::ptr::eq(
            global_allocator.as_ref(),
            GLOBAL_ALLOCATOR_REF.as_ref()
        ));
        assert!(!std::ptr::eq(
            global_allocator.as_ref(),
            static_allocator.as_ref()
        ));

        let custom_allocator = BufferAllocatorRef::new_arc(Arc::new(TrackingAllocator::default()));
        assert!(std::ptr::eq(
            custom_allocator.as_ref(),
            custom_allocator.clone().as_ref()
        ));
        assert!(!std::ptr::eq(
            custom_allocator.as_ref(),
            static_allocator.as_ref()
        ));
        let other = BufferAllocatorRef::new_arc(Arc::new(TrackingAllocator::default()));
        assert!(!std::ptr::eq(custom_allocator.as_ref(), other.as_ref()));
    }

    #[test]
    fn allocation_lives_until_last_view() {
        let allocator = TrackingAllocator::default();
        let state = Arc::clone(&allocator.state);
        let allocator = BufferAllocatorRef::new_arc(Arc::new(allocator));
        let buffer = BufferMut::copy_from_in([1u32, 2, 3, 4], allocator).freeze();
        let view = buffer.slice(0..2);

        assert_eq!(state.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(
            state.alignment.load(Ordering::Relaxed),
            Alignment::of::<u8>().as_usize()
        );
        drop(buffer);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 0);
        drop(view);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn buffer_growth_uses_allocator_grow() {
        let allocator = TrackingAllocator::default();
        let state = Arc::clone(&allocator.state);
        let allocator = BufferAllocatorRef::new_arc(Arc::new(allocator));
        let mut buffer = BufferMut::<u32>::with_capacity_in(1, allocator);
        let initial_capacity = buffer.capacity();
        buffer.extend(std::iter::repeat_n(7, initial_capacity));

        buffer.push(u32::MAX);

        assert_eq!(&buffer[..initial_capacity], vec![7; initial_capacity]);
        assert_eq!(buffer[initial_capacity], u32::MAX);
        assert_eq!(state.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 0);
        assert_eq!(state.grows.load(Ordering::Relaxed), 1);

        drop(buffer);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn zero_capacity_does_not_allocate() {
        let allocator = TrackingAllocator::default();
        let state = Arc::clone(&allocator.state);
        let allocator = BufferAllocatorRef::new_arc(Arc::new(allocator));
        let mut buffer = BufferMut::<u32>::with_capacity_in(0, allocator);

        assert_eq!(buffer.capacity(), 0);
        assert!(Alignment::of::<u32>().is_offset_aligned(buffer.as_ptr().addr()));
        assert_eq!(state.allocations.load(Ordering::Relaxed), 0);

        buffer.push(42);

        assert_eq!(buffer.as_slice(), [42]);
        assert_eq!(state.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(state.grows.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn zero_sized_buffers_do_not_call_the_allocator() {
        let allocator = TrackingAllocator::default();
        let state = Arc::clone(&allocator.state);
        let allocator = BufferAllocatorRef::new_arc(Arc::new(allocator));

        let mut buffer = BufferMut::<()>::with_capacity_in(usize::MAX, allocator.clone());
        buffer.extend([(); 4]);
        let buffer = buffer.freeze();
        assert_eq!(buffer.len(), 4);
        assert!(std::ptr::eq(
            buffer.allocator().as_ref(),
            allocator.as_ref()
        ));
        drop(buffer);

        let buffer = BufferMut::<()>::zeroed_in(4, allocator);
        assert_eq!(buffer.len(), 4);
        drop(buffer);

        assert_eq!(state.allocations.load(Ordering::Relaxed), 0);
        assert_eq!(state.grows.load(Ordering::Relaxed), 0);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 0);
    }
}
