// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Allocator-backed storage for Vortex buffers.

use std::alloc::Layout;
use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;

use allocator_api2::alloc::AllocError;
use allocator_api2::alloc::Allocator;
use allocator_api2::alloc::Global;
use allocator_api2::alloc::handle_alloc_error;
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
/// The static allocator does not need shared ownership, so it is stored without an [`Arc`]. This
/// makes cloning the common static allocator a simple value copy.
#[derive(Clone)]
pub struct BufferAllocatorRef(
    // `None` selects the static allocator without allocating or updating an Arc reference count.
    // `Some` keeps a custom allocator alive for as long as its buffers need it.
    Option<Arc<dyn BufferAllocator>>,
);

impl BufferAllocatorRef {
    /// Wrap an allocator in a shared reference.
    pub fn new(allocator: impl BufferAllocator) -> Self {
        Self(Some(Arc::new(allocator)))
    }

    /// Return a shared reference to the static allocator.
    pub fn statically_allocated() -> Self {
        Self(None)
    }

    /// Return a borrowed reference to the static allocator.
    pub fn static_ref() -> &'static Self {
        &STATIC_ALLOCATOR
    }

    pub(crate) fn is_statically_allocated(&self) -> bool {
        self.0.is_none()
    }

    /// Returns true if both references point to the same allocator.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (None, None) => true,
            (Some(lhs), Some(rhs)) => Arc::ptr_eq(lhs, rhs),
            _ => false,
        }
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
    pub fn copy_from<T: Copy>(&self, values: impl AsRef<[T]>) -> BufferMut<T> {
        BufferMut::copy_from_in(values, self.clone())
    }
}

impl Debug for BufferAllocatorRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(allocator) => allocator.fmt(f),
            None => StaticBufferAllocator.fmt(f),
        }
    }
}

// SAFETY: all calls are forwarded to the same allocator value held by the Arc.
unsafe impl Allocator for BufferAllocatorRef {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match &self.0 {
            Some(allocator) => allocator.allocate(layout),
            None => Global.allocate(layout),
        }
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match &self.0 {
            Some(allocator) => allocator.allocate_zeroed(layout),
            None => Global.allocate_zeroed(layout),
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the caller upholds the Allocator contract.
        match &self.0 {
            Some(allocator) => unsafe { allocator.deallocate(ptr, layout) },
            None => unsafe { Global.deallocate(ptr, layout) },
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        match &self.0 {
            Some(allocator) => unsafe { allocator.grow(ptr, old_layout, new_layout) },
            None => unsafe { Global.grow(ptr, old_layout, new_layout) },
        }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        match &self.0 {
            Some(allocator) => unsafe { allocator.grow_zeroed(ptr, old_layout, new_layout) },
            None => unsafe { Global.grow_zeroed(ptr, old_layout, new_layout) },
        }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        match &self.0 {
            Some(allocator) => unsafe { allocator.shrink(ptr, old_layout, new_layout) },
            None => unsafe { Global.shrink(ptr, old_layout, new_layout) },
        }
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
    pub fn copy_from<T: Copy>(values: impl AsRef<[T]>) -> BufferMut<T> {
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

static STATIC_ALLOCATOR: BufferAllocatorRef = BufferAllocatorRef(None);

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
            allocator: BufferAllocatorRef::statically_allocated(),
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
            // the caller only grows the allocation.
            unsafe { self.allocator.grow(self.ptr, self.layout, new_layout) }
        }
        .unwrap_or_else(|_| handle_alloc_error(new_layout));
        self.ptr = allocation.cast();
        self.layout = new_layout;
    }

    /// Shrink the allocation to `new_size` bytes, keeping its layout alignment.
    ///
    /// The first `new_size` bytes are preserved, but the block may move. Returns `false` and
    /// leaves the allocation unchanged if the allocator cannot shrink it.
    pub(crate) fn shrink(&mut self, new_size: usize) -> bool {
        debug_assert!(new_size > 0 && new_size <= self.layout.size());
        let Ok(new_layout) = Layout::from_size_align(new_size, self.layout.align()) else {
            return false;
        };
        // SAFETY: ptr denotes a live block owned by allocator, old_layout fits the block, and the
        // new layout keeps the alignment and is no larger than the old one.
        match unsafe { self.allocator.shrink(self.ptr, self.layout, new_layout) } {
            Ok(allocation) => {
                self.ptr = allocation.cast();
                self.layout = new_layout;
                true
            }
            Err(_) => false,
        }
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

pub(crate) enum BufferBacking {
    Owned(Allocation),
    Bytes(bytes::Bytes),
    #[cfg(feature = "arrow")]
    Arrow(arrow_buffer::Buffer),
    External {
        _owner: Box<dyn Any + Send + Sync>,
    },
}

impl BufferBacking {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn allocator(&self) -> &BufferAllocatorRef {
        match self {
            Self::Owned(allocation) => allocation.allocator(),
            Self::Bytes(_) | Self::External { .. } => &STATIC_ALLOCATOR,
            #[cfg(feature = "arrow")]
            Self::Arrow(_) => &STATIC_ALLOCATOR,
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
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

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
        let static_allocator = BufferAllocatorRef::statically_allocated();
        assert!(static_allocator.ptr_eq(&BufferAllocatorRef::statically_allocated()));

        let custom_allocator = BufferAllocatorRef::new(TrackingAllocator::default());
        assert!(custom_allocator.ptr_eq(&custom_allocator.clone()));
        assert!(!custom_allocator.ptr_eq(&static_allocator));
        assert!(!custom_allocator.ptr_eq(&BufferAllocatorRef::new(TrackingAllocator::default())));
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
            Alignment::of::<u8>().as_usize()
        );
        drop(buffer);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 0);
        drop(view);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 1);
    }

    #[rstest]
    fn buffer_growth_uses_allocator_grow(#[values(4, 64, 4096)] alignment: usize) {
        let allocator = TrackingAllocator::default();
        let state = Arc::clone(&allocator.state);
        let alignment = Alignment::new(alignment);
        let mut buffer =
            BufferAllocatorRef::new(allocator).with_capacity_aligned::<u32>(1, alignment);
        let initial_capacity = buffer.capacity();
        buffer.extend(std::iter::repeat_n(7, initial_capacity));

        buffer.push(u32::MAX);
        assert!(alignment.is_ptr_aligned(buffer.as_ptr()));

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
        let mut buffer = BufferAllocatorRef::new(allocator).with_capacity::<u32>(0);

        assert_eq!(buffer.capacity(), 0);
        assert!(Alignment::DEFAULT_ALIGNMENT.is_offset_aligned(buffer.as_ptr().addr()));
        assert_eq!(state.allocations.load(Ordering::Relaxed), 0);

        buffer.push(42);

        assert_eq!(buffer.as_slice(), [42]);
        assert_eq!(state.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(state.grows.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn empty_buffers_preserve_allocator_without_allocating() -> VortexResult<()> {
        let allocator = TrackingAllocator::default();
        let state = Arc::clone(&allocator.state);
        let allocator = BufferAllocatorRef::new(allocator);
        let buffer = BufferMut::<u32>::zeroed_in(0, allocator.clone());
        let buffer = buffer.freeze();
        let copy = buffer.clone().into_mut();
        assert!(copy.allocation.allocator().ptr_eq(&allocator));
        let mut buffer = buffer
            .try_into_mut()
            .map_err(|_| vortex_err!("unique buffer"))?;
        buffer.reserve(0);
        assert!(buffer.is_empty());
        assert!(buffer.allocation.allocator().ptr_eq(&allocator));
        drop((copy, buffer));
        assert_eq!(state.allocations.load(Ordering::Relaxed), 0);
        assert_eq!(state.grows.load(Ordering::Relaxed), 0);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn shared_into_mut_preserves_allocator() {
        let allocator = TrackingAllocator::default();
        let state = Arc::clone(&allocator.state);
        let allocator = BufferAllocatorRef::new(allocator);
        let original = allocator.copy_from([1u32, 2, 3]).freeze();
        let mut copy = original.clone().into_mut();
        assert!(copy.allocation.allocator().ptr_eq(&allocator));
        copy[0] = 42;
        assert_eq!(original.as_slice(), [1, 2, 3]);
        assert_eq!(copy.as_slice(), [42, 2, 3]);
        assert_eq!(state.allocations.load(Ordering::Relaxed), 2);
        drop(copy);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 1);
        drop(original);
        assert_eq!(state.deallocations.load(Ordering::Relaxed), 2);
    }

    /// Forwards to [`Global`], but places every block at a varying offset past a 64 KiB boundary.
    /// Each shrink moves the block, so a buffer whose allocation is less aligned than the buffer
    /// itself lands at a different alignment offset after the shrink.
    #[derive(Debug, Default)]
    struct MovingAllocator {
        calls: AtomicUsize,
        shrinks: AtomicUsize,
    }

    impl MovingAllocator {
        const REAL_ALIGN: usize = 1 << 16;

        /// The real block behind `ptr`: the shift is below `REAL_ALIGN`, so rounding down finds it.
        fn real_block(ptr: NonNull<u8>, layout: Layout) -> (NonNull<u8>, Layout) {
            let shift = ptr.addr().get() % Self::REAL_ALIGN;
            // SAFETY: ptr was returned by allocate, which placed it shift bytes into a real block.
            let real = unsafe { ptr.sub(shift) };
            let real_layout = Layout::from_size_align(layout.size() + shift, Self::REAL_ALIGN)
                .unwrap_or_else(|_| unreachable!("allocate checked this layout"));
            (real, real_layout)
        }
    }

    // SAFETY: every block is a live Global block, offset by a multiple of the requested alignment,
    // and deallocation recovers the real block it came from.
    unsafe impl Allocator for MovingAllocator {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            let shift = layout.align() * (1 + call % 7);
            if shift >= Self::REAL_ALIGN {
                return Err(AllocError);
            }
            let real_layout = Layout::from_size_align(layout.size() + shift, Self::REAL_ALIGN)
                .map_err(|_| AllocError)?;
            let real = Global.allocate(real_layout)?.cast::<u8>();
            // SAFETY: real_layout reserves shift bytes before the returned block.
            let ptr = unsafe { real.add(shift) };
            Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            let (real, real_layout) = Self::real_block(ptr, layout);
            // SAFETY: real and real_layout describe the block from Global.allocate.
            unsafe { Global.deallocate(real, real_layout) }
        }

        unsafe fn shrink(
            &self,
            ptr: NonNull<u8>,
            old_layout: Layout,
            new_layout: Layout,
        ) -> Result<NonNull<[u8]>, AllocError> {
            self.shrinks.fetch_add(1, Ordering::Relaxed);
            let new = self.allocate(new_layout)?;
            // SAFETY: both blocks are live and disjoint, and new_layout is no larger than old.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr(),
                    new.cast::<u8>().as_ptr(),
                    new_layout.size(),
                );
                self.deallocate(ptr, old_layout);
            }
            Ok(new)
        }
    }

    fn assert_shrunk<T: Copy + PartialEq + std::fmt::Debug>(
        buffer: &BufferMut<T>,
        expected: &[T],
        alignment: Alignment,
    ) {
        assert_eq!(buffer.as_slice(), expected);
        assert!(alignment.is_ptr_aligned(buffer.as_ptr()));
        assert!(buffer.capacity() >= buffer.len());
        // The data keeps its offset of up to the over-alignment, plus padding to realign it.
        let max_bytes = buffer.len() * size_of::<T>()
            + alignment
                .as_usize()
                .max(Alignment::DEFAULT_ALIGNMENT.as_usize())
            + alignment.as_usize()
            - 1;
        assert!(buffer.allocation.size() <= max_bytes.max(1));
    }

    #[rstest]
    fn shrink_to_fit_realigns_moved_blocks(
        #[values(8, 64, 4096)] alignment: usize,
        #[values(1, 3, 1000)] len: u64,
    ) {
        let allocator = Arc::new(MovingAllocator::default());
        let alignment = Alignment::new(alignment);
        // Growing from empty makes an allocation that is only byte aligned.
        let mut buffer = BufferMut::<u64>::empty_aligned_in(
            alignment,
            BufferAllocatorRef::new(ArcAllocator(Arc::clone(&allocator))),
        );
        let expected: Vec<u64> = (0..len).collect();
        buffer.reserve(8192);
        buffer.extend_from_slice(&expected);
        assert!(buffer.allocation.alignment() < alignment.as_usize());

        buffer.shrink_to_fit();

        assert_shrunk(&buffer, &expected, alignment);
        assert_eq!(allocator.shrinks.load(Ordering::Relaxed), 1);
        buffer.push(u64::MAX);
        assert_eq!(buffer[expected.len()], u64::MAX);
        assert!(alignment.is_ptr_aligned(buffer.as_ptr()));
    }

    #[rstest]
    fn shrink_to_fit_with_global(
        #[values(2, 8, 64, 4096)] alignment: usize,
        #[values(0, 1, 1000)] len: u16,
    ) {
        let alignment = Alignment::new(alignment);
        let mut buffer = BufferMut::<u16>::with_capacity_aligned(1 << 16, alignment);
        let expected: Vec<u16> = (0..len).collect();
        buffer.extend_from_slice(&expected);

        buffer.shrink_to_fit();

        assert_shrunk(&buffer, &expected, alignment);
        assert_eq!(buffer.alignment(), alignment);
    }

    #[test]
    fn shrink_to_fit_drops_slice_prefix() -> VortexResult<()> {
        let allocator = Arc::new(MovingAllocator::default());
        let alignment = Alignment::new(64);
        let values: Vec<u64> = (0..1024).collect();
        let buffer = BufferMut::<u64>::copy_from_aligned_in(
            &values,
            alignment,
            BufferAllocatorRef::new(ArcAllocator(Arc::clone(&allocator))),
        )
        .freeze()
        .slice(512..520);
        let mut buffer = buffer
            .try_into_mut()
            .map_err(|_| vortex_err!("unique buffer"))?;

        buffer.shrink_to_fit();

        assert_shrunk(&buffer, &values[512..520], alignment);
        // The data sits after a prefix, so it is copied once into a new allocation.
        assert_eq!(allocator.shrinks.load(Ordering::Relaxed), 0);
        assert_eq!(allocator.calls.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[rstest]
    fn shrink_to_fit_is_idempotent(#[values(1, 64)] alignment: usize) {
        let allocator = Arc::new(MovingAllocator::default());
        let alignment = Alignment::new(alignment);
        let mut buffer = BufferAllocatorRef::new(ArcAllocator(Arc::clone(&allocator)))
            .with_capacity_aligned::<u8>(1024, alignment);
        buffer.extend_from_slice(&[1, 2, 3]);

        buffer.shrink_to_fit();
        let ptr = buffer.as_ptr();
        buffer.shrink_to_fit();

        assert_eq!(buffer.as_slice(), [1, 2, 3]);
        assert_eq!(buffer.as_ptr(), ptr);
        assert_eq!(allocator.shrinks.load(Ordering::Relaxed), 1);
    }

    /// Shares a [`MovingAllocator`] between the buffer and the test.
    #[derive(Debug)]
    struct ArcAllocator(Arc<MovingAllocator>);

    // SAFETY: forwards every call to the shared allocator.
    unsafe impl Allocator for ArcAllocator {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            self.0.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            // SAFETY: the caller upholds the Allocator contract.
            unsafe { self.0.deallocate(ptr, layout) }
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
}
