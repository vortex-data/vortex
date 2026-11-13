// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Account Vortex buffer memory in a DataFusion [`MemoryPool`].
//!
//! [`MemoryPoolAllocator`] is a Vortex buffer allocator that reports every allocation it makes to
//! a DataFusion memory pool. Install it on the Vortex session that scans run with, and the pool
//! sees the memory held by decoded Vortex buffers alongside the memory of DataFusion's own
//! operators:
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use datafusion_execution::memory_pool::GreedyMemoryPool;
//! use datafusion_execution::memory_pool::MemoryPool;
//! use vortex::VortexSessionDefault;
//! use vortex::array::memory::MemorySessionExt;
//! use vortex::buffer::BufferAllocatorRef;
//! use vortex::session::VortexSession;
//! use vortex_datafusion::MemoryPoolAllocator;
//!
//! let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
//! let session = VortexSession::default()
//!     .with_allocator(BufferAllocatorRef::new(MemoryPoolAllocator::new(&pool)));
//! ```
//!
//! # Accounting, not enforcement
//!
//! The allocator grows its reservation with [`MemoryReservation::grow`], which always succeeds,
//! rather than [`MemoryReservation::try_grow`]. Vortex buffers treat a failed allocation as fatal
//! and abort the process, so refusing an allocation over the pool limit would turn a recoverable
//! "resources exhausted" error into an abort. The pool still sees the usage, so pool
//! implementations that weigh consumers against each other, such as `FairSpillPool`, make other
//! operators spill earlier. Enforcing the limit needs fallible allocation paths in
//! `vortex-buffer`.

use std::alloc::AllocError;
use std::alloc::Allocator;
use std::alloc::Global;
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;

use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_execution::memory_pool::MemoryPool;
use datafusion_execution::memory_pool::MemoryReservation;

/// A Vortex buffer allocator that accounts its memory in a DataFusion [`MemoryPool`].
///
/// Memory comes from the global allocator. Every block is added to a single
/// [`MemoryReservation`] when it is allocated or grown and removed when it is shrunk or freed.
/// See the [module documentation](self) for how to install it and why it does not enforce the
/// pool limit.
#[derive(Debug)]
pub struct MemoryPoolAllocator {
    reservation: MemoryReservation,
}

impl MemoryPoolAllocator {
    /// Register a consumer named `vortex` with `pool` and account allocations against it.
    pub fn new(pool: &Arc<dyn MemoryPool>) -> Self {
        Self::with_consumer(MemoryConsumer::new("vortex"), pool)
    }

    /// Register `consumer` with `pool` and account allocations against it.
    pub fn with_consumer(consumer: MemoryConsumer, pool: &Arc<dyn MemoryPool>) -> Self {
        Self {
            reservation: consumer.register(pool),
        }
    }

    /// The reservation that holds the bytes currently allocated through this allocator.
    pub fn reservation(&self) -> &MemoryReservation {
        &self.reservation
    }
}

// SAFETY: every memory operation is forwarded to Global, which satisfies the Allocator contract.
// The reservation only records sizes. Global returns blocks of exactly the requested size, so the
// layouts passed back to deallocate, grow and shrink match the sizes added to the reservation.
unsafe impl Allocator for MemoryPoolAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let block = Global.allocate(layout)?;
        self.reservation.grow(layout.size());
        Ok(block)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let block = Global.allocate_zeroed(layout)?;
        self.reservation.grow(layout.size());
        Ok(block)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { Global.deallocate(ptr, layout) };
        self.reservation.shrink(layout.size());
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        let block = unsafe { Global.grow(ptr, old_layout, new_layout) }?;
        self.reservation.grow(new_layout.size() - old_layout.size());
        Ok(block)
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        let block = unsafe { Global.grow_zeroed(ptr, old_layout, new_layout) }?;
        self.reservation.grow(new_layout.size() - old_layout.size());
        Ok(block)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        let block = unsafe { Global.shrink(ptr, old_layout, new_layout) }?;
        self.reservation.shrink(old_layout.size() - new_layout.size());
        Ok(block)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_execution::memory_pool::MemoryPool;
    use vortex::buffer::Buffer;
    use vortex::buffer::BufferAllocatorRef;

    use super::MemoryPoolAllocator;

    fn pool() -> Arc<dyn MemoryPool> {
        Arc::new(GreedyMemoryPool::new(1 << 20))
    }

    #[test]
    fn buffers_are_accounted_until_dropped() {
        let pool = pool();
        let allocator = BufferAllocatorRef::new(MemoryPoolAllocator::new(&pool));

        let mut buffer = allocator.with_capacity::<u64>(16);
        buffer.extend(0..16u64);
        let reserved = pool.reserved();
        assert!(reserved >= 16 * size_of::<u64>());

        buffer.extend(16..1024u64);
        assert!(pool.reserved() >= 1024 * size_of::<u64>());
        assert!(pool.reserved() > reserved);

        let frozen = buffer.freeze();
        let view = frozen.slice(0..8);
        drop(frozen);
        assert!(pool.reserved() > 0);

        drop(view);
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn vec_in_allocator_stays_accounted_as_a_buffer() {
        let pool = pool();
        let allocator = BufferAllocatorRef::new(MemoryPoolAllocator::new(&pool));

        let mut vec = Vec::with_capacity_in(32, allocator);
        vec.extend(0..32u32);
        assert_eq!(pool.reserved(), 32 * size_of::<u32>());

        let buffer = Buffer::from(vec);
        assert_eq!(pool.reserved(), 32 * size_of::<u32>());

        drop(buffer);
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn allocations_over_the_limit_are_accounted_not_refused() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64));
        let allocator = BufferAllocatorRef::new(MemoryPoolAllocator::new(&pool));

        let buffer = allocator.zeroed::<u8>(1024);
        assert_eq!(buffer.len(), 1024);
        assert!(pool.reserved() >= 1024);
    }
}
