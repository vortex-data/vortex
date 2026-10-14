// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A buffer allocator that keeps large freed blocks for reuse.
//!
//! A freshly allocated large block is usually backed by pages that the kernel has not mapped yet.
//! The first write to each page takes a page fault, and each fault takes the process-wide memory
//! map lock. When many threads read and decode at the same time, these faults wait behind any
//! thread that changes the memory map (`mmap`, `munmap`, `mprotect`, `madvise`), and scans stall.
//!
//! [`PooledBufferAllocator`] keeps freed large blocks and gives them back to later allocations of
//! the same size class. The pages of a reused block are already mapped, so writes to it do not
//! fault and do not take the memory map lock.

use std::alloc::Layout;
use std::cell::Cell;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use allocator_api2::alloc::AllocError;
use allocator_api2::alloc::Allocator;
use allocator_api2::alloc::Global;
use parking_lot::Mutex;
use vortex_utils::parallelism::get_available_parallelism;

/// Allocations smaller than this go directly to the global allocator.
const MIN_POOLED_SIZE: usize = 64 << 10;

/// Allocations larger than this go directly to the global allocator.
const MAX_POOLED_SIZE: usize = 64 << 20;

/// The alignment of every pooled block. Requests with a larger alignment are not pooled.
const BLOCK_ALIGNMENT: usize = 4096;

/// Each power of two is divided into this many size classes, which limits the unused tail of a
/// block to a quarter of its size.
const CLASSES_PER_DOUBLING: usize = 4;

const MIN_POOLED_LOG2: usize = MIN_POOLED_SIZE.trailing_zeros() as usize;

const MAX_POOLED_LOG2: usize = MAX_POOLED_SIZE.trailing_zeros() as usize;

const NUM_CLASSES: usize = (MAX_POOLED_LOG2 - MIN_POOLED_LOG2) * CLASSES_PER_DOUBLING + 1;

/// The number of independent free lists per size class. Threads use different shards to avoid
/// contention on a single lock.
const NUM_SHARDS: usize = 16;

/// The minimum default limit on the bytes that the pool keeps.
const MIN_DEFAULT_RETAINED_BYTES: usize = 256 << 20;

/// The default limit on the bytes that the pool keeps, per available thread.
const DEFAULT_RETAINED_BYTES_PER_THREAD: usize = 8 << 20;

/// A buffer allocator that keeps large freed blocks and reuses them.
///
/// Allocations from 64 KiB to 64 MiB are rounded up to a size class. When such a block is freed,
/// the pool keeps it, until the pool holds `max_retained_bytes`. A later allocation of the same
/// size class takes a kept block before it asks the global allocator. Other allocations go
/// directly to the global allocator.
///
/// Use this allocator when many threads allocate and free large buffers at a high rate, for
/// example the read and decode buffers of a parallel scan. See the module documentation for the
/// reason.
pub(crate) struct PooledBufferAllocator {
    classes: Box<[SizeClass]>,
    retained_bytes: AtomicUsize,
    max_retained_bytes: usize,
}

impl PooledBufferAllocator {
    /// Create a pool that keeps at most `max_retained_bytes` of freed blocks.
    pub(crate) fn new(max_retained_bytes: usize) -> Self {
        Self {
            classes: (0..NUM_CLASSES).map(|_| SizeClass::default()).collect(),
            retained_bytes: AtomicUsize::new(0),
            max_retained_bytes,
        }
    }

    /// Returns the number of bytes in freed blocks that the pool keeps now.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes.load(Ordering::Relaxed)
    }

    fn try_retain(&self, size: usize) -> bool {
        self.retained_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |retained| {
                let retained = retained.checked_add(size)?;
                (retained <= self.max_retained_bytes).then_some(retained)
            })
            .is_ok()
    }

    fn pop(&self, class: usize) -> Option<NonNull<u8>> {
        let shards = &self.classes[class].shards;
        let home = shard_index();

        for offset in 0..NUM_SHARDS {
            let shard = &shards[(home + offset) % NUM_SHARDS];

            // Only the home shard is worth a wait. Other shards are a fallback that must not
            // add contention.
            let mut blocks = if offset == 0 {
                shard.lock()
            } else {
                match shard.try_lock() {
                    Some(blocks) => blocks,
                    None => continue,
                }
            };

            if let Some(block) = blocks.pop() {
                drop(blocks);
                self.retained_bytes
                    .fetch_sub(class_size(class), Ordering::Relaxed);
                return Some(block.0);
            }
        }

        None
    }

    fn push(&self, class: usize, ptr: NonNull<u8>) -> bool {
        if !self.try_retain(class_size(class)) {
            return false;
        }

        self.classes[class].shards[shard_index()]
            .lock()
            .push(Block(ptr));
        true
    }
}

impl Default for PooledBufferAllocator {
    /// Create a pool that keeps 8 MiB of freed blocks per available thread, and at least 256 MiB.
    fn default() -> Self {
        let threads = get_available_parallelism().unwrap_or(1);
        Self::new(
            threads
                .saturating_mul(DEFAULT_RETAINED_BYTES_PER_THREAD)
                .max(MIN_DEFAULT_RETAINED_BYTES),
        )
    }
}

impl fmt::Debug for PooledBufferAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PooledBufferAllocator")
            .field("retained_bytes", &self.retained_bytes())
            .field("max_retained_bytes", &self.max_retained_bytes)
            .finish()
    }
}

impl Drop for PooledBufferAllocator {
    fn drop(&mut self) {
        for (class, size_class) in self.classes.iter_mut().enumerate() {
            let layout = class_layout(class);

            for shard in size_class.shards.iter_mut() {
                let blocks = shard.get_mut();

                for block in blocks.drain(..) {
                    // SAFETY: every kept block was allocated by Global with the layout of its
                    // size class, and the pool owns it.
                    unsafe { Global.deallocate(block.0, layout) };
                }
            }
        }
    }
}

// SAFETY: blocks come from Global with the layout of their size class, and the pool hands out a
// block to at most one owner at a time. Requests outside the pooled range forward to Global
// unchanged.
unsafe impl Allocator for PooledBufferAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let Some(class) = size_class(layout) else {
            return Global.allocate(layout);
        };

        let ptr = match self.pop(class) {
            Some(ptr) => ptr,
            None => Global.allocate(class_layout(class))?.cast(),
        };
        Ok(NonNull::slice_from_raw_parts(ptr, class_size(class)))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let Some(class) = size_class(layout) else {
            // SAFETY: requests outside the pooled range were allocated by Global with `layout`.
            unsafe { Global.deallocate(ptr, layout) };
            return;
        };

        if !self.push(class, ptr) {
            // SAFETY: pooled blocks were allocated by Global with the layout of their class.
            unsafe { Global.deallocate(ptr, class_layout(class)) };
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { self.resize(ptr, old_layout, new_layout) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller upholds the Allocator contract.
        unsafe { self.resize(ptr, old_layout, new_layout) }
    }
}

impl PooledBufferAllocator {
    /// Move a block to a new layout. The block stays in place when both layouts use the same size
    /// class. Otherwise, the data moves to a new block.
    ///
    /// # Safety
    ///
    /// `ptr` must denote a block that this allocator allocated with `old_layout`.
    unsafe fn resize(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_class = size_class(old_layout);
        let new_class = size_class(new_layout);

        if let Some(class) = old_class
            && old_class == new_class
        {
            return Ok(NonNull::slice_from_raw_parts(ptr, class_size(class)));
        }

        if old_class.is_none() && new_class.is_none() {
            // SAFETY: both layouts are outside the pooled range, so Global owns the block.
            return if new_layout.size() >= old_layout.size() {
                unsafe { Global.grow(ptr, old_layout, new_layout) }
            } else {
                unsafe { Global.shrink(ptr, old_layout, new_layout) }
            };
        }

        let new_block = self.allocate(new_layout)?;
        let copy_len = old_layout.size().min(new_layout.size());

        // SAFETY: both blocks are live, distinct, and at least `copy_len` bytes long. The caller
        // gives `ptr` and `old_layout` from this allocator.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_block.cast().as_ptr(), copy_len);
            self.deallocate(ptr, old_layout);
        }
        Ok(new_block)
    }
}

#[derive(Default)]
struct SizeClass {
    shards: [Mutex<Vec<Block>>; NUM_SHARDS],
}

/// A freed block that the pool owns.
struct Block(NonNull<u8>);

// SAFETY: a kept block is plain memory that no other owner references.
unsafe impl Send for Block {}

/// Returns the size class of `layout`, or `None` if the pool must not serve it.
fn size_class(layout: Layout) -> Option<usize> {
    let size = layout.size();
    if !(MIN_POOLED_SIZE..=MAX_POOLED_SIZE).contains(&size) || layout.align() > BLOCK_ALIGNMENT {
        return None;
    }

    // `size` is in the pooled range, so `log2` is at least MIN_POOLED_LOG2.
    let log2 = size.ilog2() as usize;
    let base = 1usize << log2;
    let step = base / CLASSES_PER_DOUBLING;
    let sub_class = (size - base).div_ceil(step);

    Some((log2 - MIN_POOLED_LOG2) * CLASSES_PER_DOUBLING + sub_class)
}

/// Returns the block size of a size class. Every request of the class fits in this size.
fn class_size(class: usize) -> usize {
    let log2 = MIN_POOLED_LOG2 + class / CLASSES_PER_DOUBLING;
    let base = 1usize << log2;
    base + (class % CLASSES_PER_DOUBLING) * (base / CLASSES_PER_DOUBLING)
}

fn class_layout(class: usize) -> Layout {
    Layout::from_size_align(class_size(class), BLOCK_ALIGNMENT)
        .unwrap_or_else(|_| unreachable!("size classes have valid layouts"))
}

/// Returns the free-list shard of the current thread. Threads get shards in turn.
fn shard_index() -> usize {
    static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        static SHARD: Cell<usize> =
            Cell::new(NEXT_SHARD.fetch_add(1, Ordering::Relaxed) % NUM_SHARDS);
    }

    SHARD.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;

    use allocator_api2::alloc::Allocator;
    use rstest::rstest;
    use vortex::buffer::Alignment;
    use vortex::buffer::BufferAllocatorRef;

    use super::*;

    #[rstest]
    #[case(MIN_POOLED_SIZE)]
    #[case(MIN_POOLED_SIZE + 1)]
    #[case(100_000)]
    #[case(1 << 20)]
    #[case((1 << 20) + 1)]
    #[case(3 << 20)]
    #[case(MAX_POOLED_SIZE)]
    fn test_size_class_fits_request(#[case] size: usize) {
        let layout = Layout::from_size_align(size, 64).unwrap();
        let class = size_class(layout).unwrap();

        assert!(class < NUM_CLASSES);
        assert!(class_size(class) >= size);
        assert!(class_size(class) - size <= size / CLASSES_PER_DOUBLING);
    }

    #[rstest]
    #[case(MIN_POOLED_SIZE - 1, 64)]
    #[case(MAX_POOLED_SIZE + 1, 64)]
    #[case(1 << 20, BLOCK_ALIGNMENT * 2)]
    fn test_size_class_rejects_unpooled(#[case] size: usize, #[case] align: usize) {
        let layout = Layout::from_size_align(size, align).unwrap();
        assert_eq!(size_class(layout), None);
    }

    #[test]
    fn test_size_class_boundaries_are_monotonic() {
        for class in 1..NUM_CLASSES {
            assert!(class_size(class) > class_size(class - 1));
        }
        assert_eq!(class_size(NUM_CLASSES - 1), MAX_POOLED_SIZE);
    }

    #[test]
    fn test_pooled_allocator_reuses_freed_block() {
        let pool = PooledBufferAllocator::new(64 << 20);
        let layout = Layout::from_size_align(1 << 20, 64).unwrap();

        let first = pool.allocate(layout).unwrap().cast::<u8>();
        // SAFETY: `first` was allocated by `pool` with `layout`.
        unsafe { pool.deallocate(first, layout) };
        assert!(pool.retained_bytes() >= 1 << 20);

        let second = pool.allocate(layout).unwrap().cast::<u8>();
        assert_eq!(first, second);
        assert_eq!(pool.retained_bytes(), 0);

        // SAFETY: `second` was allocated by `pool` with `layout`.
        unsafe { pool.deallocate(second, layout) };
    }

    #[test]
    fn test_pooled_allocator_respects_retention_limit() {
        let pool = PooledBufferAllocator::new(0);
        let layout = Layout::from_size_align(1 << 20, 64).unwrap();

        let block = pool.allocate(layout).unwrap().cast::<u8>();
        // SAFETY: `block` was allocated by `pool` with `layout`.
        unsafe { pool.deallocate(block, layout) };
        assert_eq!(pool.retained_bytes(), 0);
    }

    #[test]
    fn test_pooled_allocator_grow_keeps_data() {
        let pool = PooledBufferAllocator::new(64 << 20);
        let small = Layout::from_size_align(1 << 10, 8).unwrap();
        let in_class = Layout::from_size_align((1 << 20) + 1000, 8).unwrap();
        let same_class = Layout::from_size_align((1 << 20) + 2000, 8).unwrap();
        let larger = Layout::from_size_align(4 << 20, 8).unwrap();

        let ptr = pool.allocate(small).unwrap().cast::<u8>();
        // SAFETY: the block is at least `small.size()` bytes long.
        unsafe { ptr.as_ptr().write_bytes(7, small.size()) };

        // SAFETY: every step passes the block and layout from the previous step.
        unsafe {
            let ptr = pool.grow(ptr, small, in_class).unwrap().cast::<u8>();
            let same = pool.grow(ptr, in_class, same_class).unwrap().cast::<u8>();
            assert_eq!(ptr, same);

            let moved = pool.grow(same, same_class, larger).unwrap().cast::<u8>();
            let data = std::slice::from_raw_parts(moved.as_ptr(), small.size());
            assert!(data.iter().all(|&byte| byte == 7));

            let shrunk = pool.shrink(moved, larger, small).unwrap().cast::<u8>();
            let data = std::slice::from_raw_parts(shrunk.as_ptr(), small.size());
            assert!(data.iter().all(|&byte| byte == 7));

            pool.deallocate(shrunk, small);
        }
    }

    #[test]
    fn test_pooled_allocator_backs_buffers() {
        let allocator = BufferAllocatorRef::new(PooledBufferAllocator::new(64 << 20));

        let mut buffer = allocator.with_capacity_aligned::<u8>(1 << 20, Alignment::new(64));
        buffer.extend_from_slice(&[1u8; 1 << 20]);
        let frozen = buffer.freeze();

        assert_eq!(frozen.len(), 1 << 20);
        assert!(frozen.iter().all(|&byte| byte == 1));
        assert_eq!(frozen.as_ptr().align_offset(64), 0);
    }
}
