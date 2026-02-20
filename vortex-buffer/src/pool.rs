// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A global buffer pool that recycles page-aligned memory blocks using size-class buckets.
//!
//! Instead of allocating and freeing memory through the system allocator on every
//! `BufferMut` construction/drop, the pool maintains free-lists of recently freed blocks
//! grouped by size class. This reduces allocation churn in workloads with many short-lived
//! buffers.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::LazyLock;

use parking_lot::Mutex;
use vortex_error::vortex_panic;

/// Page size used for all pooled allocations.
const PAGE_SIZE: usize = 4096;

/// Maximum number of free blocks retained per bucket.
const MAX_FREE_PER_BUCKET: usize = 1024;

/// Size classes for pooled allocations.
const BUCKET_SIZES: [usize; 11] = [
    1 << 10,   // 1 KiB
    8 << 10,   // 8 KiB
    64 << 10,  // 64 KiB
    512 << 10, // 512 KiB
    1 << 20,   // 1 MiB
    4 << 20,   // 4 MiB
    8 << 20,   // 8 MiB
    16 << 20,  // 16 MiB
    32 << 20,  // 32 MiB
    64 << 20,  // 64 MiB
    128 << 20, // 128 MiB
];

/// Maximum size that the pool will handle. Larger allocations bypass the pool.
const MAX_POOLED_SIZE: usize = 128 << 20;

/// Global buffer pool instance.
pub(crate) static POOL: LazyLock<BufferPool> = LazyLock::new(BufferPool::new);

/// Construct a [`Layout`] for a bucket of the given size, or panic.
fn bucket_layout(size: usize) -> Layout {
    // All bucket sizes are valid (power-of-two page alignment, nonzero size).
    Layout::from_size_align(size, PAGE_SIZE)
        .unwrap_or_else(|_| vortex_panic!("invalid bucket layout: size={size}"))
}

/// Returns the bucket index for a given size, or `None` if the size exceeds all buckets.
fn bucket_index(size: usize) -> Option<usize> {
    BUCKET_SIZES.iter().position(|&s| s >= size)
}

/// A single size-class bucket holding a free-list of memory blocks.
struct Bucket {
    free_list: Mutex<Vec<NonNull<u8>>>,
    size: usize,
}

// SAFETY: The raw pointers in the free-list are owned exclusively by the pool.
unsafe impl Send for Bucket {}
unsafe impl Sync for Bucket {}

impl Bucket {
    fn new(size: usize) -> Self {
        Self {
            free_list: Mutex::new(Vec::new()),
            size,
        }
    }

    fn acquire(&self) -> NonNull<u8> {
        if let Some(ptr) = self.free_list.lock().pop() {
            return ptr;
        }
        let layout = bucket_layout(self.size);
        // SAFETY: layout has non-zero size (all bucket sizes are > 0).
        let ptr = unsafe { std::alloc::alloc(layout) };
        NonNull::new(ptr).unwrap_or_else(|| vortex_panic!("allocation failed"))
    }

    fn release(&self, ptr: NonNull<u8>) {
        let mut free_list = self.free_list.lock();
        if free_list.len() < MAX_FREE_PER_BUCKET {
            free_list.push(ptr);
        } else {
            let layout = bucket_layout(self.size);
            // SAFETY: ptr was allocated with this exact layout.
            unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
        }
    }
}

impl Drop for Bucket {
    fn drop(&mut self) {
        let layout = bucket_layout(self.size);
        let free_list = self.free_list.get_mut();
        for ptr in free_list.drain(..) {
            // SAFETY: ptr was allocated with this exact layout.
            unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
        }
    }
}

/// A pool of page-aligned memory blocks organized by size class.
pub(crate) struct BufferPool {
    buckets: [Bucket; 11],
}

impl BufferPool {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|i| Bucket::new(BUCKET_SIZES[i])),
        }
    }

    /// Acquire a memory block of at least `size` bytes with the given alignment.
    ///
    /// - If `size == 0`, returns an empty (dangling) allocation.
    /// - If `alignment > PAGE_SIZE` or `size > MAX_POOLED_SIZE`, allocates directly from the
    ///   system allocator (bypassing the pool).
    /// - Otherwise, rounds up to the next bucket size and pops from the free-list (or allocates
    ///   fresh).
    pub(crate) fn acquire(&self, size: usize, alignment: usize) -> PooledAllocation {
        if size == 0 {
            return PooledAllocation::empty(alignment);
        }

        if alignment > PAGE_SIZE || size > MAX_POOLED_SIZE {
            return PooledAllocation::direct_alloc(size, alignment);
        }

        let idx = bucket_index(size).unwrap_or_else(|| vortex_panic!("no bucket for size={size}"));
        let bucket = &self.buckets[idx];
        let ptr = bucket.acquire();
        PooledAllocation {
            ptr,
            capacity: bucket.size,
            align: PAGE_SIZE,
            offset: 0,
            len: 0,
        }
    }
}

/// An owned, raw memory allocation that may be backed by the global pool.
///
/// On drop, pooled allocations are returned to the pool's free-list. Direct (oversized or
/// over-aligned) allocations are freed via the system allocator.
///
/// The `offset` and `len` fields control the window exposed by `AsRef<[u8]>`, which is used
/// by `Bytes::from_owner` when freezing a `BufferMut` into a `Buffer`.
pub(crate) struct PooledAllocation {
    pub(crate) ptr: NonNull<u8>,
    pub(crate) capacity: usize,
    pub(crate) align: usize,
    /// Byte offset to the start of visible data (set before freeze).
    pub(crate) offset: usize,
    /// Byte length of visible data (set before freeze).
    pub(crate) len: usize,
}

// SAFETY: We exclusively own the memory behind `ptr`. The pool uses a Mutex for thread safety.
unsafe impl Send for PooledAllocation {}
// SAFETY: No shared mutation of the raw pointer outside of pool acquire/release.
unsafe impl Sync for PooledAllocation {}

impl PooledAllocation {
    /// Create an empty allocation with a dangling pointer aligned to the given alignment.
    fn empty(alignment: usize) -> Self {
        let align = alignment.max(PAGE_SIZE);
        Self {
            // A dangling pointer whose numeric value satisfies the alignment.
            // SAFETY: align is a power of two and > 0, so this is non-null.
            ptr: unsafe { NonNull::new_unchecked(align as *mut u8) },
            capacity: 0,
            align,
            offset: 0,
            len: 0,
        }
    }

    /// Allocate directly from the system allocator, bypassing the pool.
    fn direct_alloc(size: usize, alignment: usize) -> Self {
        let alignment = alignment.max(1);
        let layout = Layout::from_size_align(size, alignment)
            .unwrap_or_else(|_| vortex_panic!("invalid layout: size={size}, align={alignment}"));
        // SAFETY: layout has non-zero size (checked by caller).
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(ptr)
            .unwrap_or_else(|| vortex_panic!("allocation failed: size={size}, align={alignment}"));
        Self {
            ptr,
            capacity: size,
            align: alignment,
            offset: 0,
            len: 0,
        }
    }
}

impl AsRef<[u8]> for PooledAllocation {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: The range [ptr+offset .. ptr+offset+len] is within the allocation and
        // initialized (set by BufferMut::freeze before calling Bytes::from_owner).
        // For empty allocations (len=0), ptr is a properly-aligned dangling pointer and
        // from_raw_parts with len=0 is valid for any non-null, aligned pointer.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(self.offset), self.len) }
    }
}

impl Drop for PooledAllocation {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return;
        }

        // Try to return to the pool if this was a pooled allocation.
        if self.align == PAGE_SIZE
            && self.capacity <= MAX_POOLED_SIZE
            && let Some(idx) = BUCKET_SIZES.iter().position(|&s| s == self.capacity)
        {
            POOL.buckets[idx].release(self.ptr);
            return;
        }

        // Direct allocation: free via system allocator.
        let layout = Layout::from_size_align(self.capacity, self.align).unwrap_or_else(|_| {
            vortex_panic!(
                "invalid layout in drop: capacity={}, align={}",
                self.capacity,
                self.align
            )
        });
        // SAFETY: ptr was allocated with this exact layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn acquire_and_release_pooled() {
        let alloc = POOL.acquire(100, 1);
        assert!(alloc.capacity >= 100);
        assert_eq!(alloc.capacity, BUCKET_SIZES[0]); // 1 KiB
        assert_eq!(alloc.ptr.as_ptr().align_offset(PAGE_SIZE), 0);
        // Drop returns to pool
        drop(alloc);

        // Second acquire should reuse the pooled block
        let alloc2 = POOL.acquire(100, 1);
        assert_eq!(alloc2.capacity, BUCKET_SIZES[0]);
        assert_eq!(alloc2.ptr.as_ptr().align_offset(PAGE_SIZE), 0);
    }

    #[test]
    fn acquire_zero_size() {
        let alloc = POOL.acquire(0, 4096);
        assert_eq!(alloc.capacity, 0);
    }

    #[test]
    fn acquire_oversized_bypasses_pool() {
        let alloc = POOL.acquire(MAX_POOLED_SIZE + 1, 1);
        assert!(alloc.capacity > MAX_POOLED_SIZE);
        // Should not panic on drop (direct dealloc)
    }

    #[test]
    fn acquire_over_aligned_bypasses_pool() {
        let alloc = POOL.acquire(100, PAGE_SIZE * 2);
        assert!(alloc.capacity >= 100);
        assert_eq!(alloc.ptr.as_ptr().align_offset(PAGE_SIZE * 2), 0);
    }

    #[test]
    fn bucket_selection() {
        // Should get the 8 KiB bucket for a 2 KiB request
        let alloc = POOL.acquire(2048, 1);
        assert_eq!(alloc.capacity, 8 << 10);
    }

    #[test]
    fn as_ref_empty() {
        let alloc = POOL.acquire(0, 1);
        assert!(alloc.as_ref().is_empty());
    }

    #[test]
    fn as_ref_with_data() {
        let mut alloc = POOL.acquire(100, 1);
        // Write some data
        unsafe {
            alloc.ptr.as_ptr().write_bytes(42, 10);
        }
        alloc.offset = 0;
        alloc.len = 10;
        let slice = alloc.as_ref();
        assert_eq!(slice.len(), 10);
        assert!(slice.iter().all(|&b| b == 42));
    }
}
