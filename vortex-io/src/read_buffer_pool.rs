// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A pool of freed read buffers, reused for later reads.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::LazyLock;

use allocator_api2::alloc::AllocError;
use allocator_api2::alloc::Allocator;
use allocator_api2::alloc::Global;
use parking_lot::Mutex;
use vortex_buffer::BufferAllocatorRef;
use vortex_utils::aliases::hash_map::HashMap;

/// Buffers smaller than this are left to the global allocator, which reuses them well.
const MIN_POOLED_BYTES: usize = 64 * 1024;

/// The most bytes the pool keeps in freed buffers.
const MAX_POOLED_BYTES: usize = 256 * 1024 * 1024;

/// The allocator that reads use instead of the static allocator.
///
/// A scan reads every segment into a fresh buffer of up to megabytes and frees it once decoded.
/// The global allocator returns such large blocks to the operating system shortly after they are
/// freed, so the next read faults its pages in again, which costs as much as several percent of
/// a scan. The pool keeps freed buffers, up to [`MAX_POOLED_BYTES`], for the reads that follow.
pub(crate) static READ_BUFFER_POOL: LazyLock<BufferAllocatorRef> =
    LazyLock::new(|| BufferAllocatorRef::new(ReadBufferPool::default()));

/// Returns the allocator a read uses in place of `allocator`.
pub(crate) fn read_allocator(allocator: BufferAllocatorRef) -> BufferAllocatorRef {
    if allocator.ptr_eq(BufferAllocatorRef::static_ref()) {
        READ_BUFFER_POOL.clone()
    } else {
        allocator
    }
}

#[derive(Debug, Default)]
struct ReadBufferPool {
    state: Mutex<PoolState>,
}

#[derive(Debug, Default)]
struct PoolState {
    /// Freed buffers by their pooled layout.
    free: HashMap<Layout, Vec<FreeBuffer>>,
    pooled_bytes: usize,
}

#[derive(Debug)]
struct FreeBuffer(NonNull<u8>);

// SAFETY: a freed buffer is owned by the pool alone, so any thread may hand it out.
unsafe impl Send for FreeBuffer {}

/// The layout a pooled buffer for `layout` has: its size rounded up to a class of at most a
/// quarter more, so that reads of similar sizes share buffers.
fn pooled_layout(layout: Layout) -> Option<Layout> {
    if layout.size() < MIN_POOLED_BYTES {
        return None;
    }
    let granule = 1usize << (layout.size().ilog2() - 2);
    let size = layout.size().checked_next_multiple_of(granule)?;
    Layout::from_size_align(size, layout.align()).ok()
}

// SAFETY: blocks come from the global allocator with the pooled layout, which `deallocate`
// recomputes from the caller's layout, and a block is handed out at most once until it is freed.
unsafe impl Allocator for ReadBufferPool {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let Some(pooled) = pooled_layout(layout) else {
            return Global.allocate(layout);
        };
        let reused = {
            let mut state = self.state.lock();
            let reused = state.free.get_mut(&pooled).and_then(Vec::pop);
            if reused.is_some() {
                state.pooled_bytes -= pooled.size();
            }
            reused
        };
        match reused {
            Some(FreeBuffer(ptr)) => Ok(NonNull::slice_from_raw_parts(ptr, pooled.size())),
            None => Global.allocate(pooled),
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let Some(pooled) = pooled_layout(layout) else {
            // SAFETY: the block was allocated by the global allocator with `layout`.
            return unsafe { Global.deallocate(ptr, layout) };
        };
        let mut state = self.state.lock();
        if state.pooled_bytes + pooled.size() <= MAX_POOLED_BYTES {
            state.pooled_bytes += pooled.size();
            state.free.entry(pooled).or_default().push(FreeBuffer(ptr));
        } else {
            drop(state);
            // SAFETY: the block was allocated by the global allocator with `pooled`.
            unsafe { Global.deallocate(ptr, pooled) };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;

    use allocator_api2::alloc::Allocator;

    use super::ReadBufferPool;
    use super::pooled_layout;

    #[test]
    fn pooled_layout_rounds_up_by_at_most_a_quarter() {
        for size in [
            64 * 1024,
            70_000,
            100_000,
            1 << 20,
            (1 << 20) + 1,
            3_000_000,
        ] {
            let layout = Layout::from_size_align(size, 64).unwrap();
            let pooled = pooled_layout(layout).unwrap();
            assert!(pooled.size() >= size && pooled.size() <= size + size / 4);
            assert_eq!(pooled.align(), 64);
        }
        assert!(pooled_layout(Layout::from_size_align(1000, 8).unwrap()).is_none());
    }

    #[test]
    fn freed_buffers_are_reused() {
        let pool = ReadBufferPool::default();
        let layout = Layout::from_size_align(100_000, 64).unwrap();
        let first = pool.allocate(layout).unwrap();
        // SAFETY: `first` was allocated by `pool` with `layout`.
        unsafe { pool.deallocate(first.cast(), layout) };
        let second = pool
            .allocate(Layout::from_size_align(99_000, 64).unwrap())
            .unwrap();
        assert_eq!(first.cast::<u8>(), second.cast::<u8>());
        // SAFETY: `second` was allocated by `pool` with this layout.
        unsafe { pool.deallocate(second.cast(), Layout::from_size_align(99_000, 64).unwrap()) };
    }
}
