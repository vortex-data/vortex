// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A buffer allocator that keeps freed large blocks for reuse.

use std::alloc::Layout;
use std::ptr::NonNull;

use allocator_api2::alloc::AllocError;
use allocator_api2::alloc::Allocator;
use allocator_api2::alloc::Global;
use parking_lot::Mutex;
use vortex_utils::aliases::hash_map::HashMap;

/// Blocks at least this large are kept for reuse when freed.
const MIN_POOLED_SIZE: usize = 16 << 10;
/// The pool keeps at most this many bytes of free blocks.
const MAX_POOLED_BYTES: usize = 256 << 20;

/// Keeps the large blocks that scans free for the next allocation of the same size.
///
/// A scan decodes every split into buffers of about the same sizes. The process allocator returns
/// freed large blocks to the operating system soon after they are freed, so every split would
/// otherwise fault the pages of its buffers in again.
#[derive(Debug, Default)]
pub(crate) struct BufferPool {
    state: Mutex<PoolState>,
}

#[derive(Debug, Default)]
struct PoolState {
    /// The addresses of the free blocks, by block size and alignment.
    free: HashMap<(usize, usize), Vec<usize>>,
    bytes: usize,
}

/// The layout of the block that holds an allocation of `layout`, if the pool keeps it.
fn pooled_layout(layout: Layout) -> Option<Layout> {
    if layout.size() < MIN_POOLED_SIZE {
        return None;
    }
    // Blocks are sized in powers of two, so that buffers whose lengths vary from split to split
    // share blocks. The pages past the end of a buffer are never touched.
    let size = layout.size().checked_next_power_of_two()?;
    Layout::from_size_align(size, layout.align()).ok()
}

unsafe impl Allocator for BufferPool {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let Some(block) = pooled_layout(layout) else {
            return Global.allocate(layout);
        };
        let reused = {
            let mut state = self.state.lock();
            let addr = state
                .free
                .get_mut(&(block.size(), block.align()))
                .and_then(Vec::pop);
            if addr.is_some() {
                state.bytes -= block.size();
            }
            addr
        };
        match reused.and_then(|addr| NonNull::new(addr as *mut u8)) {
            Some(ptr) => Ok(NonNull::slice_from_raw_parts(ptr, block.size())),
            None => Global.allocate(block),
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let Some(block) = pooled_layout(layout) else {
            // SAFETY: blocks the pool does not keep are allocated with `layout` by `Global`.
            unsafe { Global.deallocate(ptr, layout) };
            return;
        };
        {
            let mut state = self.state.lock();
            if state.bytes + block.size() <= MAX_POOLED_BYTES {
                state.bytes += block.size();
                state
                    .free
                    .entry((block.size(), block.align()))
                    .or_default()
                    .push(ptr.as_ptr() as usize);
                return;
            }
        }
        // SAFETY: the block was allocated with the pooled layout by `Global`.
        unsafe { Global.deallocate(ptr, block) };
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        for ((size, align), addrs) in self.state.get_mut().free.drain() {
            // SAFETY: every pooled block was allocated by `Global` with this size and alignment.
            let layout = unsafe { Layout::from_size_align_unchecked(size, align) };
            for addr in addrs {
                if let Some(ptr) = NonNull::new(addr as *mut u8) {
                    // SAFETY: the pool owns the free block.
                    unsafe { Global.deallocate(ptr, layout) };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;

    use allocator_api2::alloc::Allocator;

    use super::BufferPool;

    #[test]
    fn reuses_freed_large_blocks() {
        let pool = BufferPool::default();
        let layout = Layout::from_size_align(100_000, 64).unwrap();
        let first = pool.allocate(layout).unwrap();
        // SAFETY: the block was allocated by the pool with `layout`.
        unsafe { pool.deallocate(first.cast(), layout) };
        let second = pool
            .allocate(Layout::from_size_align(90_000, 64).unwrap())
            .unwrap();
        assert_eq!(first.cast::<u8>(), second.cast::<u8>());
        // SAFETY: the block was allocated by the pool with this layout.
        unsafe { pool.deallocate(second.cast(), Layout::from_size_align(90_000, 64).unwrap()) };
    }

    #[test]
    fn does_not_pool_small_blocks() {
        let pool = BufferPool::default();
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let block = pool.allocate(layout).unwrap();
        assert_eq!(block.len(), 1024);
        // SAFETY: the block was allocated by the pool with `layout`.
        unsafe { pool.deallocate(block.cast(), layout) };
        assert!(pool.state.lock().free.is_empty());
    }
}
