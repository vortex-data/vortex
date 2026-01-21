// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use crate::WriteTarget;

/// Allocates buffers for I/O reads.
pub trait BufferAllocator: Send + Sync + 'static {
    /// Allocate a buffer for the requested length and alignment.
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>>;
}

/// The default allocator that uses `ByteBufferMut`.
pub struct DefaultAllocator;

/// Allocation counters for the default allocator.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultAllocStats {
    pub count: u64,
    pub bytes: u64,
}

static DEFAULT_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static DEFAULT_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn default_alloc_stats() -> DefaultAllocStats {
    DefaultAllocStats {
        count: DEFAULT_ALLOC_COUNT.load(Ordering::Relaxed),
        bytes: DEFAULT_ALLOC_BYTES.load(Ordering::Relaxed),
    }
}

pub fn reset_default_alloc_stats() {
    DEFAULT_ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEFAULT_ALLOC_BYTES.store(0, Ordering::Relaxed);
}

impl BufferAllocator for DefaultAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>> {
        DEFAULT_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        DEFAULT_ALLOC_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
        unsafe { buffer.set_len(len) };
        Ok(Box::new(buffer))
    }
}
