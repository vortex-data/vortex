// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

impl BufferAllocator for DefaultAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>> {
        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
        unsafe { buffer.set_len(len) };
        Ok(Box::new(buffer))
    }
}
