// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_io::BufferAllocator;
use vortex_io::WriteTarget;

use crate::PinnedByteBufferPool;
use crate::PooledPinnedBuffer;

/// Allocator that sources buffers from a CUDA pinned pool.
pub struct PinnedBufferAllocator {
    pool: Arc<PinnedByteBufferPool>,
}

impl PinnedBufferAllocator {
    pub fn new(pool: Arc<PinnedByteBufferPool>) -> Self {
        Self { pool }
    }
}

impl BufferAllocator for PinnedBufferAllocator {
    fn allocate(&self, len: usize, _alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>> {
        let buffer = self.pool.get_pooled(len)?;
        Ok(Box::new(buffer))
    }
}

impl WriteTarget for PooledPinnedBuffer {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        PooledPinnedBuffer::as_mut_slice(self)
    }

    fn len(&self) -> usize {
        PooledPinnedBuffer::len(self)
    }

    fn into_handle(self: Box<Self>) -> BufferHandle {
        BufferHandle::new_host(self.into_byte_buffer())
    }
}
