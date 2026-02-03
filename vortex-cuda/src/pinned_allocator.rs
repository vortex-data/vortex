// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::BufferAllocator;
use vortex_io::WriteTarget;
use vortex_session::VortexSession;

use crate::PinnedByteBufferPool;
use crate::PooledPinnedBuffer;
use crate::device_buffer::CudaDeviceBuffer;
use crate::session::CudaSessionExt;
use crate::stream::VortexCudaStream;

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

    fn into_handle(self: Box<Self>) -> VortexResult<BufferHandle> {
        Ok(BufferHandle::new_host(self.into_byte_buffer()))
    }
}

/// Allocator that reads into pinned buffers and transfers to device memory.
pub struct PinnedDeviceAllocator {
    pool: Arc<PinnedByteBufferPool>,
    stream: VortexCudaStream,
}

impl PinnedDeviceAllocator {
    pub fn new(pool: Arc<PinnedByteBufferPool>, stream: VortexCudaStream) -> Self {
        Self { pool, stream }
    }

    pub fn from_session(
        pool: Arc<PinnedByteBufferPool>,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let stream = session.cuda_session().new_stream()?;
        Ok(Self::new(pool, stream))
    }

    pub fn synchronize(&self) -> VortexResult<()> {
        self.stream
            .0
            .synchronize()
            .map_err(|e| vortex_err!("Failed to synchronize CUDA stream: {e}"))
    }
}

impl BufferAllocator for PinnedDeviceAllocator {
    fn allocate(&self, len: usize, _alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>> {
        let buffer = self.pool.get_pooled(len)?;
        Ok(Box::new(PinnedDeviceWriteTarget {
            buffer,
            stream: self.stream.clone(),
        }))
    }
}

struct PinnedDeviceWriteTarget {
    buffer: PooledPinnedBuffer,
    stream: VortexCudaStream,
}

impl WriteTarget for PinnedDeviceWriteTarget {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn into_handle(self: Box<Self>) -> VortexResult<BufferHandle> {
        let len = self.buffer.len();
        let mut device = unsafe { self.stream.0.alloc::<u8>(len) }
            .map_err(|e| vortex_err!("Failed to allocate device memory: {e}"))?;

        self.stream
            .0
            .memcpy_htod(&self.buffer, &mut device)
            .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;

        let event = self
            .stream
            .0
            .record_event(None)
            .map_err(|e| vortex_err!("Failed to record CUDA event: {e}"))?;

        let device_buffer = CudaDeviceBuffer::new_with_host_buffer(device, event, self.buffer);

        Ok(BufferHandle::new_device(Arc::new(device_buffer)))
    }
}
