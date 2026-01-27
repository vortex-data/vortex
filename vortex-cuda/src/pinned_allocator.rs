// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::result::memcpy_htod_async;
use futures::future::BoxFuture;
use futures::FutureExt;
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
use crate::stream::await_stream_callback;

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
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>> {
        let buffer = self.pool.get_pooled(len)?;
        Ok(Box::new(AlignedPinnedWriteTarget::new(buffer, alignment)))
    }
}

struct AlignedPinnedWriteTarget {
    buffer: PooledPinnedBuffer,
    alignment: Alignment,
}

impl AlignedPinnedWriteTarget {
    fn new(buffer: PooledPinnedBuffer, alignment: Alignment) -> Self {
        Self { buffer, alignment }
    }
}

impl WriteTarget for AlignedPinnedWriteTarget {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn into_handle(self: Box<Self>) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        async move {
            let ptr = self.buffer.as_slice().as_ptr() as usize;
            let align = *self.alignment;
            // CUDA pinned allocations don't accept an explicit alignment request,
            // so we validate the actual pointer alignment after allocation.
            if align > 1 && ptr % align != 0 {
                return Err(vortex_err!(
                    "Pinned host buffer not aligned to {} (ptr=0x{:x})",
                    align,
                    ptr
                ));
            }
            Ok(BufferHandle::new_host(self.buffer.into_byte_buffer()))
        }
        .boxed()
    }
}

/// Allocator that reads into pinned buffers and transfers to device memory.
pub struct PinnedDeviceAllocator {
    pool: Arc<PinnedByteBufferPool>,
    stream: Arc<CudaStream>,
}

impl PinnedDeviceAllocator {
    pub fn new(pool: Arc<PinnedByteBufferPool>, stream: Arc<CudaStream>) -> Self {
        Self { pool, stream }
    }

    pub fn from_session(
        pool: Arc<PinnedByteBufferPool>,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let stream = session.cuda_session().new_stream()?;
        Ok(Self::new(pool, stream))
    }
}

impl BufferAllocator for PinnedDeviceAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>> {
        let buffer = self.pool.get_pooled(len)?;
        Ok(Box::new(PinnedDeviceWriteTarget {
            buffer,
            stream: self.stream.clone(),
            alignment,
        }))
    }
}

struct PinnedDeviceWriteTarget {
    buffer: PooledPinnedBuffer,
    stream: Arc<CudaStream>,
    alignment: Alignment,
}

impl WriteTarget for PinnedDeviceWriteTarget {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn into_handle(self: Box<Self>) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let len = self.buffer.len();
        let stream = self.stream.clone();
        let host = self.buffer;
        let alignment = self.alignment;
        async move {
            let ptr = host.as_slice().as_ptr() as usize;
            let align = *alignment;
            // CUDA pinned allocations don't accept an explicit alignment request,
            // so we validate the actual pointer alignment after allocation.
            if align > 1 && ptr % align != 0 {
                return Err(vortex_err!(
                    "Pinned host buffer not aligned to {} (ptr=0x{:x})",
                    align,
                    ptr
                ));
            }

            let mut device = unsafe { stream.alloc::<u8>(len) }
                .map_err(|e| vortex_err!("Failed to allocate device memory: {e}"))?;

            let device_ptr = device.device_ptr_mut(&stream).0;
            let host_slice = host.as_slice();
            unsafe {
                memcpy_htod_async(device_ptr, host_slice, stream.cu_stream())
                    .map_err(|e| vortex_err!("Failed to schedule H2D copy: {e}"))?;
            }

            await_stream_callback(&stream).await?;

            // Keep the host buffer alive until the copy completes.
            let _keep_alive = host;

            Ok(BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(device))))
        }
        .boxed()
    }
}
