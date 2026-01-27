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
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::BufferAllocator;
use vortex_io::WriteDestination;
use vortex_io::WriteRegion;
use vortex_session::VortexSession;

use crate::device_buffer::CudaDeviceBuffer;
use crate::session::CudaSessionExt;
use crate::stream::await_stream_callback;

/// Allocator that reads into host buffers and copies to device memory.
pub struct HostToDeviceAllocator {
    stream: Arc<CudaStream>,
}

impl HostToDeviceAllocator {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }

    pub fn from_session(session: &VortexSession) -> VortexResult<Self> {
        let stream = session.cuda_session().new_stream()?;
        Ok(Self::new(stream))
    }
}

impl BufferAllocator for HostToDeviceAllocator {
    fn allocate(
        &self,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<Box<dyn WriteDestination>> {
        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
        unsafe { buffer.set_len(len) };
        Ok(Box::new(NaiveDeviceWriteTarget {
            buffer,
            stream: self.stream.clone(),
            alignment,
        }))
    }
}

struct NaiveDeviceWriteTarget {
    buffer: ByteBufferMut,
    stream: Arc<CudaStream>,
    alignment: Alignment,
}

impl WriteDestination for NaiveDeviceWriteTarget {
    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn region(&mut self) -> WriteRegion<'_> {
        WriteRegion::HostSlice(self.buffer.as_mut())
    }

    fn into_handle(self: Box<Self>) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let stream = self.stream.clone();
        let alignment = self.alignment;
        let host = self.buffer;
        async move {
            let len = host.len();
            let mut device = unsafe { stream.alloc::<u8>(len) }
                .map_err(|e| vortex_err!("Failed to allocate device memory: {e}"))?;

            let device_ptr = device.device_ptr_mut(&stream).0;
            let host_slice = host.as_ref();
            unsafe {
                memcpy_htod_async(device_ptr, host_slice, stream.cu_stream())
                    .map_err(|e| vortex_err!("Failed to schedule H2D copy: {e}"))?;
            }

            await_stream_callback(&stream).await?;

            // Keep the host buffer alive until the copy completes.
            let _keep_alive = host;

            Ok(BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(
                device, alignment,
            ))))
        }
        .boxed()
    }
}
