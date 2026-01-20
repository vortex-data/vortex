// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use vortex_array::buffer::DeviceBuffer;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::PooledPinnedBuffer;

/// A device buffer backed by CUDA device memory.
pub struct CudaDeviceBuffer {
    data: Arc<CudaSlice<u8>>,
    offset: usize,
    len: usize,
    stream: Arc<CudaStream>,
    completion: Arc<CudaEvent>,
    host_buffer: Arc<parking_lot::Mutex<Option<PooledPinnedBuffer>>>,
}

impl CudaDeviceBuffer {
    pub fn new(
        data: Arc<CudaSlice<u8>>,
        offset: usize,
        len: usize,
        stream: Arc<CudaStream>,
        completion: CudaEvent,
        host_buffer: PooledPinnedBuffer,
    ) -> Self {
        Self {
            data,
            offset,
            len,
            stream,
            completion: Arc::new(completion),
            host_buffer: Arc::new(parking_lot::Mutex::new(Some(host_buffer))),
        }
    }

    fn view(&self) -> cudarc::driver::CudaView<'_, u8> {
        self.data.slice(self.offset..self.offset + self.len)
    }
}

impl DeviceBuffer for CudaDeviceBuffer {
    fn len(&self) -> usize {
        self.len
    }

    fn copy_to_host(&self) -> VortexResult<ByteBuffer> {
        let mut host = ByteBufferMut::with_capacity_aligned(self.len, Alignment::of::<u8>());
        unsafe { host.set_len(self.len) };
        self.stream
            .memcpy_dtoh(&self.view(), host.as_mut_slice())
            .map_err(|e| vortex_err!("Failed to copy from device: {e}"))?;
        Ok(host.freeze())
    }

    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        if range.start > range.end || range.end > self.len {
            vortex_panic!(
                "range out of bounds: {}..{} for length {}",
                range.start,
                range.end,
                self.len
            );
        }
        Arc::new(Self {
            data: self.data.clone(),
            offset: self.offset + range.start,
            len: range.end - range.start,
            stream: self.stream.clone(),
            completion: self.completion.clone(),
            host_buffer: self.host_buffer.clone(),
        })
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        let _ = self.completion.synchronize();
        if let Some(buffer) = self.host_buffer.lock().take() {
            drop(buffer);
        }
    }
}

impl PartialEq for CudaDeviceBuffer {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
            && self.offset == other.offset
            && self.len == other.len
    }
}

impl Eq for CudaDeviceBuffer {}

impl Hash for CudaDeviceBuffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let ptr = Arc::as_ptr(&self.data) as usize;
        ptr.hash(state);
        self.offset.hash(state);
        self.len.hash(state);
    }
}

impl fmt::Debug for CudaDeviceBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaDeviceBuffer")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}
