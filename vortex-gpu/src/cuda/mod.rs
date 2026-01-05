// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::hal::Hal;
use crate::hal::HalBuffer;
use crate::hal::HalDevice;
use crate::hal::HalKind;

pub struct Cuda;

impl Hal for Cuda {
    const KIND: HalKind = HalKind::Cuda;
    type Buffer = CudaBuffer;
    type Device = CudaDevice;
}

#[derive(Clone)]
pub struct CudaDevice {
    stream: Arc<CudaStream>,
}

impl HalDevice<Cuda> for CudaDevice {
    fn alloc(&self, size: usize) -> VortexResult<CudaBuffer> {
        let slice = unsafe { self.stream.alloc::<u8>(size)? };
        Ok(CudaBuffer {
            slice,
            device: self.clone(),
        })
    }

    fn to_host(&self, buffer: CudaBuffer) -> VortexResult<ByteBuffer> {
        let host_buf = self.stream.clone_dtoh(&buffer.slice)?;
        Ok(ByteBuffer::from(host_buf))
    }

    fn to_device(&self, buffer: ByteBuffer) -> VortexResult<CudaBuffer> {
        let slice = self.stream.clone_htod(buffer.as_slice())?;
        Ok(CudaBuffer {
            slice,
            device: self.clone(),
        })
    }
}

#[derive(Clone)]
pub struct CudaBuffer {
    slice: CudaSlice<u8>,
    device: CudaDevice,
}

impl HalBuffer<Cuda> for CudaBuffer {
    fn len(&self) -> usize {
        self.slice.len()
    }
}
