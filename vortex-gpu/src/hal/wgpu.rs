// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use wgpu::util::DeviceExt;

use crate::hal::Hal;
use crate::hal::HalBuffer;
use crate::hal::HalDevice;

pub struct Wgpu;

impl Hal for Wgpu {
    type Buffer = WgpuBuffer;
    type Device = WgpuDevice;
}

#[derive(Clone)]
pub struct WgpuDevice {
    device: wgpu::Device,
}

impl HalDevice<Wgpu> for WgpuDevice {
    fn alloc(&self, size: usize) -> VortexResult<WgpuBuffer> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Ok(WgpuBuffer {
            buffer,
            device: self.clone(),
        })
    }

    fn to_host(&self, buffer: WgpuBuffer) -> VortexResult<ByteBuffer> {
        let mapped = buffer.buffer.get_mapped_range(0..buffer.len() as u64);
        let bytes = ByteBuffer::copy_from(mapped.deref());
        buffer.buffer.unmap();
        Ok(bytes)
    }

    fn to_device(&self, buffer: ByteBuffer) -> VortexResult<WgpuBuffer> {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: buffer.as_slice(),
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::STORAGE,
            });
        Ok(WgpuBuffer {
            buffer,
            device: self.clone(),
        })
    }
}

#[derive(Clone)]
pub struct WgpuBuffer {
    buffer: wgpu::Buffer,
    device: WgpuDevice,
}

impl HalBuffer<Wgpu> for WgpuBuffer {
    fn len(&self) -> usize {
        usize::try_from(self.buffer.size()).vortex_expect("buffer fits into usize")
    }
}
