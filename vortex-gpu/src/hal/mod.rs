// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hardware Abstraction Layer (HAL) for `vortex-gpu`.

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

pub mod cuda;
pub mod wgpu;

pub trait Hal: Sized {
    type Buffer: HalBuffer<Self>;
    type Device: HalDevice<Self>;
}

pub trait HalDevice<H: Hal>: Clone {
    fn alloc(&self, size: usize) -> VortexResult<H::Buffer>;
    fn to_host(&self, buffer: H::Buffer) -> VortexResult<ByteBuffer>;
    fn to_device(&self, buffer: ByteBuffer) -> VortexResult<H::Buffer>;
}

pub trait HalBuffer<H: Hal>: Clone {
    fn len(&self) -> usize;
}
