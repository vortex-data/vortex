// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains an equivalent API to the `vortex-vector` crate, except made abstract
//! over GPU buffers. In theory, we could look to parameterize the vector crate by a buffer type.

use vortex_dtype::PType;

pub trait GpuBuffer {}
impl<T> GpuBuffer for T {}

pub enum GpuVector<B: GpuBuffer> {
    Null,
    Bool,
    Primitive(PrimitiveGpuVector<B>),
}

pub struct BoolGpuVector<B: GpuBuffer> {
    len: usize,
    buffer: B,
    // validity:
}

pub struct PrimitiveGpuVector<B: GpuBuffer> {
    ptype: PType,
    len: usize,
    buffer: B,
    // validity:
}

impl<B: GpuBuffer> PrimitiveGpuVector<B> {
    pub unsafe fn new_unchecked(ptype: PType, len: usize, buffer: B) -> Self {
        Self { ptype, len, buffer }
    }
}

// TODO(ngates): BitBuffer to wrap gpu buffer?
pub struct GpuBitBuffer<B: GpuBuffer> {
    buffer: B,
    offset: usize,
    len: usize,
}
