// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains an equivalent API to the `vortex-vector` crate, except made abstract
//! over GPU buffers. In theory, we could look to parameterize the vector crate by a buffer type.

use vortex_dtype::PType;

use crate::hal::Hal;

pub enum GpuVector<H: Hal> {
    Null,
    Bool,
    Primitive(PrimitiveGpuVector<H>),
}

pub struct BoolGpuVector<H: Hal> {
    len: usize,
    buffer: H::Buffer,
    // validity:
}

pub struct PrimitiveGpuVector<H: Hal> {
    ptype: PType,
    len: usize,
    buffer: H::Buffer,
    // validity:
}

// TODO(ngates): BitBuffer to wrap gpu buffer?
pub struct GpuBitBuffer<H: Hal> {
    buffer: H::Buffer,
    offset: usize,
    len: usize,
}
