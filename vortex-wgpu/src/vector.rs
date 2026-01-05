// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::PType;

pub enum GpuVector<B> {
    Primitive(PrimitiveGpuVector<B>),
}

pub struct PrimitiveGpuVector<B> {
    pub ptype: PType,
    pub len: usize,
    pub buffer: B,
    // validity
}
