// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::decimal::precision::PrecisionScale;
use vortex_buffer::BufferMut;
use vortex_mask::MaskMut;

/// A specifically typed mutable decimal vector.
#[derive(Debug, Clone)]
pub struct DVectorMut<D> {
    ps: PrecisionScale<D>,
    elements: BufferMut<D>,
    validity: MaskMut,
}
