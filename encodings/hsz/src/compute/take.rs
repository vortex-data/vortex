// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::stage::Hsz;

impl Hsz {
    /// Gather `indices.len()` values without materialising the full column.
    ///
    /// For each index, this performs an `O(1)` residual lookup with an
    /// outlier probe (binary search on a typically tiny outlier list). The
    /// returned [`Buffer`] is in the same order as `indices`.
    pub fn take(&self, indices: &[usize]) -> VortexResult<Buffer<f64>> {
        let mut out = BufferMut::<f64>::with_capacity(indices.len());
        for &idx in indices {
            if idx >= self.len {
                vortex_bail!("take index {} out of bounds for len {}", idx, self.len);
            }
            out.push(self.scalar_at(idx));
        }
        Ok(out.freeze())
    }
}
