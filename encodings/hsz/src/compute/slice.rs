// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::compress::HszConfig;
use crate::stage::Hsz;

impl Hsz {
    /// Return a slice of the encoded column as a fresh [`Hsz`].
    ///
    /// The slice is decoded and recompressed against the same `eps`. This
    /// preserves the lossy bound exactly and keeps the bit-packed residual
    /// invariants (each block holds `HSZ_BLOCK_SIZE` packed elements). For
    /// block-aligned slices this is roughly twice the work of a `decompress`
    /// followed by `compress`; for unaligned slices it is the same work and
    /// is required for correctness anyway.
    pub fn slice(&self, range: std::ops::Range<usize>) -> VortexResult<Self> {
        vortex_ensure!(
            range.start <= range.end && range.end <= self.len,
            "slice range {:?} out of bounds for len {}",
            range,
            self.len
        );
        let decoded = self.decompress();
        Hsz::compress(&decoded.as_slice()[range], HszConfig { eps: self.eps })
    }
}
