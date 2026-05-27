// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector-side helpers: normalization, quantization, and physical storage layout.

pub(crate) mod normalize;
pub(crate) mod quantize;
pub(crate) mod storage;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// Compute the padded SORF dimension for an original vector dimension.
///
/// The SORF transform requires a power-of-two width, so non-power-of-two input dimensions are
/// padded with zeros up to the next power of two. The padded dimension is stored implicitly via
/// [`TurboQuantMetadata::dimensions`](crate::TurboQuantMetadata) plus the codes child's
/// `FixedSizeList` width and recovered at decode time via this function. Returns an error when
/// the next power of two overflows the input integer type.
pub(crate) fn tq_padded_dim(dimensions: u32) -> VortexResult<usize> {
    let padded_dim = dimensions
        .checked_next_power_of_two()
        .ok_or_else(|| vortex_err!("TurboQuant padded dimension overflow for {dimensions}"))?;

    usize::try_from(padded_dim)
        .map_err(|_| vortex_err!("TurboQuant padded dimension does not fit usize"))
}
