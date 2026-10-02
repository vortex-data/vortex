// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::FL_CHUNK_SIZE;

pub mod affine_compress;
pub mod affine_decompress;

#[array_slots(crate::Affine)]
pub struct AffineSlots {
    /// The residuals left after removing each chunk's model, divided by the chunk's scale.
    #[slot(0)]
    pub encoded: ArrayRef,
    /// One reference per [`FL_CHUNK_SIZE`]-element chunk, of the array's integer type.
    #[slot(1)]
    pub references: ArrayRef,
    /// One scale per chunk, of the array's integer type.
    #[slot(2)]
    pub scales: ArrayRef,
    /// One fixed-point slope per chunk, as `i64` in units of `2^-slope_shift`.
    #[slot(3)]
    pub slopes: ArrayRef,
}

/// Affine-encoded array: a per-chunk linear model plus a scaled residual.
///
/// Element `i` sits at position `p = offset + i`, in chunk `c = p / FL_CHUNK_SIZE` at position
/// `j = p % FL_CHUNK_SIZE`, and decodes as
///
/// ```text
/// value = encoded[i] * scales[c] + references[c] + ((slopes[c] * j) >> slope_shift)
/// ```
///
/// All arithmetic wraps in the array's integer type, except the slope term, which is computed in
/// `i64` with wrapping multiplication and an arithmetic shift before truncating to the array's
/// type. Both encoder and decoder compute the same terms, so the encoding is lossless for any
/// parameters.
///
/// The children select the mode. Constant `scales` of one and constant `slopes` of zero reduce to
/// per-chunk [`FoR`](crate::FoR). A scale above one divides out a common divisor, such as the step
/// of timestamps on a fixed grid. A slope removes a linear trend, such as row ids or regularly
/// sampled timestamps.
#[derive(Clone, Debug)]
pub struct AffineData {
    pub(super) offset: u16,
    pub(super) slope_shift: u8,
}

pub trait AffineArrayExt: AffineArraySlotsExt {
    /// The position of the first element within the first chunk.
    fn offset(&self) -> u16 {
        self.offset
    }

    /// The number of fractional bits in each slope.
    fn slope_shift(&self) -> u8 {
        self.slope_shift
    }

    #[inline]
    fn ptype(&self) -> PType {
        self.as_ref().dtype().as_ptype()
    }
}

impl<T: TypedArrayRef<crate::Affine>> AffineArrayExt for T {}

impl Display for AffineData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "offset: {}, slope_shift: {}",
            self.offset, self.slope_shift
        )
    }
}

/// The largest supported slope shift. Slopes are `i64`, so wider shifts would leave too few
/// integer bits for useful slopes.
pub const MAX_SLOPE_SHIFT: u8 = 32;

impl AffineData {
    pub(crate) fn try_new(offset: u16, slope_shift: u8) -> VortexResult<Self> {
        vortex_ensure!(
            usize::from(offset) < FL_CHUNK_SIZE,
            "Affine offset must be less than {FL_CHUNK_SIZE}, got {offset}"
        );
        vortex_ensure!(
            slope_shift <= MAX_SLOPE_SHIFT,
            "Affine slope shift must be at most {MAX_SLOPE_SHIFT}, got {slope_shift}"
        );
        Ok(Self {
            offset,
            slope_shift,
        })
    }
}

/// The number of chunks spanned by `len` elements starting at `offset` within the first chunk.
pub(crate) fn num_chunks(offset: u16, len: usize) -> usize {
    (usize::from(offset) + len).div_ceil(FL_CHUNK_SIZE)
}

/// The slope term `(slope * j) >> shift` that the model adds at position `j` of a chunk.
#[inline(always)]
pub(crate) fn slope_term(slope: i64, j: usize, shift: u8) -> i64 {
    slope.wrapping_mul(j as i64) >> shift
}
