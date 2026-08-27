// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

pub mod blocked_for_compress;
pub mod blocked_for_decompress;

/// Number of values covered by a single reference value.
///
/// This matches [`crate::FL_CHUNK_SIZE`] so that a block lines up exactly with a FastLanes
/// bit-packing chunk, which lets decompression fuse the reference addition into unpacking.
pub const BLOCK_SIZE: usize = crate::FL_CHUNK_SIZE;

#[array_slots(crate::BlockedFoR)]
pub struct BlockedFoRSlots {
    /// The encoded array with the block-local frame-of-reference subtracted.
    #[slot(0)]
    pub encoded: ArrayRef,
    /// One reference (minimum) value per [`BLOCK_SIZE`] values of `encoded`.
    #[slot(1)]
    pub references: ArrayRef,
}

/// Block-wise Frame of Reference (FoR) encoded array.
///
/// Where [`crate::FoR`] subtracts a single reference from the whole array, this encoding
/// subtracts a separate reference from each [`BLOCK_SIZE`]-value block. Locally clustered data
/// whose values drift over the array — timestamps, sorted keys, counters — then produce much
/// smaller residuals, so the single bit width that [`crate::BitPacked`] picks for the whole
/// array can be far narrower.
#[derive(Clone, Debug)]
pub struct BlockedFoRData {
    /// The offset within the first block, created by slicing. `0 <= offset < BLOCK_SIZE`.
    pub(super) offset: u16,
}

impl Display for BlockedFoRData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}", self.offset)
    }
}

impl BlockedFoRData {
    pub(crate) fn try_new(offset: u16) -> VortexResult<Self> {
        vortex_ensure!(
            (offset as usize) < BLOCK_SIZE,
            "Offset must be less than the block size {BLOCK_SIZE}, got {offset}"
        );
        Ok(Self { offset })
    }

    /// The offset of the first value within its block.
    #[inline]
    pub fn offset(&self) -> u16 {
        self.offset
    }

    #[inline]
    pub fn ptype(&self, dtype: &DType) -> PType {
        dtype.as_ptype()
    }
}

/// The number of reference values needed to cover `len` values starting at `offset`.
#[inline]
pub(crate) fn num_blocks(len: usize, offset: u16) -> usize {
    (len + offset as usize).div_ceil(BLOCK_SIZE)
}

pub trait BlockedFoRArrayExt: BlockedFoRArraySlotsExt {
    /// The offset of the first value within its block, `0 <= offset < BLOCK_SIZE`.
    #[inline]
    fn offset(&self) -> u16 {
        BlockedFoRData::offset(self)
    }

    /// The number of reference values held by this array.
    #[inline]
    fn num_blocks(&self) -> usize {
        num_blocks(self.as_ref().len(), self.offset())
    }
}

impl<T: TypedArrayRef<crate::BlockedFoR>> BlockedFoRArrayExt for T {}
