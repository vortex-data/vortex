// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::FL_CHUNK_SIZE;

pub mod for_compress;
pub mod for_decompress;

#[array_slots(crate::FoR)]
pub struct FoRSlots {
    /// The encoded array with each chunk's reference subtracted.
    #[slot(0)]
    pub encoded: ArrayRef,
    /// One reference per [`FL_CHUNK_SIZE`]-element chunk.
    #[slot(1)]
    pub references: ArrayRef,
}

/// Frame of Reference (FoR) encoded array.
///
/// This encoding stores values as offsets from a reference value, which can significantly reduce
/// storage requirements when values are clustered around a specific point.
///
/// Every [`FL_CHUNK_SIZE`]-element chunk has its own reference: element `i` decodes as
/// `encoded[i] + references[(offset + i) / FL_CHUNK_SIZE]` with wrapping arithmetic, where
/// `offset` is the position of the first element within the first chunk. Arrays with a single
/// reference store a constant `references` child.
#[derive(Clone, Debug)]
pub struct FoRData {
    pub(super) offset: u16,
}

pub trait FoRArrayExt: FoRArraySlotsExt {
    /// The reference shared by every chunk, if the references are constant.
    fn constant_reference(&self) -> Option<Scalar> {
        self.references().as_constant()
    }

    /// The position of the first element within the first chunk of `references`.
    fn offset(&self) -> u16 {
        self.offset
    }

    #[inline]
    fn ptype(&self) -> PType {
        self.as_ref().dtype().as_ptype()
    }
}

impl<T: TypedArrayRef<crate::FoR>> FoRArrayExt for T {}

impl Display for FoRData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}", self.offset)
    }
}

impl FoRData {
    pub(crate) fn try_new(offset: u16) -> VortexResult<Self> {
        vortex_ensure!(
            usize::from(offset) < FL_CHUNK_SIZE,
            "FoR offset must be less than {FL_CHUNK_SIZE}, got {offset}"
        );
        Ok(Self { offset })
    }
}

/// The number of chunks spanned by `len` elements starting at `offset` within the first chunk.
pub(crate) fn num_chunks(offset: u16, len: usize) -> usize {
    (usize::from(offset) + len).div_ceil(FL_CHUNK_SIZE)
}
