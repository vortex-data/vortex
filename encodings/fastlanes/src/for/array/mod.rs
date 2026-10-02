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
use vortex_error::vortex_panic;

use crate::FL_CHUNK_SIZE;

pub mod for_compress;
pub mod for_decompress;

#[array_slots(crate::FoR)]
pub struct FoRSlots {
    /// The encoded array with each chunk's reference subtracted.
    #[slot(0)]
    pub encoded: ArrayRef,
    /// One reference per [`FL_CHUNK_SIZE`]-element chunk, unless the array has a global reference.
    #[slot(1)]
    pub blocked_references: Option<ArrayRef>,
}

/// Frame of Reference (FoR) encoded array.
///
/// This encoding stores values as offsets from a reference value, which can significantly reduce
/// storage requirements when values are clustered around a specific point.
///
/// An array has either one global reference, or a reference per [`FL_CHUNK_SIZE`]-element chunk.
/// With a global reference, element `i` decodes as `encoded[i] + global_reference`. With blocked
/// references, it decodes as `encoded[i] + blocked_references[(offset + i) / FL_CHUNK_SIZE]`,
/// where `offset` is the position of the first element within the first chunk. Both use wrapping
/// arithmetic.
#[derive(Clone, Debug)]
pub struct FoRData {
    pub(super) offset: u16,
    pub(super) global_reference: Option<Scalar>,
}

/// The references of a FoR array.
#[derive(Clone, Copy, Debug)]
pub enum FoRReferences<'a> {
    /// One reference for every element.
    Global(&'a Scalar),
    /// One reference per [`FL_CHUNK_SIZE`]-element chunk.
    Blocked(&'a ArrayRef),
}

pub trait FoRArrayExt: FoRArraySlotsExt {
    /// The array's references.
    fn references(&self) -> FoRReferences<'_> {
        match (&self.global_reference, self.blocked_references()) {
            (Some(reference), _) => FoRReferences::Global(reference),
            (None, Some(references)) => FoRReferences::Blocked(references),
            (None, None) => vortex_panic!("FoR array has neither a global nor blocked references"),
        }
    }

    /// The position of the first element within the first chunk of the blocked references. Always
    /// 0 with a global reference.
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
        match &self.global_reference {
            Some(reference) => write!(f, "global_reference: {reference}"),
            None => write!(f, "offset: {}", self.offset),
        }
    }
}

impl FoRData {
    pub(crate) fn try_new(offset: u16, global_reference: Option<Scalar>) -> VortexResult<Self> {
        vortex_ensure!(
            usize::from(offset) < FL_CHUNK_SIZE,
            "FoR offset must be less than {FL_CHUNK_SIZE}, got {offset}"
        );
        vortex_ensure!(
            offset == 0 || global_reference.is_none(),
            "FoR with a global reference must have offset 0, got {offset}"
        );
        Ok(Self {
            offset,
            global_reference,
        })
    }
}

/// The number of chunks spanned by `len` elements starting at `offset` within the first chunk.
pub(crate) fn num_chunks(offset: u16, len: usize) -> usize {
    (usize::from(offset) + len).div_ceil(FL_CHUNK_SIZE)
}
