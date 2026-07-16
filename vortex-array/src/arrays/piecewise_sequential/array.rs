// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::smallvec;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::EmptyArrayData;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::PiecewiseSequential;
use crate::dtype::PType;

#[array_slots(PiecewiseSequential)]
pub struct PiecewiseSequentialSlots {
    /// The inclusive start index of each sequential piece.
    pub starts: ArrayRef,
    /// The length of each sequential piece.
    pub lengths: ArrayRef,
}

/// Extension methods for [`PiecewiseSequentialArray`].
pub trait PiecewiseSequentialArrayExt:
    TypedArrayRef<PiecewiseSequential> + PiecewiseSequentialArraySlotsExt
{
}
impl<T: TypedArrayRef<PiecewiseSequential>> PiecewiseSequentialArrayExt for T {}

impl Array<PiecewiseSequential> {
    /// Constructs a new `PiecewiseSequentialArray` from start and length arrays.
    ///
    /// This validates only structural invariants: both children must be non-nullable unsigned
    /// integer arrays with matching lengths, and the outer array length is the declared expanded
    /// index length. Individual ranges are checked when the index array is executed or consumed by
    /// a take implementation.
    pub fn try_new(starts: ArrayRef, lengths: ArrayRef, len: usize) -> VortexResult<Self> {
        Array::try_from_parts(
            ArrayParts::new(PiecewiseSequential, PType::U64.into(), len, EmptyArrayData)
                .with_slots(smallvec![Some(starts), Some(lengths)]),
        )
    }

    /// Constructs a new `PiecewiseSequentialArray` without validation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee the same structural invariants as [`Self::try_new`].
    pub unsafe fn new_unchecked(starts: ArrayRef, lengths: ArrayRef, len: usize) -> Self {
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(PiecewiseSequential, PType::U64.into(), len, EmptyArrayData)
                    .with_slots(smallvec![Some(starts), Some(lengths)]),
            )
        }
    }
}
