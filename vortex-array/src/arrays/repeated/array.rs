// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::array::ArrayParts;
use crate::array_slots;
use crate::arrays::Repeated;
use crate::arrays::RepeatedArray;

/// The child slots of a [`RepeatedArray`].
#[array_slots(Repeated)]
pub struct RepeatedSlots {
    /// The array that is repeated.
    #[slot(0)]
    pub child: ArrayRef,
}

impl RepeatedArray {
    /// Creates an array of `len` rows that repeats `child` end to end, without reading any values.
    ///
    /// Row `i` is row `i % child.len()` of `child`, so `len` need not be a multiple of the child's
    /// length: the last repetition is cut short.
    ///
    /// Returns an error if `len` is non-zero and `child` is empty.
    pub fn try_new(child: ArrayRef, len: usize) -> VortexResult<Self> {
        let dtype = child.dtype().clone();
        Self::try_from_parts(ArrayParts::new(
            Repeated,
            dtype,
            len,
            EmptyArrayData,
            RepeatedSlots { child }.into_slots(),
        ))
    }
}
