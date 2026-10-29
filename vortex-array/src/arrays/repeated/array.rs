// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
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

/// The data of a [`RepeatedArray`]: how many times its child is repeated.
#[derive(Clone, Debug)]
pub struct RepeatedData {
    pub(super) count: usize,
}

impl Display for RepeatedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "count: {}", self.count)
    }
}

impl RepeatedData {
    /// Returns the number of times the child is repeated.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Returns the length of `count` copies of an array of `child_len` rows.
pub(super) fn repeated_len(child_len: usize, count: usize) -> VortexResult<usize> {
    child_len.checked_mul(count).ok_or_else(|| {
        vortex_err!("Repeated length overflows usize: {child_len} rows repeated {count} times")
    })
}

impl RepeatedArray {
    /// Creates an array that repeats `child` end to end `count` times, without reading any values.
    ///
    /// Returns an error if the length of the result overflows `usize`.
    pub fn try_new(child: ArrayRef, count: usize) -> VortexResult<Self> {
        let len = repeated_len(child.len(), count)?;
        let dtype = child.dtype().clone();
        Self::try_from_parts(ArrayParts::new(
            Repeated,
            dtype,
            len,
            RepeatedData { count },
            RepeatedSlots { child }.into_slots(),
        ))
    }
}
