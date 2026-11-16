// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::array::ArrayParts;
use crate::array_slots;
use crate::arrays::Tiled;
use crate::arrays::TiledArray;

/// The child slots of a [`TiledArray`].
#[array_slots(Tiled)]
pub struct TiledSlots {
    /// The array that is repeated.
    #[slot(0)]
    pub child: ArrayRef,
}

/// The data of a [`TiledArray`]: how many times its child is repeated.
#[derive(Clone, Debug)]
pub struct TiledData {
    pub(super) count: usize,
}

impl Display for TiledData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "count: {}", self.count)
    }
}

impl TiledData {
    /// Returns the number of times the child is repeated.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Returns the length of `count` copies of an array of `child_len` rows.
pub(super) fn tiled_len(child_len: usize, count: usize) -> VortexResult<usize> {
    child_len.checked_mul(count).ok_or_else(|| {
        vortex_err!("Tiled length overflows usize: {child_len} rows repeated {count} times")
    })
}

impl TiledArray {
    /// Creates an array that repeats `child` end to end `count` times, without reading any values.
    ///
    /// Returns an error if the length of the result overflows `usize`.
    pub fn try_new(child: ArrayRef, count: usize) -> VortexResult<Self> {
        let len = tiled_len(child.len(), count)?;
        let dtype = child.dtype().clone();
        Self::try_from_parts(ArrayParts::new(
            Tiled,
            dtype,
            len,
            TiledData { count },
            TiledSlots { child }.into_slots(),
        ))
    }
}
