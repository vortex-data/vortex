// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::N;
use crate::bits::BitVector;

/// Defines a selection of elements from a view or vector.
pub enum Selection {
    /// Select all elements in the vector from zero up to the given length.
    Prefix { len: usize },
    /// The element in the vector to be considered the constant value.
    Constant { element: usize, len: usize },
    /// Select from the vector using a mask, which is a bit array of length `N`.
    Mask(BitVector),
}

/// By default, select all `N` elements in the vector.
impl Default for Selection {
    fn default() -> Self {
        Selection::Prefix { len: N }
    }
}
