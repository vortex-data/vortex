// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter function.

mod bitbuffer;
mod bool;
mod buffer;
mod mask;

use vortex_mask::Mask;

/// Function for filtering based on a selection mask.
pub trait Filter {
    /// Filters the vector using the provided mask, returning a new value.
    ///
    /// The result value will have length equal to the true count of the provided mask.
    ///
    /// # Panics
    ///
    /// If the length of the mask does not equal the length of the value being filtered.
    fn filter(&self, mask: &Mask) -> Self;
}
