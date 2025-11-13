// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter function.

mod bitbuffer;
mod buffer;
mod mask;
mod vector;

/// Function for filtering based on a selection mask.
pub trait Filter<By> {
    /// The result type after performing the operation.
    type Output;

    /// Filters the vector using the provided structure, returning a new value.
    ///
    /// The result value will have length equal to the true count of the provided mask.
    ///
    /// # Panics
    ///
    /// If the length of the mask does not equal the length of the value being filtered.
    fn filter(self, selection_mask: &By) -> Self::Output;
}
