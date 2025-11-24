// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter function.

mod bitbuffer;
mod buffer;
mod mask;
pub mod slice;
mod vector;

/// Function for filtering based on a selection mask.
pub trait Filter<Selection: ?Sized> {
    /// The result type after performing the operation.
    type Output;

    /// Filters an object using the provided mask, returning a new value.
    ///
    /// The result value will have length equal to the true count of the provided mask.
    ///
    /// # Panics
    ///
    /// If the length of the mask does not equal the length of the value being filtered.
    fn filter(self, selection: &Selection) -> Self::Output;
}
