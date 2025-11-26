// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take function.

mod buffer;
pub mod slice;

/// Function for taking based on indices (which can have different representations).
pub trait Take<Indices: ?Sized> {
    /// The result type after performing the operation.
    type Output;

    /// Creates a new object using the elements from the input indexed by `indices`.
    ///
    /// For example, if we have an array `[1, 2, 3, 4, 5]` and `indices` `[4, 2]`, the resulting
    /// array would be `[5, 3]`.
    ///
    /// The output should have the same length as the `indices`.
    ///
    /// # Panics
    ///
    /// This should panic if an index in `indices` is out-of-bounds with respect to `self`.
    fn take(self, indices: &Indices) -> Self::Output;
}
