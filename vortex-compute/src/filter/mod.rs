// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter function.

use std::ops::Deref;

mod bitbuffer;
mod buffer;
mod mask;
mod vector;

/// Function for filtering based on a selection mask.
pub trait Filter<By: ?Sized> {
    /// The result type after performing the operation.
    type Output;

    /// Filters the vector using the provided mask, returning a new value.
    ///
    /// The result value will have length equal to the true count of the provided mask.
    ///
    /// # Panics
    ///
    /// If the length of the mask does not equal the length of the value being filtered.
    fn filter(self, selection: &By) -> Self::Output;
}

/// A view over a set of strictly sorted indices from a bit mask.
///
/// Unlike other indices, `MaskIndices` are always strict-sorted, meaning they are
/// always unique and monotonic.
///
/// You can treat a `MaskIndices` just like a `&[usize]` by iterating or indexing
/// into it just like you would a slice.
pub struct MaskIndices<'a>(&'a [usize]);

impl<'a> MaskIndices<'a> {
    /// Create new indices from a slice of strict-sorted index values.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the indices are strict-sorted, i.e. that they
    /// are monotonic and unique.
    ///
    /// Users of the `Indices` type assume this and failure to uphold this guarantee
    /// can result in UB downstream.
    pub unsafe fn new_unchecked(indices: &'a [usize]) -> Self {
        Self(indices)
    }
}

impl Deref for MaskIndices<'_> {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
