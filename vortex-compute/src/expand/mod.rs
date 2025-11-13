// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expand function.

mod buffer;

use vortex_mask::Mask;

/// Function for expanding values of `self` to the true positions of a mask.
pub trait Expand {
    /// The result type after expansion.
    type Output: Default;

    /// Expands `self` using the provided mask.
    ///
    ///
    /// The result will have length equal to the mask. All values of `self` are
    /// scattered to the true positions of the mask. False positions are set to
    /// `Output::default`.
    ///
    ///
    /// # Panics
    ///
    /// Panics if the number of true count of the mask does not equal the length of `self`.
    fn expand(self, mask: &Mask) -> Self::Output;
}
