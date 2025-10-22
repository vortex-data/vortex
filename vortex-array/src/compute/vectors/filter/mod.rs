// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bitbuffer;
mod bool;
mod buffer;
mod mask;

use vortex_mask::Mask;

/// Trait for filtering vectors based on a selection mask.
pub trait Filter {
    type Mutable;

    /// Filters the vector using the provided mask, returning a new vector.
    fn filter(&self, mask: &Mask) -> Self;

    /// Filters the vector using the provided mask, writing into the given output if possible.
    ///
    /// Note that because implementations _may not_ write into the given output, the caller should
    /// use `with_capacity(0)` rather than attempt to pre-allocate the result using
    /// `mask.true_count()`
    fn filter_into(&self, mask: &Mask, out: Self::Mutable) -> Self;
}
