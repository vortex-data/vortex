// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryMaskedArray;

mod array;
pub use array::MaskedArray;

pub(crate) mod compute;
mod execute;

pub use execute::mask_validity_canonical;

mod vtable;
pub use vtable::MaskedVTable;

#[cfg(test)]
mod tests;
