// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::MaskedData;
pub use vtable::MaskedArray;

pub(crate) mod compute;
mod execute;

pub use execute::mask_validity_canonical;

mod vtable;
pub use vtable::Masked;

#[cfg(test)]
mod tests;
