// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::MaskedArrayExt;
pub use array::MaskedArraySlotsExt;
pub use array::MaskedData;
pub use array::MaskedSlots;
pub use vtable::MaskedArray;

pub(crate) mod compute;
mod execute;

pub use execute::mask_validity_canonical;

mod vtable;
pub use vtable::Masked;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod pure_data_tests;
