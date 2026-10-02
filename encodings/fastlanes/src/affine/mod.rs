// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::AffineArrayExt;
pub use array::AffineArraySlotsExt;
pub use array::AffineData;
pub use array::AffineSlots;
pub use array::affine_compress::AffineOptions;

#[cfg(test)]
mod tests;

mod plugin;
pub use plugin::AffinePlugin;
pub use plugin::affine_id;

mod vtable;
pub use vtable::Affine;
pub use vtable::AffineArray;
