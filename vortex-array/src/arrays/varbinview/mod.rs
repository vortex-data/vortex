// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::VarBinViewArray;
pub use array::VarBinViewArrayParts;

mod accessor;
pub(crate) mod compact;

pub(crate) mod compute;

mod vtable;
pub use vtable::VarBinViewVTable;

pub mod build_views;

// Re-export BinaryView types from vortex-vector
pub use vortex_vector::binaryview::BinaryView;
pub use vortex_vector::binaryview::Inlined;
pub use vortex_vector::binaryview::Ref;

#[cfg(test)]
mod tests;
