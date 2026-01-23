// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::VarBinViewArray;
pub use array::VarBinViewArrayParts;

mod accessor;
pub mod compact;

mod compute;

mod vtable;
pub use vtable::VarBinViewVTable;

pub mod build_views;
pub use build_views::BinaryView;
pub use build_views::Inlined;
pub use build_views::Ref;

#[cfg(test)]
mod tests;
