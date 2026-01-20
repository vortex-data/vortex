// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::VarBinViewArray;
pub use array::VarBinViewArrayParts;

mod accessor;
pub(crate) mod compact;

mod compute;

mod vtable;
pub use vtable::VarBinViewVTable;

pub mod build_views;
#[cfg(test)]
mod tests;
