// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::VarBinViewArray;

mod accessor;
pub(crate) mod compact;

pub mod binary_view;

mod compute;

mod vtable;
pub use vtable::{VarBinViewEncoding, VarBinViewVTable};

#[cfg(test)]
mod tests;
