// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::VarBinViewArrayExt;
pub use array::VarBinViewData;
pub use array::VarBinViewDataParts;
pub use vtable::VarBinViewArray;

mod accessor;
pub(crate) mod compact;

pub(crate) mod compute;

mod vtable;
pub use vtable::VarBinView;

pub(crate) fn initialize(session: &mut vortex_session::VortexSessionBuilder) {
    vtable::initialize(session);
}

pub mod build_views;

mod view;
pub use view::BinaryView;
pub use view::Inlined;
pub use view::Ref;

#[cfg(test)]
mod tests;
