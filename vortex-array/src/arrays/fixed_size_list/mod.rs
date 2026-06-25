// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::FixedSizeListArrayExt;
pub use array::FixedSizeListData;
pub use array::FixedSizeListDataParts;
pub use vtable::FixedSizeListArray;

pub(crate) mod compute;

mod vtable;
pub use vtable::FixedSizeList;

pub(crate) fn initialize(session: &mut vortex_session::VortexSessionBuilder) {
    vtable::initialize(session);
}

#[cfg(test)]
mod tests;
