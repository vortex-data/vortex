// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::FixedSizeListArrayExt;
pub use array::FixedSizeListArraySlotsExt;
pub use array::FixedSizeListData;
pub use array::FixedSizeListDataParts;
pub use array::FixedSizeListSlots;
pub use array::FixedSizeListSlotsView;
pub use vtable::FixedSizeListArray;

pub(crate) mod compute;

mod vtable;
pub use vtable::FixedSizeList;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
    compute::rules::register_parent_rules(session);
}

#[cfg(test)]
mod tests;
