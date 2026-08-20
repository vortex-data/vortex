// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical map arrays backed by [`ListView`](crate::arrays::ListView) entry storage.

mod array;
pub use array::MapArrayExt;
pub use array::MapArraySlotsExt;
pub use array::MapData;
pub use array::MapDataParts;
pub use array::MapSlots;
pub use array::MapSlotsView;

pub(crate) mod compute;

mod vtable;
pub use vtable::Map;
pub use vtable::MapArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}

#[cfg(test)]
mod tests;
