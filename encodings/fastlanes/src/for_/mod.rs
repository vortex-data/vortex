// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::FoRArrayExt;
pub use array::FoRArraySlotsExt;
pub use array::FoRData;
pub use array::FoRSlots;

pub(crate) mod compute;

#[cfg(test)]
mod tests;

mod plugin;
pub use plugin::FoRPlugin;
pub use plugin::for_v1_id;
pub use plugin::for_v2_id;

mod vtable;
pub use vtable::FoR;
pub use vtable::FoRArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}
