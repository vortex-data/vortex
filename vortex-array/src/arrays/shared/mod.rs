// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod kernel;
mod vtable;

pub use array::SharedArrayExt;
pub use array::SharedArraySlotsExt;
pub use array::SharedData;
pub use array::SharedSlots;
pub use array::SharedSlotsView;
#[doc(hidden)]
pub use array::current_array_ref_for_dispatch;
pub use vtable::Shared;
pub use vtable::SharedArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    kernel::initialize(session);
}

#[cfg(test)]
mod tests;
