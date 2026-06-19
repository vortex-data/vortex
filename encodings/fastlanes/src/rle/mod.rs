// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::RLEArrayExt;
pub use array::RLEData;

mod compute;
mod kernel;

mod vtable;
pub use vtable::RLE;
pub use vtable::RLEArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    kernel::initialize(session);
}
