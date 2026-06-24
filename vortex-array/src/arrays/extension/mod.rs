// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::ExtensionArrayExt;
pub use vtable::ExtensionArray;

pub(crate) mod compute;

mod vtable;
pub use vtable::Extension;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}
