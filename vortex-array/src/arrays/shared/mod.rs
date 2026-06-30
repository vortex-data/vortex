// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod kernel;
mod vtable;

pub use array::SharedArrayExt;
pub use array::SharedData;
pub use vtable::Shared;
pub use vtable::SharedArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    kernel::initialize(session);
}

#[cfg(test)]
mod tests;
