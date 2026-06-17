// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::FoRArrayExt;
pub use array::FoRData;

pub(crate) mod compute;

mod vtable;
pub use vtable::FoR;
pub use vtable::FoRArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}
