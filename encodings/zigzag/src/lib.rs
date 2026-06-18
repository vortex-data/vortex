// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use array::*;
pub use compress::*;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

mod array;
mod compress;
mod compute;
mod kernel;
mod rules;
mod slice;

/// Initialize zigzag encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(ZigZag);
    kernel::initialize(session);
}
