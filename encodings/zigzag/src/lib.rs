// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use array::*;
pub use compress::*;
use vortex_array::session::ArraySession;
use vortex_session::VortexSessionBuilder;

mod array;
mod compress;
mod compute;
mod kernel;
mod rules;
mod slice;

/// Initialize zigzag encoding in the given session.
pub fn initialize(session: &mut VortexSessionBuilder) {
    session.get_mut::<ArraySession>().register(ZigZag);
    kernel::initialize(session);
}
