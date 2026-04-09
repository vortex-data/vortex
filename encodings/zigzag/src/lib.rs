// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use compress::*;
pub use vtable::*;

mod compress;
mod vtable;

use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize sequence encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(ScalarFnArrayPlugin::new(ZigZag));
}
