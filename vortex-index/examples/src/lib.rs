// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! **Experimental proof of concept.** Examples, a benchmark and cross-kind tests of the index
//! crates under `vortex-index/`: the shared `vortex-index` crate and one crate per index kind.

use vortex_error::VortexResult;
use vortex_session::VortexSession;

/// Register every index kind, the `text_match` function, and the edition for writing
/// `vortex.indexed` layouts.
pub fn initialize(session: &VortexSession) -> VortexResult<()> {
    vortex_index::initialize(session)?;
    vortex_index_trigram::register(session);
    vortex_index_inverted::register(session);
    vortex_index_ivf::register(session);
    Ok(())
}
