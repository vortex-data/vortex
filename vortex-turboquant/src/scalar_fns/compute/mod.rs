// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant-specific session-scoped optimizer kernels.
//!
//! Each kernel module owns its own [`ArrayKernelsExt::register_execute_parent`] call. New
//! kernels (e.g. for `InnerProduct` or `CosineSimilarity`) should be added as sibling modules
//! and threaded through [`register_kernels`] with a single line.

mod l2_norm;

use vortex_session::VortexSession;

/// Register every TurboQuant kernel on `session`.
pub(crate) fn register_kernels(session: &VortexSession) {
    l2_norm::register(session);
}
