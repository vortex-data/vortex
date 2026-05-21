// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant-specific session-scoped optimizer kernels.
//!
//! Each kernel module owns its own
//! [`register_execute_parent`](vortex_array::optimizer::kernels::ArrayKernelsExt::register_execute_parent)
//! call. New kernels (for example `InnerProduct` or `CosineSimilarity`) should be added as
//! sibling modules and threaded through [`register_kernels`] with a single line.

mod l2_norm;

use vortex_session::VortexSession;

/// Register every TurboQuant-specific optimizer kernel on `session`.
///
/// Called from the crate-level [`crate::initialize`] after the TurboQuant extension type and
/// the [`crate::TQEncode`] / [`crate::TQDecode`] scalar functions are registered, so kernels
/// can resolve the scalar-fn ids they intercept.
pub(crate) fn register_kernels(session: &VortexSession) {
    l2_norm::register(session);
}
