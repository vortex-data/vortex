// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Process-wide [`VortexSession`] used by the benchmark.
//!
//! TurboQuant arrays only round-trip through a Vortex file when the tensor scalar-fn array
//! plugins have been registered on the session. Registration is gated on
//! [`vortex_tensor::SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV`] — the env var must be set **before**
//! [`vortex_tensor::initialize`] runs. Both happen inside the [`SESSION`]
//! `LazyLock` initializer, so the order is guaranteed for any caller that touches the
//! session through this module.

use std::sync::LazyLock;

use vortex::VortexSessionDefault;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

/// The single shared Vortex session for the benchmark.
///
/// Initialization, in order:
///
/// 1. Set [`vortex_tensor::SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV`] so [`vortex_tensor::initialize`]
///    will register the tensor scalar-fn array plugins.
/// 2. Build a default session attached to the running Tokio runtime.
/// 3. Run [`vortex_tensor::initialize`] to register the tensor types and scalar functions.
pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    // SAFETY: called from inside the LazyLock initializer, before any other access to
    // `SESSION`. The first thread to dereference SESSION runs this once.
    unsafe {
        std::env::set_var(vortex_tensor::SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV, "1");
    }
    let session = VortexSession::default().with_tokio();
    vortex_tensor::initialize(&session);
    session
});
