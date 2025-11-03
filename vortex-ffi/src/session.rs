// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::box_wrapper;
use std::sync::LazyLock;
use tokio::runtime;
use tokio::runtime::Runtime;
use vortex::error::VortexExpect;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .vortex_expect("Cannot start runtime")
});

box_wrapper!(
    /// A handle to a Vortex session.
    VortexSession,
    vx_session
);

/// Create a new Vortex session.
///
/// The caller is responsible for freeing the session with [`vx_session_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_new() -> *mut vx_session {
    vx_session::new(Box::new(
        VortexSession::default().with_current_thread_runtime(RUNTIME),
    ))
}
