// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::VortexSessionDefault;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionMutExt;
use vortex::session::VortexSession;
use vortex::session::VortexSessionRef;

use crate::RUNTIME;
use crate::box_wrapper;

box_wrapper!(
    /// A handle to a Vortex session.
    VortexSessionRef,
    vx_session
);

/// Create a new Vortex session.
///
/// The caller is responsible for freeing the session with [`vx_session_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_new() -> *mut vx_session {
    vx_session::new(Box::new(
        VortexSession::new_with_defaults()
            .with_handle(RUNTIME.handle())
            .freeze(),
    ))
}
