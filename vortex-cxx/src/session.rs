// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::VortexSessionDefault;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession as RustVortexSession;

use crate::RUNTIME;

pub(crate) struct VortexSession {
    pub(crate) inner: RustVortexSession,
}

pub(crate) fn session_new() -> Box<VortexSession> {
    Box::new(VortexSession {
        inner: RustVortexSession::default().with_handle(RUNTIME.handle()),
    })
}
