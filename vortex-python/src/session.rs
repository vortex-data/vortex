// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::OnceLock;

use vortex::VortexSessionDefault;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Handle;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::SessionExt;
use vortex::session::VortexSession;

use crate::current_runtime;

/// A `OnceLock` rather than a `LazyLock` because initialization is re-entrant: building the session
/// builds the runtime, and building the runtime calls [`reset_session_handle`], which must be able
/// to observe "not initialized yet" without blocking.
static SESSION: OnceLock<VortexSession> = OnceLock::new();

pub(crate) fn session() -> &'static VortexSession {
    SESSION.get_or_init(|| VortexSession::default().with_handle(current_runtime().handle()))
}

/// Point the shared session at `handle`, replacing any previously configured runtime handle.
///
/// A [`Handle`] is a weak reference to its executor, so after the runtime is rebuilt in a forked
/// child (see [`crate::current_runtime`]) the session must be repointed, or every spawn would panic
/// on a dropped runtime. `VortexSession` has interior mutability, so the change is visible through
/// every existing clone of the session.
///
/// Does nothing if the session has not been built yet — in that case it will pick up the current
/// runtime's handle when it is first built.
pub(crate) fn reset_session_handle(handle: Handle) {
    if let Some(session) = SESSION.get() {
        session.session().with_handle(handle);
    }
}
