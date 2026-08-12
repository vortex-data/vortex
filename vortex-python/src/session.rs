// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(unix)]
use std::ptr;
#[cfg(not(unix))]
use std::sync::LazyLock;
#[cfg(unix)]
use std::sync::atomic::AtomicPtr;
#[cfg(unix)]
use std::sync::atomic::Ordering;

use vortex::VortexSessionDefault;
use vortex::io::runtime::BlockingRuntime;
#[cfg(unix)]
use vortex::io::runtime::Handle;
use vortex::io::session::RuntimeSessionExt;
#[cfg(unix)]
use vortex::session::SessionExt;
use vortex::session::VortexSession;

use crate::current_runtime;

#[cfg(not(unix))]
static SESSION: LazyLock<VortexSession> = LazyLock::new(new_session);

/// The shared session is published without an initialization lock so a forked child cannot inherit
/// it in a permanently initializing state.
#[cfg(unix)]
static SESSION: AtomicPtr<VortexSession> = AtomicPtr::new(ptr::null_mut());

#[cfg(not(unix))]
pub(crate) fn session() -> &'static VortexSession {
    &SESSION
}

#[cfg(unix)]
pub(crate) fn session() -> &'static VortexSession {
    // Ensure a forked child has published its new runtime and repointed an existing session before
    // returning that session to a caller.
    let runtime = current_runtime();
    loop {
        let current = SESSION.load(Ordering::Acquire);
        if !current.is_null() {
            // SAFETY: A published session is never freed or replaced.
            return unsafe { &*current };
        }

        let fresh = Box::into_raw(Box::new(
            VortexSession::default().with_handle(runtime.handle()),
        ));
        match SESSION.compare_exchange(current, fresh, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                // SAFETY: `fresh` was just published and published sessions are never freed.
                return unsafe { &*fresh };
            }
            Err(_) => {
                // SAFETY: The failed compare-exchange proves `fresh` was never published.
                drop(unsafe { Box::from_raw(fresh) });
            }
        }
    }
}

#[cfg(not(unix))]
fn new_session() -> VortexSession {
    VortexSession::default().with_handle(current_runtime().handle())
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
#[cfg(unix)]
pub(crate) fn reset_session_handle(handle: Handle) {
    let current = SESSION.load(Ordering::Acquire);
    if !current.is_null() {
        // SAFETY: A published session is never freed or replaced.
        let session = unsafe { &*current };
        session.session().with_handle(handle);
    }
}
