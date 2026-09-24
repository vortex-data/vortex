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
use vortex::compressor::COMPACT_SCHEMES;
use vortex::compressor::CompressionSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Handle;
use vortex::io::session::RuntimeSessionExt;
#[cfg(unix)]
use vortex::session::SessionExt;
use vortex::session::VortexSession;

use crate::current_runtime;

#[cfg(not(unix))]
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| new_session(current_runtime().handle()));
#[cfg(not(unix))]
static COMPACT_SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| new_compact_session(current_runtime().handle()));

/// The shared sessions are published without an initialization lock so a forked child cannot
/// inherit them in a permanently initializing state.
#[cfg(unix)]
static SESSION: AtomicPtr<VortexSession> = AtomicPtr::new(ptr::null_mut());
#[cfg(unix)]
static COMPACT_SESSION: AtomicPtr<VortexSession> = AtomicPtr::new(ptr::null_mut());

#[cfg(not(unix))]
pub(crate) fn session() -> &'static VortexSession {
    &SESSION
}

/// The shared session plus the compact (Zstd and Pco) schemes, for compact writes.
#[cfg(not(unix))]
pub(crate) fn compact_session() -> &'static VortexSession {
    &COMPACT_SESSION
}

#[cfg(unix)]
pub(crate) fn session() -> &'static VortexSession {
    published(&SESSION, new_session)
}

/// The shared session plus the compact (Zstd and Pco) schemes, for compact writes.
#[cfg(unix)]
pub(crate) fn compact_session() -> &'static VortexSession {
    published(&COMPACT_SESSION, new_compact_session)
}

/// Returns the session published in `slot`, building it with `new` on first use.
#[cfg(unix)]
fn published(
    slot: &AtomicPtr<VortexSession>,
    new: fn(Handle) -> VortexSession,
) -> &'static VortexSession {
    // Ensure a forked child has published its new runtime and repointed an existing session before
    // returning that session to a caller.
    let runtime = current_runtime();
    loop {
        let current = slot.load(Ordering::Acquire);
        if !current.is_null() {
            // SAFETY: A published session is never freed or replaced.
            return unsafe { &*current };
        }

        let fresh = Box::into_raw(Box::new(new(runtime.handle())));
        match slot.compare_exchange(current, fresh, Ordering::AcqRel, Ordering::Acquire) {
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

fn new_session(handle: Handle) -> VortexSession {
    VortexSession::default().with_handle(handle)
}

fn new_compact_session(handle: Handle) -> VortexSession {
    let session = new_session(handle);
    for scheme in COMPACT_SCHEMES {
        session.register_scheme(*scheme);
    }
    session
}

/// Point the shared sessions at `handle`, replacing any previously configured runtime handle.
///
/// A [`Handle`] is a weak reference to its executor, so after the runtime is rebuilt in a forked
/// child (see [`crate::current_runtime`]) the sessions must be repointed, or every spawn would
/// panic on a dropped runtime. `VortexSession` has interior mutability, so the change is visible
/// through every existing clone of a session.
///
/// Does nothing for a session that has not been built yet — it will pick up the current runtime's
/// handle when it is first built.
#[cfg(unix)]
pub(crate) fn reset_session_handle(handle: Handle) {
    for slot in [&SESSION, &COMPACT_SESSION] {
        let current = slot.load(Ordering::Acquire);
        if !current.is_null() {
            // SAFETY: A published session is never freed or replaced.
            let session = unsafe { &*current };
            session.session().with_handle(handle.clone());
        }
    }
}
