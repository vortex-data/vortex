// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::VortexSessionDefault;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::RUNTIME;
use crate::box_wrapper;

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
    vx_session::new(VortexSession::default().with_handle(RUNTIME.handle()))
}

/// Clone a Vortex session, returning an owned copy.
///
/// The caller is responsible for freeing the session with [`vx_session_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_clone(session: *const vx_session) -> *mut vx_session {
    let session = vx_session::as_ref(session);
    vx_session::new(session.clone())
}

#[cfg(test)]
mod tests {
    use crate::session::vx_session_clone;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_basic() {
        unsafe {
            let session = vx_session_new();
            assert!(!session.is_null());
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_clone() {
        unsafe {
            let session = vx_session_new();
            assert!(!session.is_null());

            let copy = vx_session_clone(session);
            assert!(!copy.is_null());
            vx_session_free(session);

            vx_session_free(copy);
        }
    }
}
