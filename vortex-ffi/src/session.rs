// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_int;

use vortex::VortexSessionDefault;
use vortex::editions::EditionSessionExt;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::RUNTIME;
use crate::box_wrapper;
use crate::error::try_or;
use crate::error::vx_error;
use crate::string::vx_view;

box_wrapper!(
    /// A handle to a Vortex session.
    VortexSession,
    vx_session
);

/// Create an FFI session from a configured default session.
pub fn vx_session_new_with(
    configure: impl FnOnce(VortexSession) -> VortexSession,
) -> *mut vx_session {
    vx_session::new(configure(
        VortexSession::default().with_handle(RUNTIME.handle()),
    ))
}

/// Create a new Vortex session.
///
/// The caller is responsible for freeing the session with [`vx_session_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_new() -> *mut vx_session {
    vx_session_new_with(|session| session)
}

/// Clone a vx_session
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_clone(session: *const vx_session) -> *mut vx_session {
    let session = vx_session::as_ref(session);
    vx_session::new(session.clone())
}

/// Enable a registered edition by its UTF-8 name, such as `core2025.10.0`.
///
/// This replaces the enabled edition of the same family on the shared session. Call this
/// before opening a file sink. Use separate sessions for overlapping writes with different editions.
///
/// Returns 0 on success and 1 on failure. If `error_out` is non-NULL, `*error_out` is cleared on
/// success and set to an error on failure. Free that error with `vx_error_free`.
///
/// # Safety
///
/// `session` must be NULL or a live session handle created by this library. `name.ptr` must be
/// readable for `name.len` bytes, or NULL when `name.len` is zero. If non-NULL, `error_out` must be
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_enable_edition(
    session: *const vx_session,
    name: vx_view,
    error_out: *mut *mut vx_error,
) -> c_int {
    try_or(error_out, 1, || {
        // SAFETY: The caller provides a live session handle or a null pointer.
        let session = unsafe { vx_session_ref(session) }?;
        // SAFETY: The caller provides a readable byte range for the duration of this call.
        let name = unsafe { name.as_str() }?;
        let edition = session
            .editions()
            .find_by_name(name)
            .ok_or_else(|| vortex_err!("Unknown Vortex edition: {name}"))?;
        session.enable_edition(edition.id)?;
        Ok(0)
    })
}

/// Borrow the [`VortexSession`] behind a [`vx_session`] handle, erroring on a null pointer.
///
/// A building block for FFI crates layered on top of the base Vortex C API.
///
/// # Safety
///
/// `session` must be null or a valid `vx_session` pointer created by this crate, and must stay
/// valid for the returned reference.
pub unsafe fn vx_session_ref<'a>(session: *const vx_session) -> VortexResult<&'a VortexSession> {
    vortex_ensure!(!session.is_null(), "null vx_session");
    Ok(vx_session::as_ref(session))
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use vortex::editions::CORE_2025_10_0;

    use super::*;
    use crate::error::vx_error_free;
    use crate::error::vx_error_message;

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

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_enable_edition() -> VortexResult<()> {
        let session = unsafe { vx_session_new() };
        let copy = unsafe { vx_session_clone(session) };
        let mut error = ptr::null_mut();

        let name = vx_view::from_str("core2025.10.0");
        assert_eq!(
            unsafe { vx_session_enable_edition(session, name, &raw mut error) },
            0
        );
        assert!(error.is_null());

        let unknown_name = vx_view::from_str("unknown-edition");
        assert_eq!(
            unsafe { vx_session_enable_edition(session, unknown_name, &raw mut error) },
            1
        );
        assert!(!error.is_null());
        let message = unsafe { vx_error_message(error).as_str() }?;
        assert!(message.contains("Unknown Vortex edition: unknown-edition"));
        unsafe { vx_error_free(error) };

        let core_editions: Vec<_> = unsafe { vx_session_ref(copy) }?
            .enabled_editions()
            .editions()
            .into_iter()
            .filter(|edition| edition.family == CORE_2025_10_0.family)
            .collect();
        assert_eq!(core_editions, [CORE_2025_10_0]);
        unsafe {
            vx_session_free(copy);
            vx_session_free(session);
        }
        Ok(())
    }
}
