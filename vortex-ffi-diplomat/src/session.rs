// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for the Vortex session handle.
//!
//! In the hand-written C ABI this was a `box_wrapper!`-generated `vx_session` with manual
//! `vx_session_new`, `vx_session_clone`, and `vx_session_free` functions. With Diplomat the
//! opaque type owns its destructor automatically (no `_free`), and `clone` is expressed as an
//! ordinary method returning a new owned handle.

#[diplomat::bridge]
pub mod ffi {
    use std::sync::Arc;

    use vortex::VortexSessionDefault;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::session::VortexSession;

    use crate::RUNTIME;

    /// A handle to a Vortex session.
    ///
    /// A session carries the configuration and async runtime handle used by file IO and scans.
    /// Internally an `Arc<VortexSession>`, so cloning is cheap and shares the underlying state.
    #[diplomat::opaque]
    pub struct VxSession(pub(crate) Arc<VortexSession>);

    impl VxSession {
        /// Create a new Vortex session backed by the shared FFI runtime.
        ///
        /// This is the Diplomat equivalent of `vx_session_new`. The returned handle is owned by
        /// the caller; Diplomat generates the destructor automatically.
        #[diplomat::attr(auto, constructor)]
        pub fn new() -> Box<VxSession> {
            let session = VortexSession::default().with_handle(RUNTIME.handle());
            Box::new(VxSession(Arc::new(session)))
        }

        /// Clone this session, returning a new owned handle that shares the same underlying state.
        ///
        /// Replaces `vx_session_clone` from the C ABI.
        pub fn clone(&self) -> Box<VxSession> {
            Box::new(VxSession(Arc::clone(&self.0)))
        }
    }
}

impl ffi::VxSession {
    /// Borrow the underlying [`vortex::session::VortexSession`].
    ///
    /// A building block for the other FFI bridge modules (file, scan, ...) layered on top of the
    /// session handle. Replaces the C ABI `vx_session_ref` helper; null-checking is unnecessary
    /// because Diplomat guarantees `&self` is a valid reference.
    pub(crate) fn inner(&self) -> &vortex::session::VortexSession {
        &self.0
    }
}
