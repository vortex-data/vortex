// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::session::LayoutSession;

/// A test session without a runtime handle configured.
/// Use `test_session(handle)` inside `block_on` closures to get a session with a handle.
pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>()
});

/// Create a test session configured with the provided runtime handle.
/// Use this inside `block_on` closures to get a properly configured session.
pub fn test_session(handle: Handle) -> VortexSession {
    VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>()
        .with_handle(handle)
}
