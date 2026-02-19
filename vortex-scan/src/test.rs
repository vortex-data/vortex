// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

pub fn new_session() -> VortexSession {
    VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>()
}

pub fn session_with_handle(handle: Handle) -> VortexSession {
    new_session().with_handle(handle)
}

pub static SCAN_SESSION: LazyLock<VortexSession> = LazyLock::new(new_session);
