// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionBuilderExt;
use vortex_session::VortexSession;

use crate::session::LayoutSession;

pub fn new_session() -> VortexSession {
    vortex_array::default_session_builder()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .build()
}

pub fn session_with_handle(handle: Handle) -> VortexSession {
    vortex_array::default_session_builder()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_handle(handle)
        .build()
}

pub static SCAN_SESSION: LazyLock<VortexSession> = LazyLock::new(new_session);
