// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex::VortexSessionDefault;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::RUNTIME;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::default().with_handle(RUNTIME.handle()));

pub(crate) fn session() -> &'static VortexSession {
    &SESSION
}
