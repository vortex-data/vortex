// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_session::VortexSession;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);
