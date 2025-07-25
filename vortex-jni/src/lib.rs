// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex::session::VortexSession;

macro_rules! throw_runtime {
    ($($tt:tt)*) => {
        return Err(vortex::error::vortex_err!($($tt)*).into());
    };
}

mod array;
mod array_iter;
mod dtype;
mod errors;
mod file;
mod logging;

/// Shared Vortex session for the JNI instance.
static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);
