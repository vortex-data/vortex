// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::Handle;

/// Drives a future on the current thread, parking while no task is runnable.
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    smol::block_on(future)
}

/// There is no ambient runtime to discover: callers install a handle themselves.
pub(crate) fn default_handle() -> Option<Handle> {
    None
}
