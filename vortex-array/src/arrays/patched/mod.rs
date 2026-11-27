// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A primitive base array with sorted replacement patches.

mod array;
mod compute;
mod indices;
mod vtable;

use std::env;
use std::sync::LazyLock;

pub use array::*;
use vortex_session::VortexSession;

pub(crate) fn initialize(session: &VortexSession) {
    vtable::kernels::initialize(session);
}

/// Flag indicating if experimental patched array support is enabled.
///
/// This is set using the environment variable `VORTEX_EXPERIMENTAL_PATCHED_ARRAY`.
///
/// When enabled, readers externalize interior patches from ALP, BitPacked and
/// integer Sparse arrays into a `Patched` array.
///
/// The builtin compressor will also generate Patched arrays.
pub fn use_experimental_patches() -> bool {
    static USE_EXPERIMENTAL_PATCHES: LazyLock<bool> =
        LazyLock::new(|| env::var("VORTEX_EXPERIMENTAL_PATCHED_ARRAY").is_ok_and(|v| v == "1"));
    *USE_EXPERIMENTAL_PATCHES
}
