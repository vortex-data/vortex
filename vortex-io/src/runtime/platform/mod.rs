// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Target-specific pieces of the runtime, so that the rest of the module can stay `cfg`-free.
//!
//! Each platform module exposes the same two items:
//! * `block_on`, which drives a future to completion on the current thread.
//! * `default_handle`, the runtime [`Handle`](crate::runtime::Handle) to use when the caller has
//!   not installed one.

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::*;

#[cfg(target_arch = "wasm32")]
mod wasm32;
#[cfg(target_arch = "wasm32")]
pub(crate) use wasm32::*;

// Compiled off WebAssembly too, so that its behaviour is covered by the test suite on every target.
#[cfg(any(test, target_arch = "wasm32"))]
mod spin;
