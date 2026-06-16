// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod read_at;
#[cfg(not(target_arch = "wasm32"))]
mod mmap;

pub use read_at::*;
#[cfg(not(target_arch = "wasm32"))]
pub use mmap::*;
