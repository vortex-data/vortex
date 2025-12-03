// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod buffer;
mod driver;
#[cfg(feature = "object_store")]
pub mod object_store;
mod read;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod std_file;
#[cfg(all(target_os = "linux", feature = "uring"))]
pub(crate) mod uring_file;

pub(crate) use driver::*;
pub use read::*;
