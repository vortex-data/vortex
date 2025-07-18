// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This crate provides the abstract I/O interface for Vortex. It uses async/await syntax, although
//! please carefully read the documentation for each trait since many of them require
//! runtime-agnostic implementations.
//!
//! To remain runtime-agnostic, many of the implementations in this crate will dispatch I/O tasks
//! onto blocking on async worker pools.

pub use dispatcher::*;
pub use io_buf::*;
pub use limit::*;
#[cfg(feature = "object_store")]
pub use object_store::*;
pub use read::*;
#[cfg(feature = "tokio")]
pub use tokio::*;
pub use write::*;

#[cfg(feature = "compio")]
mod compio;
mod dispatcher;
mod io_buf;
mod limit;
#[cfg(feature = "object_store")]
mod object_store;
mod read;
#[cfg(feature = "tokio")]
mod tokio;
mod write;

/// Required alignment for all custom buffer allocations.
pub const ALIGNMENT: usize = 64;
