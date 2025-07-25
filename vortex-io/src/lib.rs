// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core traits and implementations for asynchronous IO.
//!
//! Vortex implements an IPC streaming format as well as a file format, both of which
//! run on top of a variety of storage systems that can be accessed from multiple async
//! runtimes.
//!
//! This crate provides core traits for positioned and streaming IO, and via feature
//! flags implements the core traits for several common async runtimes and backing stores.

use ::std::sync::Arc;
pub use io_buf::*;
pub use limit::*;
#[cfg(feature = "object_store")]
pub use object_store::*;
pub use read::*;
pub use read_at::*;
use vortex_error::VortexResult;

#[cfg(feature = "tokio")]
pub mod tokio;
pub use write::*;

mod buffer;
#[cfg(feature = "compio")]
mod compio;
pub mod dispatcher;
mod io_buf;
mod limit;
#[cfg(feature = "object_store")]
mod object_store;
mod read;
mod read_at;
mod std;
mod write;

/// Required alignment for all custom buffer allocations.
pub const ALIGNMENT: usize = 64;

/// A trait for converting supported I/O objects into Vortex I/O objects.
pub trait VortexIO {
    fn performance_hint(&self) -> PerformanceHint;

    /// Load the current object into a Vortex `ReadAt` I/O object.
    fn into_read_at(self) -> VortexResult<Arc<dyn ReadAt>>;
}
