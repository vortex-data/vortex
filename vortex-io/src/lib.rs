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

pub use io_buf::*;
pub use limit::*;
pub use read_at::*;
pub use write::*;

pub mod compat;
pub mod filesystem;
mod io_buf;
pub mod kanal_ext;
mod limit;
#[cfg(feature = "object_store")]
pub mod object_store;
mod read_at;
pub mod runtime;
pub mod session;
#[cfg(not(target_arch = "wasm32"))]
pub mod std_file;
#[cfg(feature = "tokio")]
mod tokio;
mod write;
