//! Core traits and implementations for asynchronous IO.
//!
//! Vortex implements an IPC streaming format as well as a file format, both of which
//! run on top of a variety of storage systems that can be accessed from multiple async
//! runtimes.
//!
//! This crate provides core traits for positioned and streaming IO, and via feature
//! flags implements the core traits for several common async runtimes and backing stores.

#[cfg(feature = "futures")]
pub use futures::*;
#[cfg(feature = "object_store")]
pub use object_store::*;
pub use read::*;
#[cfg(feature = "tokio")]
pub use tokio::*;
pub use write::*;

#[cfg(feature = "compio")]
mod compio;
#[cfg(feature = "futures")]
mod futures;
#[cfg(feature = "object_store")]
mod object_store;
pub mod offset;
mod read;
#[cfg(feature = "tokio")]
mod tokio;
mod write;
