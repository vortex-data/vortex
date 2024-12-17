#![feature(vec_into_raw_parts)]
#![deny(missing_docs)]

//! A byte buffer implementation for Vortex.
//!
//! Think of this is an equivalent to arrow-buffer, or Tokio bytes, but has the ability to enforce
//! alignment.

use aligned_buffer::UniqueAlignedBuffer;
pub use buffer::*;
pub use string::*;
#[cfg(feature = "arrow")]
mod arrow;
mod buffer;
mod buffer_mut;
mod flexbuffers;
mod owners;
mod string;

/// The default alignment for Vortex buffers.
pub const DEFAULT_ALIGNMENT: usize = 64;

/// The recommended way to build mutable buffers in Vortex.
pub type VortexBufferMut = UniqueAlignedBuffer<DEFAULT_ALIGNMENT>;
