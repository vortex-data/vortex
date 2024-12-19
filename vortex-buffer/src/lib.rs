#![feature(unsigned_is_multiple_of)]
#![deny(missing_docs)]

//! A byte buffer implementation for Vortex.
//!
//! Vortex arrays hold data in a set of buffers.
//!
//! # Alignment
//! See: `<https://github.com/spiraldb/vortex/issues/115>`
//!
//! We do not currently enforce any alignment guarantees on the buffer.

extern crate core;

pub use alignment::*;
pub use buffer::*;
pub use buffer_mut::*;
pub use r#const::*;
pub use string::*;

mod alignment;
#[cfg(feature = "arrow")]
mod arrow;
mod buffer;
mod buffer_mut;
mod r#const;
mod string;

/// An immutable buffer of u8.
pub type ByteBuffer = Buffer<u8>;

/// A mutable buffer of u8.
pub type ByteBufferMut = BufferMut<u8>;
