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

pub use aligned_const::*;
pub use alignment::*;
pub use scalar::*;
pub use scalar_mut::*;
pub use string::*;

mod aligned_const;
mod alignment;
#[cfg(feature = "arrow")]
mod arrow;
mod scalar;
mod scalar_mut;
mod string;

/// An immutable buffer of u8.
pub type ByteBuffer = ScalarBuffer<u8>;

/// A mutable buffer of u8.
pub type ByteBufferMut = ScalarBufferMut<u8>;
