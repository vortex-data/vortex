#![feature(unsigned_is_multiple_of)]
#![deny(missing_docs)]

//! A library for working with compile-time and runtime-aligned buffers.
//!
//! The `vortex-buffer` crate is built around `bytes::Bytes` but is capable of ensuring valid
//! alignment for any sized element type `T`. This means zero-copy cloning, slicing, etc.
//!
//! * `Buffer<T>` and `BufferMut<T>` provide immutable and mutable wrappers around `bytes::Bytes`
//!    and `bytes::BytesMut` respectively.
//! * `ByteBuffer` and `ByteBufferMut` are type aliases for `u8` buffers.
//! * `BufferString` is a wrapper around a `ByteBuffer` that enforces utf-8 encoding.
//! * `ConstBuffer<T, const A: usize>` provides similar functionality to `Buffer<T>` except with a
//!    compile-time alignment of `A`.
//!
//! ## Features
//!
//! The `arrow` feature can be enabled to provide conversion functions to/from Arrow Rust buffers,
//! including `arrow_buffer::Buffer`, `arrow_buffer::ScalarBuffer<T>`, and
//! `arrow_buffer::OffsetBuffer`.

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
