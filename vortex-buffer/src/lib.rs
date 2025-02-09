#![feature(unsigned_is_multiple_of)]
#![deny(missing_docs)]

//! A library for working with custom aligned buffers of sized values.
//!
//! The `vortex-buffer` crate is built around `bytes::Bytes` and therefore supports zero-copy
//! cloning and slicing, but differs in that it can define and maintain a custom alignment.
//!
//! * `Buffer<T>` and `BufferMut<T>` provide immutable and mutable wrappers around `bytes::Bytes`
//!    and `bytes::BytesMut` respectively.
//! * `ByteBuffer` and `ByteBufferMut` are type aliases for `u8` buffers.
//! * `BufferString` is a wrapper around a `ByteBuffer` that enforces utf-8 encoding.
//! * `ConstBuffer<T, const A: usize>` provides similar functionality to `Buffer<T>` except with a
//!    compile-time alignment of `A`.
//! * `buffer!` and `buffer_mut!` macros with the same syntax as the builtin `vec!` macro for
//!    inline construction of buffers.
//!
//! You can think of `BufferMut<T>` as similar to a `Vec<T>`, except that any operation that may
//! cause a re-allocation, e.g. extend, will ensure the new allocation maintains the buffer's
//! defined alignment.
//!
//! For example, it's possible to incrementally build a `Buffer<T>` with a 4KB alignment.
//! ```
//! use vortex_buffer::{Alignment, BufferMut};
//!
//! let mut buf = BufferMut::<i32>::empty_aligned(Alignment::new(4096));
//! buf.extend(0i32..1_000);
//! assert_eq!(buf.as_ptr().align_offset(4096), 0)
//! ```
//!
//! ## Comparison
//!
//! | Implementation                   | Zero-copy | Custom Alignment | Typed    |
//! | -------------------------------- | --------- | ---------------- | -------- |
//! | `vortex_buffer::Buffer<T>`       | ✔️        | ✔️               | ✔️       |
//! | `arrow_buffer::ScalarBuffer<T> ` | ✔️        | ❌️️️               | ✔️       |
//! | `bytes::Bytes`                   | ✔️        | ❌️️️               | ❌️️️       |
//! | `Vec<T>`                         | ❌️        | ❌️️               | ✔️       |
//!
//! ## Features
//!
//! The `arrow` feature can be enabled to provide conversion functions to/from Arrow Rust buffers,
//! including `arrow_buffer::Buffer`, `arrow_buffer::ScalarBuffer<T>`, and
//! `arrow_buffer::OffsetBuffer`.

pub use alignment::*;
pub use buffer::*;
pub use buffer_mut::*;
pub use bytes::*;
pub use r#const::*;
pub use string::*;

mod alignment;
#[cfg(feature = "arrow")]
mod arrow;
mod buffer;
mod buffer_mut;
mod bytes;
#[cfg(feature = "compio")]
mod compio;
mod r#const;
mod debug;
mod macros;
#[cfg(feature = "rkyv")]
mod rkyv;
mod string;

/// An immutable buffer of u8.
pub type ByteBuffer = Buffer<u8>;

/// A mutable buffer of u8.
pub type ByteBufferMut = BufferMut<u8>;

/// A const-aligned buffer of u8.
pub type ConstByteBuffer<const A: usize> = ConstBuffer<u8, A>;
