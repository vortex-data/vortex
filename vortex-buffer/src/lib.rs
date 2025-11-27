// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]
// cudarc HostSlice has len and is_empty methods that duplicate BufferMut methods.
#![allow(clippy::same_name_method)]

//! A library for working with custom aligned buffers of sized values.
//!
//! The `vortex-buffer` crate is built around `bytes::Bytes` and therefore supports zero-copy
//! cloning and slicing, but differs in that it can define and maintain a custom alignment.
//!
//! * `Buffer<T>` and `BufferMut<T>` provide immutable and mutable wrappers around `bytes::Bytes`
//!   and `bytes::BytesMut` respectively.
//! * `ByteBuffer` and `ByteBufferMut` are type aliases for `u8` buffers.
//! * `BufferString` is a wrapper around a `ByteBuffer` that enforces utf-8 encoding.
//! * `ConstBuffer<T, const A: usize>` provides similar functionality to `Buffer<T>` except with a
//!   compile-time alignment of `A`.
//! * `buffer!` and `buffer_mut!` macros with the same syntax as the builtin `vec!` macro for
//!   inline construction of buffers.
//! * `BitBuffer` and `BitBufferMut` provide packed bitsets that can be used to store boolean values.
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
pub use bit::*;
pub use buffer::*;
pub use buffer_mut::*;
pub use bytes::*;
pub use r#const::*;
pub use string::*;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

mod alignment;
#[cfg(feature = "arrow")]
mod arrow;
mod bit;
mod buffer;
mod buffer_mut;
mod bytes;
mod r#const;
#[cfg(gpu_unstable)]
mod cuda;
mod debug;
mod macros;
#[cfg(feature = "memmap2")]
mod memmap2;
#[cfg(feature = "serde")]
mod serde;
mod string;
mod trusted_len;

/// An immutable buffer of u8.
pub type ByteBuffer = Buffer<u8>;

/// A mutable buffer of u8.
pub type ByteBufferMut = BufferMut<u8>;

/// A const-aligned buffer of u8.
pub type ConstByteBuffer<const A: usize> = ConstBuffer<u8, A>;

#[derive(Debug, Clone)]
/// A buffer can be either on the CPU or on an attached device (e.g. GPU).
/// The Device implementation will come later.
pub enum BufferHandle {
    /// On the host/cpu.
    Buffer(ByteBuffer),
    /// On the device.
    // TODO: impl this.
    DeviceBuffer,
}

impl BufferHandle {
    /// Fetches the cpu buffer and fails otherwise.
    pub fn bytes(&self) -> &ByteBuffer {
        match self {
            BufferHandle::Buffer(b) => b,
            BufferHandle::DeviceBuffer => todo!(),
        }
    }

    /// Fetches the cpu buffer and fails otherwise.
    pub fn into_bytes(self) -> ByteBuffer {
        match self {
            BufferHandle::Buffer(b) => b,
            BufferHandle::DeviceBuffer => todo!(),
        }
    }

    /// Attempts to convert this handle into a CPU ByteBuffer.
    /// Returns an error if the buffer is on a device.
    pub fn try_to_bytes(self) -> VortexResult<ByteBuffer> {
        match self {
            BufferHandle::Buffer(b) => Ok(b),
            BufferHandle::DeviceBuffer => vortex_bail!("cannot move device_buffer to buffer"),
        }
    }
}
