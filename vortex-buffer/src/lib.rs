#![deny(missing_docs)]

//! A byte buffer implementation for Vortex.
//!
//! Vortex arrays hold data in a set of buffers.
//!
//! # Alignment
//! See: `<https://github.com/spiraldb/vortex/issues/115>`
//!
//! We do not currently enforce any alignment guarantees on the buffer.

use core::cmp::Ordering;
use core::ops::{Deref, Range};

use arrow_buffer::{ArrowNativeType, Buffer as ArrowBuffer, MutableBuffer as ArrowMutableBuffer};
pub use string::*;

mod flexbuffers;
pub mod io_buf;
mod string;

/// Buffer is an owned, cheaply cloneable byte array.
///
/// Buffers form the building blocks of all in-memory storage in Vortex.
#[derive(Debug, Clone)]
pub struct Buffer(Inner);

#[derive(Debug, Clone)]
enum Inner {
    // TODO(ngates): we could add Aligned(Arc<AVec>) from aligned-vec package
    /// A Buffer that wraps an Apache Arrow buffer
    Arrow(ArrowBuffer),

    /// A Buffer that wraps an owned [`bytes::Bytes`].
    Bytes(bytes::Bytes),
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Create a new buffer of the provided length with all bytes set to `0u8`.
    /// If len is 0, does not perform any allocations.
    pub fn from_len_zeroed(len: usize) -> Self {
        Self::from(ArrowMutableBuffer::from_len_zeroed(len))
    }

    /// Length of the buffer in bytes
    pub fn len(&self) -> usize {
        match &self.0 {
            Inner::Arrow(b) => b.len(),
            Inner::Bytes(b) => b.len(),
        }
    }

    /// Predicate for empty buffers
    pub fn is_empty(&self) -> bool {
        match &self.0 {
            Inner::Arrow(b) => b.is_empty(),
            Inner::Bytes(b) => b.is_empty(),
        }
    }

    #[allow(clippy::same_name_method)]
    /// Return a new view on the buffer, but limited to the given index range.
    pub fn slice(&self, range: Range<usize>) -> Self {
        match &self.0 {
            Inner::Arrow(b) => Buffer(Inner::Arrow(
                b.slice_with_length(range.start, range.end - range.start),
            )),
            Inner::Bytes(b) => {
                if range.is_empty() {
                    // bytes::Bytes::slice does not preserve alignment if the range is empty
                    let mut empty_b = b.clone();
                    empty_b.truncate(0);
                    Buffer(Inner::Bytes(empty_b))
                } else {
                    Buffer(Inner::Bytes(b.slice(range)))
                }
            }
        }
    }

    #[allow(clippy::same_name_method)]
    /// Access the buffer as an immutable byte slice.
    pub fn as_slice(&self) -> &[u8] {
        match &self.0 {
            Inner::Arrow(b) => b.as_ref(),
            Inner::Bytes(b) => b.as_ref(),
        }
    }

    /// Convert the buffer into a `Vec` of the given native type `T`.
    ///
    /// # Ownership
    /// The caller takes ownership of the underlying memory.
    ///
    /// # Errors
    /// This method will fail if the underlying buffer is an owned [`bytes::Bytes`].
    ///
    /// This method will also fail if we attempt to pass a `T` that is not aligned to the `T` that
    /// it was originally allocated with.
    pub fn into_vec<T: ArrowNativeType>(self) -> Result<Vec<T>, Self> {
        match self.0 {
            Inner::Arrow(buffer) => buffer.into_vec::<T>().map_err(|b| Buffer(Inner::Arrow(b))),
            // Cannot convert bytes into a mutable vec
            Inner::Bytes(_) => Err(self),
        }
    }

    /// Convert a Buffer into an ArrowBuffer with no copying.
    pub fn into_arrow(self) -> ArrowBuffer {
        match self.0 {
            Inner::Arrow(a) => a,
            // This is cheeky. But it uses From<bytes::Bytes> for arrow_buffer::Bytes, even though
            // arrow_buffer::Bytes is only pub(crate). Seems weird...
            // See: https://github.com/apache/arrow-rs/issues/6033
            Inner::Bytes(b) => ArrowBuffer::from_bytes(b.into()),
        }
    }
}

impl Deref for Buffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[u8]> for Buffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl From<&[u8]> for Buffer {
    fn from(value: &[u8]) -> Self {
        // We prefer Arrow since it retains mutability
        Buffer(Inner::Arrow(ArrowBuffer::from(value)))
    }
}

impl<T: ArrowNativeType> From<Vec<T>> for Buffer {
    fn from(value: Vec<T>) -> Self {
        // We prefer Arrow since it retains mutability
        Buffer(Inner::Arrow(ArrowBuffer::from_vec(value)))
    }
}

impl From<bytes::Bytes> for Buffer {
    fn from(value: bytes::Bytes) -> Self {
        Buffer(Inner::Bytes(value))
    }
}

impl From<ArrowBuffer> for Buffer {
    fn from(value: ArrowBuffer) -> Self {
        Buffer(Inner::Arrow(value))
    }
}

impl From<ArrowMutableBuffer> for Buffer {
    fn from(value: ArrowMutableBuffer) -> Self {
        Buffer(Inner::Arrow(ArrowBuffer::from(value)))
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl Eq for Buffer {}

impl PartialOrd for Buffer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_ref().partial_cmp(other.as_ref())
    }
}
