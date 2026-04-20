// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zero-copy conversions between Vortex buffer types and Arrow buffer types.
//!
//! These extension traits provide the conversions that used to live in
//! `vortex-buffer`. Keeping them in `vortex-array` avoids forcing `vortex-buffer`
//! to depend on Arrow.

use arrow_buffer::ArrowNativeType;
use arrow_buffer::BooleanBuffer;
use arrow_buffer::OffsetBuffer;
use bytes::Bytes;
use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::vortex_panic;

/// A wrapper that lets an `arrow_buffer::Buffer` be consumed by `Bytes::from_owner`.
struct ArrowWrapper(arrow_buffer::Buffer);

impl AsRef<[u8]> for ArrowWrapper {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

/// Convert an Arrow `Buffer` into `Bytes` with zero copy, preserving the original allocation.
fn arrow_buffer_to_bytes(arrow: arrow_buffer::Buffer) -> Bytes {
    Bytes::from_owner(ArrowWrapper(arrow))
}

/// Extension trait on `Buffer<T>` for zero-copy conversions to Arrow typed buffers.
pub trait BufferArrowExt<T: ArrowNativeType>: Sized {
    /// Convert this buffer zero-copy into an [`arrow_buffer::ScalarBuffer`].
    fn into_arrow_scalar_buffer(self) -> arrow_buffer::ScalarBuffer<T>;

    /// Convert an Arrow scalar buffer into a Vortex `Buffer<T>`.
    ///
    /// ## Panics
    ///
    /// Panics if the Arrow buffer is not aligned to the alignment of `T`.
    fn from_arrow_scalar_buffer(arrow: arrow_buffer::ScalarBuffer<T>) -> Self;

    /// Convert this buffer zero-copy into an [`arrow_buffer::OffsetBuffer`].
    ///
    /// # Safety
    ///
    /// The caller must ensure the buffer contains monotonically increasing values
    /// that are greater than or equal to zero.
    fn into_arrow_offset_buffer(self) -> OffsetBuffer<T>;
}

impl<T: ArrowNativeType> BufferArrowExt<T> for Buffer<T> {
    fn into_arrow_scalar_buffer(self) -> arrow_buffer::ScalarBuffer<T> {
        let buffer = arrow_buffer::Buffer::from(self.into_inner());
        arrow_buffer::ScalarBuffer::from(buffer)
    }

    fn from_arrow_scalar_buffer(arrow: arrow_buffer::ScalarBuffer<T>) -> Self {
        let length = arrow.len();
        let alignment = Alignment::of::<T>();
        let bytes = arrow_buffer_to_bytes(arrow.into_inner());

        if bytes.as_ptr().align_offset(*alignment) != 0 {
            vortex_panic!(
                "Arrow buffer is not aligned to the requested alignment: {}",
                alignment
            );
        }

        // SAFETY: we asserted the pointer is aligned, and `ScalarBuffer<T>` stores
        // exactly `length * size_of::<T>()` bytes by construction.
        Self::from_bytes_aligned(bytes.slice(..length * size_of::<T>()), alignment)
    }

    fn into_arrow_offset_buffer(self) -> OffsetBuffer<T> {
        // SAFETY: callers are documented to uphold the monotonicity invariant.
        unsafe { OffsetBuffer::new_unchecked(self.into_arrow_scalar_buffer()) }
    }
}

/// Extension trait on `ByteBuffer` for zero-copy conversions to Arrow `Buffer`.
pub trait ByteBufferArrowExt: Sized {
    /// Convert this buffer zero-copy into an [`arrow_buffer::Buffer`].
    fn into_arrow_buffer(self) -> arrow_buffer::Buffer;

    /// Convert an Arrow buffer into a Vortex `ByteBuffer` with the requested alignment.
    ///
    /// ## Panics
    ///
    /// Panics if the Arrow buffer is not sufficiently aligned.
    fn from_arrow_buffer(arrow: arrow_buffer::Buffer, alignment: Alignment) -> Self;
}

impl ByteBufferArrowExt for ByteBuffer {
    fn into_arrow_buffer(self) -> arrow_buffer::Buffer {
        arrow_buffer::Buffer::from(self.into_inner())
    }

    fn from_arrow_buffer(arrow: arrow_buffer::Buffer, alignment: Alignment) -> Self {
        let bytes = arrow_buffer_to_bytes(arrow);
        if bytes.as_ptr().align_offset(*alignment) != 0 {
            vortex_panic!(
                "Arrow buffer is not aligned to the requested alignment: {}",
                alignment
            );
        }
        Self::from_bytes_aligned(bytes, alignment)
    }
}

/// Extension trait for constructing a `BitBuffer` from an Arrow `BooleanBuffer`.
pub trait BitBufferArrowExt: Sized {
    /// Convert an Arrow `BooleanBuffer` into a Vortex `BitBuffer` preserving the bit offset.
    fn from_arrow_boolean_buffer(value: BooleanBuffer) -> Self;
}

impl BitBufferArrowExt for BitBuffer {
    fn from_arrow_boolean_buffer(value: BooleanBuffer) -> Self {
        let offset = value.offset();
        let len = value.len();
        let buffer = value.into_inner();
        let buffer = ByteBuffer::from_arrow_buffer(buffer, Alignment::of::<u8>());
        BitBuffer::new_with_offset(buffer, len, offset)
    }
}

/// Extension trait for converting a `BitBuffer` into an Arrow `BooleanBuffer`.
pub trait BitBufferIntoArrow {
    /// Convert this bit buffer zero-copy into an Arrow [`BooleanBuffer`].
    fn into_arrow_boolean_buffer(self) -> BooleanBuffer;
}

impl BitBufferIntoArrow for BitBuffer {
    fn into_arrow_boolean_buffer(self) -> BooleanBuffer {
        let (offset, len, buffer) = self.into_inner();
        BooleanBuffer::new(buffer.into_arrow_buffer(), offset, len)
    }
}
