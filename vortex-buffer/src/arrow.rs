use arrow_buffer::ArrowNativeType;
use bytes::Bytes;
use vortex_dtype::NativePType;
use vortex_error::vortex_panic;

use crate::{Alignment, Buffer, ByteBuffer};

impl<T: NativePType + ArrowNativeType> Buffer<T> {
    /// Converts the buffer zero-copy into a `arrow_buffer::Buffer`.
    pub fn into_arrow(self) -> arrow_buffer::ScalarBuffer<T> {
        let bytes = self.into_inner();
        // This is cheeky. But it uses From<bytes::Bytes> for arrow_buffer::Bytes, even though
        // arrow_buffer::Bytes is only pub(crate). Seems weird...
        // See: https://github.com/apache/arrow-rs/issues/6033
        let buffer = arrow_buffer::Buffer::from_bytes(bytes.into());
        arrow_buffer::ScalarBuffer::from(buffer)
    }

    /// Convert an Arrow scalar buffer into a Vortex scalar buffer.
    ///
    /// ## Panics
    ///
    /// Panics if the Arrow buffer is not sufficiently aligned.
    pub fn from_arrow(arrow: arrow_buffer::ScalarBuffer<T>, alignment: Alignment) -> Self {
        let length = arrow.len();

        let bytes = Bytes::from_owner(ArrowWrapper(arrow.into_inner()));
        if bytes.as_ptr().align_offset(*alignment) != 0 {
            vortex_panic!(
                "Arrow buffer is not aligned to the requested alignment: {}",
                alignment
            );
        }

        Self {
            bytes,
            length,
            alignment,
            _marker: Default::default(),
        }
    }
}

impl ByteBuffer {
    /// Converts the buffer zero-copy into a `arrow_buffer::Buffer`.
    pub fn into_arrow_buffer(self) -> arrow_buffer::Buffer {
        let bytes = self.into_inner();
        // This is cheeky. But it uses From<bytes::Bytes> for arrow_buffer::Bytes, even though
        // arrow_buffer::Bytes is only pub(crate). Seems weird...
        // See: https://github.com/apache/arrow-rs/issues/6033
        arrow_buffer::Buffer::from_bytes(bytes.into())
    }

    /// Convert an Arrow scalar buffer into a Vortex scalar buffer.
    ///
    /// ## Panics
    ///
    /// Panics if the Arrow buffer is not sufficiently aligned.
    pub fn from_arrow_buffer(arrow: arrow_buffer::Buffer, alignment: Alignment) -> Self {
        let length = arrow.len();

        let bytes = Bytes::from_owner(ArrowWrapper(arrow));
        if bytes.as_ptr().align_offset(*alignment) != 0 {
            vortex_panic!(
                "Arrow buffer is not aligned to the requested alignment: {}",
                alignment
            );
        }

        Self {
            bytes,
            length,
            alignment,
            _marker: Default::default(),
        }
    }
}

/// A wrapper struct to allow `arrow_buffer::Buffer` to implement `AsRef<[u8]>` for
/// `Bytes::from_owner`.
struct ArrowWrapper(arrow_buffer::Buffer);

impl AsRef<[u8]> for ArrowWrapper {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}
