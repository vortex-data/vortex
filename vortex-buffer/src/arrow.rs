use arrow_buffer::{ArrowNativeType, OffsetBuffer};
use bytes::Bytes;
use vortex_error::vortex_panic;

use crate::{Alignment, Buffer, ByteBuffer};

impl<T: ArrowNativeType> Buffer<T> {
    /// Converts the buffer zero-copy into a `arrow_buffer::Buffer`.
    pub fn into_arrow_scalar_buffer(self) -> arrow_buffer::ScalarBuffer<T> {
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
    /// Panics if the Arrow buffer is not aligned to the requested alignment, or if the requested
    /// alignment is not sufficient for type T.
    pub fn from_arrow_scalar_buffer(arrow: arrow_buffer::ScalarBuffer<T>) -> Self {
        let length = arrow.len();
        let bytes = Bytes::from_owner(ArrowWrapper(arrow.into_inner()));

        let alignment = Alignment::of::<T>();
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

    /// Converts the buffer zero-copy into a `arrow_buffer::OffsetBuffer`.
    ///
    /// SAFETY: The caller should ensure that the buffer contains monotonically increasing values
    /// greater than or equal to zero.
    pub fn into_arrow_offset_buffer(self) -> OffsetBuffer<T> {
        unsafe { OffsetBuffer::new_unchecked(self.into_arrow_scalar_buffer()) }
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

#[cfg(test)]
mod test {
    use arrow_buffer::{Buffer as ArrowBuffer, ScalarBuffer};

    use crate::{buffer, Alignment, Buffer};

    #[test]
    fn into_arrow_buffer() {
        let buf = buffer![0u8, 1, 2];
        let arrow: ArrowBuffer = buf.clone().into_arrow_buffer();
        assert_eq!(arrow.as_ref(), buf.as_slice(), "Buffer values differ");
        assert_eq!(arrow.as_ptr(), buf.as_ptr(), "Conversion not zero-copy")
    }

    #[test]
    fn into_arrow_scalar_buffer() {
        let buf = buffer![0i32, 1, 2];
        let scalar: ScalarBuffer<i32> = buf.clone().into_arrow_scalar_buffer();
        assert_eq!(scalar.as_ref(), buf.as_slice(), "Buffer values differ");
        assert_eq!(scalar.as_ptr(), buf.as_ptr(), "Conversion not zero-copy")
    }

    #[test]
    fn from_arrow_buffer() {
        let arrow = ArrowBuffer::from_vec(vec![0i32, 1, 2]);
        let buf = Buffer::from_arrow_buffer(arrow.clone(), Alignment::of::<i32>());
        assert_eq!(arrow.as_ref(), buf.as_slice(), "Buffer values differ");
        assert_eq!(arrow.as_ptr(), buf.as_ptr(), "Conversion not zero-copy");
    }
}
