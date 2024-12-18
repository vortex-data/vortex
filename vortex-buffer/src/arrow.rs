use arrow_buffer::ArrowNativeType;
use bytes::Bytes;
use vortex_dtype::NativePType;

use crate::{AlignedBuffer, Alignment, ScalarBuffer};

struct ArrowWrapper(arrow_buffer::Buffer);

impl AsRef<[u8]> for ArrowWrapper {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl AlignedBuffer {
    /// Converts the buffer zero-copy into a `arrow_buffer::Buffer`.
    pub fn into_arrow(self) -> arrow_buffer::Buffer {
        let bytes = self.into_inner();
        // This is cheeky. But it uses From<bytes::Bytes> for arrow_buffer::Bytes, even though
        // arrow_buffer::Bytes is only pub(crate). Seems weird...
        // See: https://github.com/apache/arrow-rs/issues/6033
        arrow_buffer::Buffer::from_bytes(bytes.into())
    }

    /// Creates a new `AlignedBuffer` from an `arrow_buffer::Buffer`.
    pub fn from_arrow(buffer: arrow_buffer::Buffer, alignment: Alignment) -> Self {
        Self::new_with_alignment(Bytes::from_owner(ArrowWrapper(buffer)), alignment)
    }
}

impl<T: NativePType + ArrowNativeType> ScalarBuffer<T> {
    /// Converts the buffer zero-copy into a `arrow_buffer::ScalarBuffer`.
    pub fn into_arrow(self) -> arrow_buffer::ScalarBuffer<T> {
        arrow_buffer::ScalarBuffer::from(self.into_inner().into_arrow())
    }
}
