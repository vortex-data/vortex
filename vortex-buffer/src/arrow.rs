// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::ArrowNativeType;
use arrow_buffer::OffsetBuffer;
use vortex_error::vortex_panic;

use crate::Alignment;
use crate::Buffer;
use crate::ByteBuffer;

impl<T: ArrowNativeType> Buffer<T> {
    /// Converts the buffer zero-copy into a `arrow_buffer::Buffer`.
    pub fn into_arrow_scalar_buffer(self) -> arrow_buffer::ScalarBuffer<T> {
        if self.is_empty() {
            return Vec::new().into();
        }
        let buffer = self.into_byte_buffer().into_arrow_buffer();
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
        let arrow = arrow.into_inner();

        let alignment = Alignment::of::<T>();
        if arrow.as_ptr().align_offset(alignment.as_usize()) != 0 {
            vortex_panic!(
                "Arrow buffer is not aligned to the requested alignment: {}",
                alignment
            );
        }

        debug_assert_eq!(length, arrow.len() / size_of::<T>());
        Self::from_arrow_owner(arrow, length, alignment)
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
        if let Some(crate::BufferBacking::Arrow(arrow)) = self.backing.as_deref() {
            let offset = self.ptr.addr().get() - arrow.as_ptr().addr();
            return arrow.slice_with_length(offset, self.length);
        }
        arrow_buffer::Buffer::from(self.into_bytes())
    }

    /// Convert an Arrow scalar buffer into a Vortex scalar buffer.
    ///
    /// ## Panics
    ///
    /// Panics if the Arrow buffer is not sufficiently aligned.
    pub fn from_arrow_buffer(arrow: arrow_buffer::Buffer, alignment: Alignment) -> Self {
        let length = arrow.len();

        if arrow.as_ptr().align_offset(alignment.as_usize()) != 0 {
            vortex_panic!(
                "Arrow buffer is not aligned to the requested alignment: {}",
                alignment
            );
        }

        Self::from_arrow_owner(arrow, length, alignment)
    }
}

#[cfg(test)]
mod test {
    use arrow_buffer::Buffer as ArrowBuffer;
    use arrow_buffer::ScalarBuffer;

    use crate::Alignment;
    use crate::Buffer;
    use crate::buffer;

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
    fn empty_into_arrow_scalar_buffer() {
        let scalar = Buffer::<i64>::empty().into_arrow_scalar_buffer();

        assert!(scalar.is_empty());
        assert_eq!(scalar.as_ptr().align_offset(align_of::<i64>()), 0);
    }

    #[test]
    fn from_arrow_buffer() {
        let arrow = ArrowBuffer::from_vec(vec![0i32, 1, 2]);
        let buf = Buffer::from_arrow_buffer(arrow.clone(), Alignment::of::<i32>());
        assert_eq!(arrow.as_ref(), buf.as_slice(), "Buffer values differ");
        assert_eq!(arrow.as_ptr(), buf.as_ptr(), "Conversion not zero-copy");

        let round_trip = buf.into_arrow_buffer();
        assert_eq!(round_trip.as_ptr(), arrow.as_ptr());
    }
}
