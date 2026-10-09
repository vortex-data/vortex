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
    /// ## Panics
    ///
    /// Panics if the buffer is empty, its first offset is negative, or it is not monotonically
    /// increasing -- the invariants `OffsetBuffer` requires of its contents.
    pub fn into_arrow_offset_buffer(self) -> OffsetBuffer<T> {
        OffsetBuffer::new(self.into_arrow_scalar_buffer())
    }

    /// Converts the buffer zero-copy into a `arrow_buffer::OffsetBuffer` without validation.
    ///
    /// Use this when the offsets were already validated, for example when they come from an
    /// array whose invariants guarantee valid offsets.
    ///
    /// # Safety
    ///
    /// The buffer must be non-empty, its first offset must be greater than or equal to zero, and
    /// it must be monotonically non-decreasing.
    pub unsafe fn into_arrow_offset_buffer_unchecked(self) -> OffsetBuffer<T> {
        // SAFETY: the caller guarantees the `OffsetBuffer` invariants.
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

    #[test]
    fn into_arrow_offset_buffer() {
        let buf = buffer![0i32, 2, 2, 5];
        let offsets = buf.clone().into_arrow_offset_buffer();
        assert_eq!(offsets.as_ref(), buf.as_slice(), "Buffer values differ");
        assert_eq!(offsets.as_ptr(), buf.as_ptr(), "Conversion not zero-copy");
    }

    #[test]
    fn into_arrow_offset_buffer_unchecked() {
        let buf = buffer![0i32, 2, 2, 5];

        // SAFETY: the offsets are non-empty, non-negative and monotonically non-decreasing.
        let offsets = unsafe { buf.clone().into_arrow_offset_buffer_unchecked() };

        assert_eq!(offsets.as_ref(), buf.as_slice(), "Buffer values differ");
        assert_eq!(offsets.as_ptr(), buf.as_ptr(), "Conversion not zero-copy");
    }

    #[test]
    fn into_arrow_offset_buffer_allows_nonzero_start() {
        // Offsets into a sliced child array need not begin at zero.
        let offsets = buffer![3i64, 4, 9].into_arrow_offset_buffer();
        assert_eq!(offsets.as_ref(), &[3, 4, 9]);
    }

    #[test]
    #[should_panic(expected = "offsets must be monotonically increasing")]
    fn into_arrow_offset_buffer_rejects_decreasing_offsets() {
        drop(buffer![0i32, 5, 3].into_arrow_offset_buffer());
    }

    #[test]
    #[should_panic(expected = "offsets must be greater than 0")]
    fn into_arrow_offset_buffer_rejects_negative_offsets() {
        drop(buffer![-1i32, 2].into_arrow_offset_buffer());
    }

    #[test]
    #[should_panic(expected = "offsets cannot be empty")]
    fn into_arrow_offset_buffer_rejects_empty() {
        drop(Buffer::<i32>::empty().into_arrow_offset_buffer());
    }
}
