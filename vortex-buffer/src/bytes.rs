// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bytes::Buf;
use vortex_error::VortexExpect;

use crate::Alignment;
use crate::ByteBuffer;
use crate::ConstBuffer;
use crate::ConstByteBuffer;

/// An extension to the [`Buf`] trait that provides a function `copy_to_aligned` similar to
/// `copy_to_bytes` that allows for zero-copy aligned reads where possible.
pub trait AlignedBuf: Buf {
    /// Copy the next `len` bytes from the buffer into a new buffer with the given alignment.
    /// This will be zero-copy wherever possible.
    ///
    /// The [`Buf`] trait has a specialized `copy_to_bytes` function that allows the implementation
    /// of `Buf` for `Bytes` and `BytesMut` to return bytes with zero-copy.
    ///
    /// This function provides similar functionality for `ByteBuffer`.
    ///
    /// TODO(ngates): what should this do the alignment of the current buffer? We have to advance
    ///  it by len..
    fn copy_to_aligned(&mut self, len: usize, alignment: Alignment) -> ByteBuffer {
        // The default implementation uses copy_to_bytes, and then tries to align.
        // When the underlying `copy_to_bytes` is zero-copy, this may perform one copy to align
        // the bytes. When the underlying `copy_to_bytes` is not zero-copy, this may perform two
        // copies.
        // The only way to fix this would be to invert the implementation so `copy_to_bytes`
        // invokes `copy_to_aligned` with an alignment of 1. But we cannot override this in the
        // default trait.
        // In practice, we tend to only call this function on `ByteBuffer: AlignedBuf`, and
        // therefore we have a maximum of one copy, so I'm not too worried about it.
        ByteBuffer::from(self.copy_to_bytes(len)).aligned(alignment)
    }

    /// See [`AlignedBuf::copy_to_aligned`].
    fn copy_to_const_aligned<const A: usize>(&mut self, len: usize) -> ConstByteBuffer<A> {
        // The default implementation uses copy_to_bytes, and then returns a ByteBuffer with
        // alignment of 1. This will be zero-copy if the underlying `copy_to_bytes` is zero-copy.
        ConstBuffer::try_from(self.copy_to_aligned(len, Alignment::new(A)))
            .vortex_expect("we just aligned the buffer")
    }
}

impl<B: Buf> AlignedBuf for B {}

#[cfg(test)]
mod tests {
    use bytes::Buf;

    use crate::AlignedBuf;
    use crate::Alignment;
    use crate::ByteBuffer;

    #[test]
    fn copy_to_aligned_and_const_aligned_advance_and_align() {
        // Use Bytes (whose Buf::copy_to_bytes is zero-copy) starting at an 8-byte aligned address
        // so advancing by 1 byte produces a genuinely misaligned pointer for the 8-byte read.
        let mut buf = ByteBuffer::copy_from_aligned([0u8, 1, 2, 3, 4, 5, 6, 7], Alignment::new(8))
            .into_bytes();

        let head = buf.copy_to_aligned(1, Alignment::new(1));
        assert_eq!(head.as_slice(), &[0]);
        assert_eq!(buf.remaining(), 7);

        let misaligned_ptr = buf.chunk().as_ptr();
        assert!(!Alignment::new(8).is_offset_aligned(misaligned_ptr.addr()));
        let mid = buf.copy_to_aligned(4, Alignment::new(8));
        assert_ne!(mid.as_ptr(), misaligned_ptr);
        assert_eq!(mid.as_slice(), &[1, 2, 3, 4]);
        assert!(mid.is_aligned(Alignment::new(8)));
        assert_eq!(mid.alignment(), Alignment::new(8));
        assert_eq!(buf.remaining(), 3);

        let tail = buf.copy_to_const_aligned::<16>(3);
        assert_eq!(tail.as_slice(), &[5, 6, 7]);
        assert!(tail.inner().is_aligned(Alignment::new(16)));
        assert_eq!(buf.remaining(), 0);
    }
}
