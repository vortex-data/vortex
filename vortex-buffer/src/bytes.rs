use bytes::Buf;
use vortex_error::VortexExpect;

use crate::{Alignment, ByteBuffer, ConstBuffer, ConstByteBuffer};

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
        //
        // The only way to fix this would be to invert the implementation so `copy_to_bytes`
        // invokes `copy_to_aligned` with an alignment of 1. But we cannot override this in the
        // default trait.
        //
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
