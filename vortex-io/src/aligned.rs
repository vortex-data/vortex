#![allow(dead_code)]
use std::ops::{Deref, DerefMut};

use bytes::Bytes;

pub trait PowerOfTwo<const N: usize> {}
impl<const N: usize> PowerOfTwo<N> for usize where usize: sealed::Sealed<N> {}

mod sealed {
    pub trait Sealed<const N: usize> {}

    impl Sealed<1> for usize {}
    impl Sealed<2> for usize {}
    impl Sealed<4> for usize {}
    impl Sealed<8> for usize {}
    impl Sealed<16> for usize {}
    impl Sealed<32> for usize {}
    impl Sealed<64> for usize {}
    impl Sealed<128> for usize {}
    impl Sealed<256> for usize {}
    impl Sealed<512> for usize {}
}

/// A variant of [`BytesMut`][bytes::BytesMut] that freezes into a [`Bytes`] that is guaranteed
/// to begin at a multiple of a target byte-alignment.
///
/// Internally, it accomplishes this by over-allocating by up to the alignment size, padding the
/// front as necessary. Reads and writes will only be able to access the region after the padding.
///
/// It is required for the alignment to be a valid power of 2 <= 512, any other value will be
/// a compile-time failure.
pub(crate) struct AlignedBytesMut<const ALIGN: usize> {
    buf: Vec<u8>,
    padding: usize,
    capacity: usize,
}

impl<const ALIGN: usize> AlignedBytesMut<ALIGN>
where
    usize: PowerOfTwo<ALIGN>,
{
    /// Allocate a new mutable buffer with capacity to hold at least `capacity` bytes.
    ///
    /// The mutable buffer may allocate more than  the requested amount to pad the memory for
    /// alignment.
    pub fn with_capacity(capacity: usize) -> Self {
        // Allocate up to `ALIGN` extra bytes, in case we need to pad the returned pointer.
        let allocation_size = (capacity + ALIGN - 1).next_multiple_of(ALIGN);
        let mut buf = Vec::<u8>::with_capacity(allocation_size);
        let padding = buf.as_ptr().align_offset(ALIGN);
        unsafe {
            buf.set_len(padding);
        }

        Self {
            buf,
            padding,
            capacity,
        }
    }

    /// Usable capacity of this buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Set the length of the mutable buffer directly.
    ///
    /// # Safety
    ///
    /// The caller is responsible for ensuring that the provided length fits within the original
    /// capacity request.
    ///
    /// Failure to do so could cause uninitialized memory to be readable.
    pub unsafe fn set_len(&mut self, len: usize) {
        assert!(
            len <= self.capacity,
            "set_len call out of bounds: {} > {}",
            len,
            self.capacity
        );
        unsafe { self.buf.set_len(len + self.padding) }
    }

    /// Extend this mutable buffer with the contents of the provided slice.
    pub fn extend_from_slice(&mut self, slice: &[u8]) {
        // The internal `buf` is padded, so appends will land after the padded region.
        self.buf.extend_from_slice(slice)
    }

    /// Freeze the existing allocation into a readonly [`Bytes`], guaranteed to be aligned to
    /// the target [`ALIGN`] size.
    pub fn freeze(self) -> Bytes {
        // bytes_unaligned will contain the entire allocation, so that on Drop the entire buf
        // is freed.
        //
        // bytes_aligned is a sliced view on top of bytes_unaligned.
        //
        // bytes_aligned
        //     | parent    \  *ptr
        //     v            |
        // bytes_unaligned  |
        //     |            |
        //     | *ptr       |
        //     v            v
        //     +------------+------------------+----------------+
        //     | padding    |   content        | spare capacity |
        //     +------------+------------------+----------------+
        let bytes_unaligned = Bytes::from(self.buf);
        let bytes_aligned = bytes_unaligned.slice(self.padding..);

        assert_eq!(
            bytes_aligned.as_ptr().align_offset(ALIGN),
            0,
            "bytes_aligned must be aligned to {}",
            ALIGN
        );

        bytes_aligned
    }
}

impl<const ALIGN: usize> Deref for AlignedBytesMut<ALIGN>
where
    usize: PowerOfTwo<ALIGN>,
{
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.buf[self.padding..]
    }
}

impl<const ALIGN: usize> DerefMut for AlignedBytesMut<ALIGN>
where
    usize: PowerOfTwo<ALIGN>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buf[self.padding..]
    }
}

#[cfg(test)]
mod tests {
    use crate::aligned::AlignedBytesMut;

    #[test]
    fn test_align() {
        let mut buf = AlignedBytesMut::<128>::with_capacity(1);
        buf.extend_from_slice(b"a");

        let data = buf.freeze();

        assert_eq!(data.as_ref(), b"a");
        assert_eq!(data.as_ptr().align_offset(128), 0);
    }

    #[test]
    fn test_extend() {
        let mut buf = AlignedBytesMut::<128>::with_capacity(256);
        buf.extend_from_slice(b"a");
        buf.extend_from_slice(b"bcdefgh");

        let data = buf.freeze();
        assert_eq!(data.as_ref(), b"abcdefgh");
    }
}
