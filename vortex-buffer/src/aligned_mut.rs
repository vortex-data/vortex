use std::ptr;

use bytes::{Buf, BufMut, BytesMut};
use vortex_error::{vortex_panic, VortexExpect};

use crate::alignment::Alignment;
use crate::AlignedBuffer;

/// A mutable buffer with runtime-guaranteed alignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlignedBufferMut {
    /// The underlying bytes holding the data.
    bytes: BytesMut,
    /// The minimum alignment required for this buffer when (de)serialized.
    alignment: Alignment,
}

impl AlignedBufferMut {
    /// Create a new `AlignedBufferMut` with the requested alignment.
    pub fn new(alignment: Alignment) -> Self {
        Self::with_capacity(1024, alignment)
    }

    /// Create a new `AlignedBufferMut` with the requested capacity and alignment.
    pub fn with_capacity(capacity: usize, alignment: Alignment) -> Self {
        let mut bytes = BytesMut::with_capacity(capacity + *alignment);
        bytes.align_empty(alignment);
        Self { bytes, alignment }
    }

    /// Get the alignment of the buffer.
    #[inline(always)]
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Returns the length of the buffer.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns whether the buffer is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Returns the capacity of the buffer.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    /// Reserves capacity for at least `additional` more bytes to be inserted in the buffer.
    ///
    /// # Example:
    ///
    /// ```
    /// use vortex_buffer::{AlignedBufferMut, Alignment};
    ///
    /// let mut buffer = AlignedBufferMut::with_capacity(10, Alignment::of::<u8>());
    /// buffer.reserve(10);
    ///
    /// assert!(buffer.capacity() >= 20);
    /// ```
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        if additional <= self.capacity() - self.len() {
            // We can fit the additional bytes in the remaining capacity. Nothing to do.
            return;
        }
        // Otherwise, reserve additional + alignment bytes in case we need to realign the buffer.
        self.reserve_allocate(additional);
    }

    /// A separate function so we can inline the reserve call's fast path. According to `BytesMut`
    /// this has significant performance implications.
    fn reserve_allocate(&mut self, additional: usize) {
        let mut bytes = BytesMut::with_capacity(self.len() + additional + *self.alignment);
        bytes.align_empty(self.alignment);
        bytes.extend_from_slice(&self.bytes);
        self.bytes = bytes;
    }

    /// Appends given bytes to this `AlignedBufferMut`.
    ///
    /// If there is insufficient capacity, it is resized first.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_buffer::{AlignedBufferMut, Alignment};
    ///
    /// let mut buf = AlignedBufferMut::new(Alignment::of::<u8>());
    /// buf.extend_from_slice(b"aaabbb");
    /// buf.extend_from_slice(b"cccddd");
    ///
    /// assert_eq!(b"aaabbbcccddd", &buf[..]);
    /// ```
    #[inline]
    pub fn extend_from_slice(&mut self, extend: &[u8]) {
        self.reserve(extend.len());
        self.bytes.extend_from_slice(extend);
    }

    /// Returns a slice of the buffer.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// Returns a mutable slice of the buffer.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.bytes.as_mut()
    }

    /// Freezes this `AlignedBufferMut`, returning an immutable shareable `AlignedBuffer`.
    pub fn freeze(self) -> AlignedBuffer {
        AlignedBuffer::new_with_alignment(self.bytes.freeze(), self.alignment)
    }
}

/// Extension trait for [`BytesMut`] that provides functions for aligning the buffer.
trait AlignedBytesMut {
    /// Align an empty `BytesMut` to the specified alignment.
    ///
    /// ## Panics
    ///
    /// Panics if the buffer is not empty, or if there is not enough capacity to align the buffer.
    fn align_empty(&mut self, alignment: Alignment);
}

impl AlignedBytesMut for BytesMut {
    fn align_empty(&mut self, alignment: Alignment) {
        if !self.is_empty() {
            vortex_panic!("AlignedBufferMut must be empty");
        }

        let padding = self.as_ptr().align_offset(*alignment);
        self.capacity()
            .checked_sub(padding)
            .vortex_expect("Not enough capacity to align buffer");

        // SAFETY: We know the buffer is empty, and we know we have enough capacity, so we can
        // safely set the length to the padding and advance the buffer to the aligned offset.
        unsafe { self.set_len(padding) };
        self.advance(padding);
    }
}
