use std::ops::Deref;

use bytes::Bytes;
use vortex_error::{vortex_panic, VortexExpect};

use crate::Buffer;

/// A buffer with runtime-validated alignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlignedBuffer {
    /// The underlying bytes holding the data.
    bytes: Bytes,
    /// The minimum alignment required for this buffer when (de)serialized.
    alignment: usize,
}

impl AlignedBuffer {
    /// Create a new `ArrayBuffer` from the provided buffer and alignment.
    ///
    /// ## Panics
    ///
    /// Panics if `alignment` is greater than `u16::MAX`, is not a power of 2, or the buffer
    /// is not aligned to `alignment`.
    pub fn new_with_alignment(buffer: Buffer, alignment: usize) -> Self {
        u16::try_from(alignment).vortex_expect("Alignment must fit into u16");
        if !alignment.is_power_of_two() {
            vortex_panic!("Alignment must be a power of 2");
        }
        if buffer.as_ptr().align_offset(alignment) != 0 {
            vortex_panic!("Buffer must be aligned to {}", alignment);
        }
        Self { buffer, alignment }
    }

    /// Create a new `ArrayBuffer` from the provided buffer with alignment derived from `T`.
    pub fn new<T>(buffer: Buffer) -> Self {
        Self::new_with_alignment(buffer, align_of::<T>())
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    #[inline]
    pub fn alignment_u16(&self) -> u16 {
        u16::try_from(self.alignment).vortex_expect("Alignment must fit into u16")
    }

    pub fn into_inner(self) -> Buffer {
        self.buffer
    }
}

impl Deref for AlignedBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
