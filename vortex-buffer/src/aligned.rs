use std::ops::Deref;

use bytes::Bytes;
use vortex_error::vortex_panic;

use crate::alignment::Alignment;

/// A buffer with runtime-validated alignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlignedBuffer {
    /// The underlying bytes holding the data.
    bytes: Bytes,
    /// The minimum alignment required for this buffer when (de)serialized.
    alignment: Alignment,
}

impl AlignedBuffer {
    /// Create a new `AlignedBuffer` from the provided buffer and alignment.
    ///
    /// ## Panics
    ///
    /// Panics if `alignment` is greater than `u16::MAX`, is not a power of 2, or the buffer
    /// is not aligned to `alignment`.
    pub fn new_with_alignment(bytes: Bytes, alignment: Alignment) -> Self {
        if bytes.as_ptr().align_offset(*alignment) != 0 {
            vortex_panic!("Buffer must be aligned to {}", alignment);
        }
        Self { bytes, alignment }
    }

    /// Create a new `AlignedBuffer` from the provided buffer with alignment derived from `T`.
    pub fn new<T>(bytes: Bytes) -> Self {
        Self::new_with_alignment(bytes, align_of::<T>().into())
    }

    /// The alignment of the buffer.
    #[inline]
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Extracts the underlying `Bytes` from the buffer.
    pub fn into_inner(self) -> Bytes {
        self.bytes
    }
}

impl Deref for AlignedBuffer {
    type Target = Bytes;

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}
