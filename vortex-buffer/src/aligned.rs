use std::fmt::Debug;
use std::ops::{Bound, Deref, RangeBounds};

use bytes::Bytes;
use vortex_error::vortex_panic;

use crate::alignment::Alignment;
use crate::AlignedBufferMut;

/// A buffer with runtime-validated alignment.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
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

    /// Create a new `AlignedBuffer` by copying the provided slice.
    pub fn copy_from_slice(slice: &[u8], alignment: Alignment) -> Self {
        let mut buffer = AlignedBufferMut::with_capacity(slice.len(), alignment);
        buffer.extend_from_slice(slice);
        buffer.freeze()
    }

    /// The alignment of the buffer.
    #[inline]
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// The length of the buffer in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Return the buffer as a slice of bytes.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// Extracts the underlying `Bytes` from the buffer.
    pub fn into_inner(self) -> Bytes {
        self.bytes
    }

    /// Try to convert self into `AlignedBufferMut` if there is only a single strong reference.
    pub fn try_into_mut(self) -> Result<AlignedBufferMut, Self> {
        self.bytes
            .try_into_mut()
            .map(|bytes| AlignedBufferMut {
                bytes,
                alignment: self.alignment,
            })
            .map_err(|bytes| Self {
                bytes,
                alignment: self.alignment,
            })
    }

    /// Returns a slice of self for the provided range.
    ///
    /// FIXME(ngates): what should this do to the alignment? The underlying buffer is still
    ///  aligned... But the new sliced one might not be? Should we panic if the caller tries to
    ///  slice using unaligned indices?
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`, otherwise slicing
    /// will panic.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let len = self.len();
        let begin = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1).expect("out of range"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).expect("out of range"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => len,
        };

        assert!(
            begin <= end,
            "range start must not be greater than end: {:?} <= {:?}",
            begin,
            end,
        );
        assert!(
            end <= len,
            "range end out of bounds: {:?} <= {:?}",
            end,
            len,
        );

        if end == begin {
            // We prefer to return a new empty buffer instead of sharing this one and creating a
            // strong reference just to hold an empty slice.
            return AlignedBuffer::new_with_alignment(Bytes::new(), self.alignment);
        }

        // Currently this panics if the begin/end are not aligned to the buffer's alignment...
        // For unaligned access, the caller should go via `as_slice`.
        // Alternatively, we could add a slice_with_alignment call that relaxes the alignment.
        Self::new_with_alignment(self.bytes.slice(begin..end), self.alignment)
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[u8]> for AlignedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl From<Vec<u8>> for AlignedBuffer {
    fn from(value: Vec<u8>) -> Self {
        Self {
            bytes: Bytes::from(value),
            alignment: Alignment::of::<u8>(),
        }
    }
}
