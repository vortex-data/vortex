use std::collections::Bound;
use std::ops::RangeBounds;

use vortex_error::{vortex_panic, VortexExpect};

use crate::{AlignedBuffer, AlignedBufferMut, Alignment, ScalarBufferMut};

/// A buffer of Vortex primitive scalars.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarBuffer<T> {
    /// The underlying aligned buffer.
    /// We hold an `AlignedBuffer` instead of a `Bytes` to allow defining a wider alignment than
    /// the scalar type's alignment.
    pub(crate) buffer: AlignedBuffer,
    pub(crate) length: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

impl<T: Copy> ScalarBuffer<T> {
    /// Returns a new `ScalarBuffer<T>` copied from the provided `Vec<T>`.
    ///
    /// Due to our underlying usage of `bytes::Bytes`, we are unable to take zero-copy ownership
    /// of the provided `Vec<T>` while maintaining the ability to convert it back into a mutable
    /// buffer. We could fix this by forking `Bytes`, or in many other complex ways, but for now
    /// callers should prefer to construct `ScalarBuffer<T>` from a `ScalarBufferMut<T>`.
    pub fn copy_from_vec(vec: Vec<T>) -> Self {
        let byte_len = vec.len() * size_of::<T>();
        let mut buffer = AlignedBufferMut::with_capacity(byte_len, align_of::<T>().into());

        let raw_slice: &[u8] = unsafe { std::slice::from_raw_parts(vec.as_ptr().cast(), byte_len) };
        buffer.extend_from_slice(raw_slice);

        Self {
            buffer: buffer.freeze(),
            length: vec.len(),
            _marker: Default::default(),
        }
    }

    /// Returns the length of the buffer in elements of type T.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the buffer is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the alignment of the buffer.
    #[inline(always)]
    pub fn alignment(&self) -> Alignment {
        self.buffer.alignment()
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        let raw_slice = self.buffer.as_slice();
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(raw_slice.as_ptr().cast(), self.length) }
    }

    /// Returns an iterator over the buffer of elements of type T.
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.as_slice().iter().copied()
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
            Bound::Included(&n) => n.checked_mul(size_of::<T>()).vortex_expect("out of range"),
            Bound::Excluded(&n) => n
                .checked_mul(size_of::<T>())
                .and_then(|n| n.checked_add(size_of::<T>()))
                .vortex_expect("out of range"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n
                .checked_mul(size_of::<T>())
                .and_then(|n| n.checked_add(1))
                .vortex_expect("out of range"),
            Bound::Excluded(&n) => n.checked_mul(size_of::<T>()).vortex_expect("out of range"),
            Bound::Unbounded => len
                .checked_mul(size_of::<T>())
                .vortex_expect("out of range"),
        };

        Self {
            buffer: self.buffer.slice(begin..end),
            length: end - begin,
            _marker: Default::default(),
        }
    }

    /// Returns the underlying aligned buffer.
    pub fn into_inner(self) -> AlignedBuffer {
        self.buffer
    }

    /// Try to convert self into `ScalarBufferMut<T>` if there is only a single strong reference.
    pub fn try_into_mut(self) -> Result<ScalarBufferMut<T>, Self> {
        self.buffer
            .try_into_mut()
            .map(|buffer| ScalarBufferMut {
                buffer,
                length: self.length,
                _marker: Default::default(),
            })
            .map_err(|buffer| Self {
                buffer,
                length: self.length,
                _marker: Default::default(),
            })
    }
}

impl<T: Copy> From<AlignedBuffer> for ScalarBuffer<T> {
    fn from(buffer: AlignedBuffer) -> Self {
        if !buffer.alignment().is_multiple_of(align_of::<T>()) {
            vortex_panic!("Alignment must be a multiple of the scalar type's alignment");
        }
        let length = buffer.len() / size_of::<T>();
        Self {
            buffer,
            length,
            _marker: Default::default(),
        }
    }
}

impl<T: Sized + Copy> FromIterator<T> for ScalarBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        ScalarBufferMut::from_iter(iter).freeze()
    }
}
