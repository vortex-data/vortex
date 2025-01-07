use std::collections::Bound;
use std::fmt::{Debug, Formatter};
use std::ops::{Deref, RangeBounds};

use bytes::{Buf, Bytes};
use vortex_error::{vortex_panic, VortexExpect};

use crate::{Alignment, BufferMut, ByteBuffer};

/// An immutable buffer of items of `T`.
#[derive(Clone, PartialEq, Eq, PartialOrd)]
pub struct Buffer<T> {
    pub(crate) bytes: Bytes,
    pub(crate) length: usize,
    pub(crate) alignment: Alignment,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

impl<T> Buffer<T> {
    /// Returns a new `Buffer<T>` copied from the provided `Vec<T>`, `&[T]`, etc.
    ///
    /// Due to our underlying usage of `bytes::Bytes`, we are unable to take zero-copy ownership
    /// of the provided `Vec<T>` while maintaining the ability to convert it back into a mutable
    /// buffer. We could fix this by forking `Bytes`, or in many other complex ways, but for now
    /// callers should prefer to construct `Buffer<T>` from a `BufferMut<T>`.
    pub fn copy_from(values: impl AsRef<[T]>) -> Self {
        BufferMut::copy_from(values).freeze()
    }

    /// Returns a new `Buffer<T>` copied from the provided slice and with the requested alignment.
    pub fn copy_from_aligned(values: impl AsRef<[T]>, alignment: Alignment) -> Self {
        BufferMut::copy_from_aligned(values, alignment).freeze()
    }

    /// Create a new empty `ByteBuffer` with the provided alignment.
    pub fn empty() -> Self {
        BufferMut::empty().freeze()
    }

    /// Create a new empty `ByteBuffer` with the provided alignment.
    pub fn empty_aligned(alignment: Alignment) -> Self {
        BufferMut::empty_aligned(alignment).freeze()
    }

    /// Create a new full `ByteBuffer` with the given value.
    pub fn full(item: T, len: usize) -> Self
    where
        T: Copy,
    {
        BufferMut::full(item, len).freeze()
    }

    /// Create a `Buffer<T>` zero-copy from a `ByteBuffer`.
    ///
    /// ## Panics
    ///
    /// Panics if the buffer is not aligned to the size of `T`, or the length is not a multiple of
    /// the size of `T`.
    pub fn from_byte_buffer(buffer: ByteBuffer) -> Self {
        // TODO(ngates): should this preserve the current alignment of the buffer?
        Self::from_byte_buffer_aligned(buffer, Alignment::of::<T>())
    }

    /// Create a `Buffer<T>` zero-copy from a `ByteBuffer`.
    ///
    /// ## Panics
    ///
    /// Panics if the buffer is not aligned to the given alignment, if the length is not a multiple
    /// of the size of `T`, or if the given alignment is not aligned to that of `T`.
    pub fn from_byte_buffer_aligned(buffer: ByteBuffer, alignment: Alignment) -> Self {
        Self::from_bytes_aligned(buffer.into_inner(), alignment)
    }

    /// Create a `Buffer<T>` zero-copy from a `Bytes`.
    ///
    /// ## Panics
    ///
    /// Panics if the buffer is not aligned to the size of `T`, or the length is not a multiple of
    /// the size of `T`.
    pub fn from_bytes_aligned(bytes: Bytes, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must be compatible with the scalar type's alignment {}",
                alignment,
                Alignment::of::<T>(),
            );
        }
        if bytes.as_ptr().align_offset(*alignment) != 0 {
            vortex_panic!(
                "Bytes alignment must align to the scalar type's alignment {}",
                Alignment::of::<T>()
            );
        }
        if bytes.len() % size_of::<T>() != 0 {
            vortex_panic!(
                "Bytes length {} must be a multiple of the scalar type's size {}",
                bytes.len(),
                size_of::<T>()
            );
        }
        let length = bytes.len() / size_of::<T>();
        Self {
            bytes,
            length,
            alignment,
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
        self.alignment
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        let raw_slice = self.bytes.as_ref();
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(raw_slice.as_ptr().cast(), self.length) }
    }

    /// Returns an iterator over the buffer of elements of type T.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.as_slice().iter()
    }

    /// Returns a slice of self for the provided range.
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`.
    /// Also requires that both `begin` and `end` are aligned to the buffer's required alignment.
    #[inline(always)]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        self.slice_with_alignment(range, self.alignment)
    }

    /// Returns a slice of self for the provided range, with no guarantees about the resulting
    /// alignment.
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`.
    #[inline(always)]
    pub fn slice_unaligned(&self, range: impl RangeBounds<usize>) -> Self {
        self.slice_with_alignment(range, Alignment::of::<u8>())
    }

    /// Returns a slice of self for the provided range, ensuring the resulting slice has the
    /// given alignment.
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`.
    /// Also requires that both `begin` and `end` are aligned to the given alignment.
    pub fn slice_with_alignment(
        &self,
        range: impl RangeBounds<usize>,
        alignment: Alignment,
    ) -> Self {
        let len = self.len();
        let begin = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1).vortex_expect("out of range"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).vortex_expect("out of range"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => len,
        };

        if begin > end {
            vortex_panic!(
                "range start must not be greater than end: {:?} <= {:?}",
                begin,
                end
            );
        }
        if end > len {
            vortex_panic!("range end out of bounds: {:?} <= {:?}", end, len);
        }

        if end == begin {
            // We prefer to return a new empty buffer instead of sharing this one and creating a
            // strong reference just to hold an empty slice.
            return Self::empty_aligned(alignment);
        }

        let begin_byte = begin * size_of::<T>();
        let end_byte = end * size_of::<T>();

        if !begin_byte.is_multiple_of(*alignment) {
            vortex_panic!("range start must be aligned to {:?}", alignment);
        }
        if !end_byte.is_multiple_of(*alignment) {
            vortex_panic!("range end must be aligned to {:?}", alignment);
        }
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!("Slice alignment must at least align to type T")
        }

        Self {
            bytes: self.bytes.slice(begin_byte..end_byte),
            length: end - begin,
            alignment,
            _marker: Default::default(),
        }
    }

    /// Returns the underlying aligned buffer.
    pub fn into_inner(self) -> Bytes {
        self.bytes
    }

    /// Return the ByteBuffer for this `Buffer<T>`.
    pub fn into_byte_buffer(self) -> ByteBuffer {
        ByteBuffer {
            bytes: self.bytes,
            length: self.length * size_of::<T>(),
            alignment: self.alignment,
            _marker: Default::default(),
        }
    }

    /// Convert self into `BufferMut<T>`, copying if there are multiple strong references.
    pub fn into_mut(self) -> BufferMut<T> {
        self.try_into_mut()
            .unwrap_or_else(|buffer| BufferMut::<T>::copy_from(&buffer))
    }

    /// Try to convert self into `BufferMut<T>` if there is only a single strong reference.
    pub fn try_into_mut(self) -> Result<BufferMut<T>, Self> {
        self.bytes
            .try_into_mut()
            .map(|bytes| BufferMut {
                bytes,
                length: self.length,
                alignment: self.alignment,
                _marker: Default::default(),
            })
            .map_err(|bytes| Self {
                bytes,
                length: self.length,
                alignment: self.alignment,
                _marker: Default::default(),
            })
    }

    /// Return a `Buffer<T>` with the given alignment. Where possible, this will be zero-copy.
    pub fn aligned(self, alignment: Alignment) -> Self {
        if self.as_ptr().align_offset(*alignment) == 0 {
            self
        } else {
            Self::copy_from_aligned(self, alignment)
        }
    }
}

impl<T> Debug for Buffer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        const TRUNC_SIZE: usize = 512;
        let mut binding = f.debug_struct("Buffer");
        let mut fields = binding
            .field("length", &self.length)
            .field("alignment", &self.alignment);

        let mut bytes = self.bytes.clone();
        if bytes.len() > TRUNC_SIZE {
            fields = fields.field("truncated", &true);
        }

        bytes.truncate(TRUNC_SIZE);
        fields.field("bytes", &bytes).finish()
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> AsRef<[T]> for Buffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> FromIterator<T> for Buffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        BufferMut::from_iter(iter).freeze()
    }
}

/// Only for `Buffer<u8>` can we zero-copy from a `Vec<u8>` since we can use a 1-byte alignment.
impl From<Vec<u8>> for ByteBuffer {
    fn from(value: Vec<u8>) -> Self {
        Self::from(Bytes::from(value))
    }
}

/// Only for `Buffer<u8>` can we zero-copy from a `Bytes` since we can use a 1-byte alignment.
impl From<Bytes> for ByteBuffer {
    fn from(bytes: Bytes) -> Self {
        let length = bytes.len();
        Self {
            bytes,
            length,
            alignment: Alignment::of::<u8>(),
            _marker: Default::default(),
        }
    }
}

impl Buf for ByteBuffer {
    fn remaining(&self) -> usize {
        self.len()
    }

    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

    fn advance(&mut self, cnt: usize) {
        if !cnt.is_multiple_of(*self.alignment) {
            vortex_panic!(
                "Cannot advance buffer by {} items, resulting alignment is not {}",
                cnt,
                self.alignment
            );
        }
        self.bytes.advance(cnt);
        self.length -= cnt;
    }
}

/// Owned iterator over a `Buffer<T>`.
pub struct BufferIterator<T> {
    buffer: Buffer<T>,
    index: usize,
}

impl<T: Copy> Iterator for BufferIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.buffer.len()).then(move || {
            let value = self.buffer.as_slice()[self.index];
            self.index += 1;
            value
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buffer.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<T: Copy> IntoIterator for Buffer<T> {
    type Item = T;
    type IntoIter = BufferIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        BufferIterator {
            buffer: self,
            index: 0,
        }
    }
}

impl<T> From<BufferMut<T>> for Buffer<T> {
    fn from(value: BufferMut<T>) -> Self {
        value.freeze()
    }
}

#[cfg(test)]
mod test {
    use bytes::Buf;

    use crate::{buffer, ByteBuffer};

    #[test]
    fn slice() {
        let buf = buffer![0, 1, 2, 3, 4];
        assert_eq!(buf.slice(1..3).as_slice(), &[1, 2]);
        assert_eq!(buf.slice(1..=3).as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn slice_unaligned() {
        let buf = buffer![0i32, 1, 2, 3, 4].into_byte_buffer();
        // With a regular slice, this would panic. See [`slice_bad_alignment`].
        buf.slice_unaligned(1..2);
    }

    #[test]
    #[should_panic]
    fn slice_bad_alignment() {
        let buf = buffer![0i32, 1, 2, 3, 4].into_byte_buffer();
        // We should only be able to slice this buffer on 4-byte (i32) boundaries.
        buf.slice(1..2);
    }

    #[test]
    fn bytes_buf() {
        let mut buf = ByteBuffer::copy_from("helloworld".as_bytes());
        assert_eq!(buf.remaining(), 10);
        assert_eq!(buf.chunk(), b"helloworld");

        Buf::advance(&mut buf, 5);
        assert_eq!(buf.remaining(), 5);
        assert_eq!(buf.as_slice(), b"world");
        assert_eq!(buf.chunk(), b"world");
    }
}
