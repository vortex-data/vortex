// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::cmp::Ordering;
use std::collections::Bound;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::RangeBounds;

use bytes::Buf;
use bytes::Bytes;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::Alignment;
use crate::BufferMut;
use crate::ByteBuffer;
use crate::debug::TruncatedDebug;
use crate::trusted_len::TrustedLen;

/// An immutable buffer of items of `T`.
#[derive(Clone)]
pub struct Buffer<T> {
    pub(crate) bytes: Bytes,
    pub(crate) length: usize,
    pub(crate) alignment: Alignment,
    pub(crate) _marker: PhantomData<T>,
}

/// Zero-length backing for empty buffers, "aligned" to [`Alignment::MAX`] so it satisfies any
/// valid alignment without allocating. A zero-length slice never reads memory, so it may use a
/// dangling pointer as long as it is non-null and aligned.
const EMPTY_BACKING: &[u8] = {
    let addr = 1usize << 20;
    assert!(Alignment::MAX.is_offset_aligned(addr));
    // SAFETY: the pointer is non-null and aligned, and the slice is zero-length.
    unsafe { std::slice::from_raw_parts(std::ptr::without_provenance(addr), 0) }
};

impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self {
            bytes: Bytes::from_static(EMPTY_BACKING),
            length: 0,
            alignment: Alignment::of::<T>(),
            _marker: PhantomData,
        }
    }
}

impl<T> PartialEq for Buffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl<T> Eq for Buffer<T> {}

impl<T> Ord for Buffer<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.bytes.cmp(&other.bytes)
    }
}

impl<T> PartialOrd for Buffer<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Hash for Buffer<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes.as_ref().hash(state)
    }
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
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`copy_from_preferred_aligned`] to control the over-alignment.
    ///
    /// [`copy_from_preferred_aligned`]: Self::copy_from_preferred_aligned
    pub fn copy_from_aligned(values: impl AsRef<[T]>, alignment: Alignment) -> Self {
        Self::copy_from_preferred_aligned(values, alignment, Some(Alignment::DEFAULT_ALIGNMENT))
    }

    /// Returns a new `Buffer<T>` copied from the provided slice and with the requested alignment.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn copy_from_preferred_aligned(
        values: impl AsRef<[T]>,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        BufferMut::copy_from_preferred_aligned(values, alignment, preferred_alignment).freeze()
    }

    /// Create a new zeroed `Buffer` with the given value.
    pub fn zeroed(len: usize) -> Self {
        Self::zeroed_aligned(len, Alignment::of::<T>())
    }

    /// Create a new zeroed `Buffer` with the requested alignment.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`zeroed_preferred_aligned`] to control the over-alignment.
    ///
    /// [`zeroed_preferred_aligned`]: Self::zeroed_preferred_aligned
    pub fn zeroed_aligned(len: usize, alignment: Alignment) -> Self {
        Self::zeroed_preferred_aligned(len, alignment, Some(Alignment::DEFAULT_ALIGNMENT))
    }

    /// Create a new zeroed `Buffer` with the requested alignment.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn zeroed_preferred_aligned(
        len: usize,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        BufferMut::zeroed_preferred_aligned(len, alignment, preferred_alignment).freeze()
    }

    /// Create a new empty `ByteBuffer` with the provided alignment.
    pub fn empty() -> Self {
        Self::empty_aligned(Alignment::of::<T>())
    }

    /// Create a new empty `ByteBuffer` with the provided alignment.
    ///
    /// This does not allocate: empty buffers are backed by a zero-length `Bytes` that is
    /// aligned to [`Alignment::MAX`].
    pub fn empty_aligned(alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must align to the scalar type's alignment {}",
                alignment,
                Alignment::of::<T>(),
            );
        }
        Self {
            bytes: Bytes::from_static(EMPTY_BACKING),
            length: 0,
            alignment,
            _marker: PhantomData,
        }
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
        if !alignment.is_ptr_aligned(bytes.as_ptr()) {
            vortex_panic!(
                "Bytes alignment must align to the requested alignment {}",
                alignment,
            );
        }
        if !bytes.len().is_multiple_of(size_of::<T>()) {
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

    /// Create a buffer with values from the TrustedLen iterator.
    /// Should be preferred over `from_iter` when the iterator is known to be `TrustedLen`.
    pub fn from_trusted_len_iter<I: TrustedLen<Item = T>>(iter: I) -> Self {
        let (_, upper_bound) = iter.size_hint();
        let mut buffer = BufferMut::with_capacity(
            upper_bound.vortex_expect("TrustedLen iterator has no upper bound"),
        );
        buffer.extend_trusted(iter);
        buffer.freeze()
    }

    /// Map each element of the buffer with a closure.
    pub fn map_each_in_place<R, F>(self, mut f: F) -> BufferMut<R>
    where
        T: Copy,
        F: FnMut(T) -> R,
    {
        match self.try_into_mut() {
            Ok(mut_buf) => mut_buf.map_each_in_place(f),
            Err(buf) => {
                let len = buf.len();
                let mut out_buf = BufferMut::with_capacity(len);
                out_buf
                    .spare_capacity_mut()
                    .iter_mut()
                    .zip(buf)
                    .for_each(|(out, in_)| {
                        out.write(f(in_));
                    });
                // Safety: just assigned to each value
                unsafe { out_buf.set_len(len) }
                out_buf
            }
        }
    }

    /// Clear the buffer, preserving existing capacity.
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.length = 0;
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
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), self.length) }
    }

    /// Return a view over the buffer as an opaque byte slice.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// Returns an iterator over the buffer of elements of type T.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.as_slice().iter(),
        }
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
            vortex_panic!("range end out of bounds: {:?} > {:?}", end, len);
        }

        if end == begin {
            // We prefer to return a new empty buffer instead of sharing this one and creating a
            // strong reference just to hold an empty slice.
            return Self::empty_aligned(alignment);
        }

        let begin_byte = begin * size_of::<T>();
        let end_byte = end * size_of::<T>();

        if !alignment.is_offset_aligned(begin_byte) {
            vortex_panic!(
                "range start must be aligned to {alignment:?}, byte {}",
                begin_byte
            );
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

    /// Returns a slice of self that is equivalent to the given subset.
    ///
    /// When processing the buffer you will often end up with `&[T]` that is a subset
    /// of the underlying buffer. This function turns the slice into a slice of the buffer
    /// it has been taken from.
    ///
    /// # Panics:
    /// Requires that the given sub slice is in fact contained within the Bytes buffer; otherwise this function will panic.
    #[inline(always)]
    pub fn slice_ref(&self, subset: &[T]) -> Self {
        self.slice_ref_with_alignment(subset, Alignment::of::<T>())
    }

    /// Returns a slice of self that is equivalent to the given subset.
    ///
    /// When processing the buffer you will often end up with `&[T]` that is a subset
    /// of the underlying buffer. This function turns the slice into a slice of the buffer
    /// it has been taken from.
    ///
    /// # Panics:
    /// Requires that the given sub slice is in fact contained within the Bytes buffer; otherwise this function will panic.
    /// Also requires that the given alignment aligns to the type of slice and is smaller or equal to the buffers alignment
    pub fn slice_ref_with_alignment(&self, subset: &[T], alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!("slice_ref alignment must at least align to type T")
        }

        if !self.alignment.is_aligned_to(alignment) {
            vortex_panic!("slice_ref subset alignment must at least align to the buffer alignment")
        }

        if !alignment.is_ptr_aligned(subset.as_ptr()) {
            vortex_panic!("slice_ref subset must be aligned to {:?}", alignment);
        }

        let subset_u8 =
            unsafe { std::slice::from_raw_parts(subset.as_ptr().cast(), size_of_val(subset)) };

        Self {
            bytes: self.bytes.slice_ref(subset_u8),
            length: subset.len(),
            alignment,
            _marker: Default::default(),
        }
    }

    /// Returns the underlying aligned buffer.
    pub fn inner(&self) -> &Bytes {
        debug_assert_eq!(
            self.length * size_of::<T>(),
            self.bytes.len(),
            "Own length has to be the same as the underlying bytes length"
        );
        &self.bytes
    }

    /// Returns the underlying aligned buffer.
    pub fn into_inner(self) -> Bytes {
        debug_assert_eq!(
            self.length * size_of::<T>(),
            self.bytes.len(),
            "Own length has to be the same as the underlying bytes length"
        );
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

    /// Convert self into `BufferMut<T>`, cloning the data if there are multiple strong references.
    pub fn into_mut(self) -> BufferMut<T> {
        self.try_into_mut()
            .unwrap_or_else(|buffer| BufferMut::<T>::copy_from_aligned(&buffer, buffer.alignment))
    }

    /// Returns whether a `Buffer<T>` is aligned to the given alignment.
    pub fn is_aligned(&self, alignment: Alignment) -> bool {
        alignment.is_ptr_aligned(self.bytes.as_ptr())
    }

    /// Return a `Buffer<T>` with the given alignment. Where possible, this will be zero-copy.
    pub fn aligned(mut self, alignment: Alignment) -> Self {
        if alignment.is_ptr_aligned(self.as_ptr()) {
            self.alignment = alignment;
            self
        } else {
            #[cfg(feature = "warn-copy")]
            {
                let bt = std::backtrace::Backtrace::capture();
                tracing::warn!(
                    "Buffer is not aligned to requested alignment {alignment}, copying: {bt}"
                )
            }
            Self::copy_from_aligned(self, alignment)
        }
    }

    /// Return a `Buffer<T>` with the given alignment. Panics if the buffer is not aligned.
    pub fn ensure_aligned(mut self, alignment: Alignment) -> Self {
        if alignment.is_ptr_aligned(self.as_ptr()) {
            self.alignment = alignment;
            self
        } else {
            vortex_panic!("Buffer is not aligned to requested alignment {}", alignment)
        }
    }
}

impl<T> Buffer<T> {
    /// Transmute a `Buffer<T>` into a `Buffer<U>`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all possible bit representations of type `T` are valid when
    /// interpreted as type `U`.
    /// See [`std::mem::transmute`] for more details.
    ///
    /// # Panics
    ///
    /// Panics if the type `U` does not have the same size and alignment as `T`.
    pub unsafe fn transmute<U>(self) -> Buffer<U> {
        assert_eq!(size_of::<T>(), size_of::<U>(), "Buffer type size mismatch");
        assert_eq!(
            align_of::<T>(),
            align_of::<U>(),
            "Buffer type alignment mismatch"
        );

        Buffer {
            bytes: self.bytes,
            length: self.length,
            alignment: self.alignment,
            _marker: PhantomData,
        }
    }
}

/// An iterator over Buffer elements.
///
/// This is an analog to the `std::slice::Iter` type.
pub struct Iter<'a, T> {
    inner: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.inner.last()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n)
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T: Debug> Debug for Buffer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("Buffer<{}>", type_name::<T>()))
            .field("length", &self.length)
            .field("alignment", &self.alignment)
            .field("as_slice", &TruncatedDebug(self.as_slice()))
            .finish()
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> AsRef<[T]> for Buffer<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> FromIterator<T> for Buffer<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        BufferMut::from_iter(iter).freeze()
    }
}

// Helper struct to allow us to zero-copy any vec into a buffer
#[repr(transparent)]
struct Wrapper<T>(Vec<T>);

impl<T> AsRef<[u8]> for Wrapper<T> {
    fn as_ref(&self) -> &[u8] {
        let data = self.0.as_ptr().cast::<u8>();
        let len = self.0.len() * size_of::<T>();
        unsafe { std::slice::from_raw_parts(data, len) }
    }
}

impl<T> From<Vec<T>> for Buffer<T>
where
    T: Send + 'static,
{
    fn from(value: Vec<T>) -> Self {
        let original_len = value.len();
        let wrapped_vec = Wrapper(value);

        let bytes = Bytes::from_owner(wrapped_vec);

        assert_eq!(bytes.as_ptr().align_offset(align_of::<T>()), 0);

        Self {
            bytes,
            length: original_len,
            alignment: Alignment::of::<T>(),
            _marker: PhantomData,
        }
    }
}

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
    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        if !self.alignment.is_offset_aligned(cnt) {
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

/// Owned iterator over a [`Buffer`].
pub struct BufferIterator<T: Copy> {
    // Keep the buffer alive for the duration of the iteration.
    _buffer: Buffer<T>,
    ptr: *const T,
    end: *const T,
}

impl<T: Copy> Iterator for BufferIterator<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            // SAFETY: ptr is within the buffer and has not reached end.
            let value = unsafe { self.ptr.read() };
            self.ptr = unsafe { self.ptr.add(1) };
            Some(value)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = unsafe { self.end.offset_from(self.ptr) } as usize;
        (remaining, Some(remaining))
    }
}

impl<T: Copy> ExactSizeIterator for BufferIterator<T> {}

impl<T: Copy> IntoIterator for Buffer<T> {
    type Item = T;
    type IntoIter = BufferIterator<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let ptr = self.as_slice().as_ptr();
        let end = unsafe { ptr.add(self.len()) };
        BufferIterator {
            _buffer: self,
            ptr,
            end,
        }
    }
}

impl<T> From<BufferMut<T>> for Buffer<T> {
    #[inline]
    fn from(value: BufferMut<T>) -> Self {
        value.freeze()
    }
}

#[cfg(test)]
mod test {
    use bytes::Buf;

    use crate::Alignment;
    use crate::Buffer;
    use crate::ByteBuffer;
    use crate::buffer;

    #[test]
    fn align() {
        let buf = buffer![0u8, 1, 2];
        let aligned = buf.aligned(Alignment::new(32));
        assert_eq!(aligned.alignment(), Alignment::new(32));
        assert_eq!(aligned.as_slice(), &[0, 1, 2]);
    }

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
        let sliced = buf.slice_unaligned(1..2);
        // Verify the slice has the expected length (1 byte from index 1 to 2).
        assert_eq!(sliced.len(), 1);
        // The original buffer has i32 values [0, 1, 2, 3, 4].
        // In little-endian bytes, 0i32 = [0, 0, 0, 0], so byte at index 1 is 0.
        assert_eq!(sliced.as_slice(), &[0]);
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

        buf.advance(5);
        assert_eq!(buf.remaining(), 5);
        assert_eq!(buf.as_slice(), b"world");
        assert_eq!(buf.chunk(), b"world");
    }

    #[test]
    fn buffer_zeroed() {
        const LEN: usize = 17;

        let buf = Buffer::<u32>::zeroed(LEN);

        assert!(buf.is_aligned(Alignment::of::<u32>()));
        assert_eq!(buf.as_slice(), &[0; LEN]);
    }

    #[test]
    fn buffer_zeroed_aligned() {
        const LEN: usize = 17;
        let alignment = Alignment::new(64);

        let buf = Buffer::<u32>::zeroed_aligned(LEN, alignment);

        assert!(buf.is_aligned(alignment));
        assert_eq!(buf.as_slice(), &[0; LEN]);
    }

    #[test]
    fn copy_from_over_aligns_to_default() {
        let values = [1u32, 2, 3];
        let buf = Buffer::<u32>::copy_from(values);

        // The buffer reports the scalar type's alignment, ...
        assert_eq!(buf.alignment(), Alignment::of::<u32>());
        // ... but the underlying allocation is over-aligned to DEFAULT_ALIGNMENT.
        assert!(buf.is_aligned(Alignment::DEFAULT_ALIGNMENT));
        assert_eq!(buf.as_slice(), &values);
    }

    #[test]
    fn zeroed_over_aligns_to_default() {
        const LEN: usize = 17;

        let buf = Buffer::<u32>::zeroed(LEN);

        assert_eq!(buf.alignment(), Alignment::of::<u32>());
        assert!(buf.is_aligned(Alignment::DEFAULT_ALIGNMENT));
        assert_eq!(buf.as_slice(), &[0; LEN]);
    }

    #[test]
    fn from_vec() {
        let vec = vec![1, 2, 3, 4, 5];
        let buff = Buffer::from(vec.clone());
        assert!(buff.is_aligned(Alignment::of::<i32>()));
        assert_eq!(vec, buff.as_ref());
    }

    #[test]
    fn empty_aligned_max_alignment() {
        // Empty buffers are backed by a static and must satisfy any valid alignment.
        let buf = Buffer::<u8>::empty_aligned(Alignment::MAX);
        assert!(buf.is_empty());
        assert!(buf.is_aligned(Alignment::MAX));
    }

    #[test]
    fn empty_slice_preserves_alignment() {
        let buf = Buffer::<u64>::zeroed_aligned(8, Alignment::new(64));
        let sliced = buf.slice(0..0);
        assert!(sliced.is_empty());
        assert_eq!(sliced.alignment(), Alignment::new(64));
        assert!(sliced.is_aligned(Alignment::new(64)));
    }

    #[test]
    fn empty_into_mut_preserves_alignment() {
        let buf = Buffer::<u8>::empty_aligned(Alignment::new(64));
        let buf_mut = buf.into_mut();
        assert_eq!(buf_mut.alignment(), Alignment::new(64));
        assert!(buf_mut.is_empty());
    }

    #[test]
    fn test_slice_unaligned_end_pos() {
        let data = vec![0u8; 2];
        // Overalign the u8 vector.
        let aligned_buffer = Buffer::copy_from_aligned(&data, Alignment::new(8));
        // Previously, `Buffer::slice` incorrectly asserted that the end position
        // must be aligned. That assertion has been removed such that the end
        // position can be arbitrary and only the beginning of the slice needs
        // to be aligned.
        aligned_buffer.slice(0..1);
    }

    #[test]
    fn test_empty_equality() {
        let a = Buffer::<u16>::empty();
        let b = Buffer::<u16>::empty();

        assert_eq!(a, b);
    }
}
