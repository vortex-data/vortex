use core::mem::MaybeUninit;
use std::any::type_name;
use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};

use bytes::buf::UninitSlice;
use bytes::{Buf, BufMut, BytesMut};
use vortex_error::{VortexExpect, vortex_panic};

use crate::debug::TruncatedDebug;
use crate::spec_extend::SpecExtend;
use crate::{Alignment, Buffer, ByteBufferMut};

/// A mutable buffer that maintains a runtime-defined alignment through resizing operations.
#[derive(PartialEq, Eq)]
pub struct BufferMut<T> {
    pub(crate) bytes: BytesMut,
    pub(crate) length: usize,
    pub(crate) alignment: Alignment,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

impl<T> BufferMut<T> {
    /// Create a new `BufferMut` with the requested alignment and capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_aligned(capacity, Alignment::of::<T>())
    }

    /// Create a new `BufferMut` with the requested alignment and capacity.
    pub fn with_capacity_aligned(capacity: usize, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must align to the scalar type's alignment {}",
                alignment,
                align_of::<T>()
            );
        }

        let mut bytes = BytesMut::with_capacity((capacity * size_of::<T>()) + *alignment);
        bytes.align_empty(alignment);

        Self {
            bytes,
            length: 0,
            alignment,
            _marker: Default::default(),
        }
    }

    /// Create a new zeroed `BufferMut`.
    pub fn zeroed(len: usize) -> Self {
        Self::zeroed_aligned(len, Alignment::of::<T>())
    }

    /// Create a new zeroed `BufferMut`.
    pub fn zeroed_aligned(len: usize, alignment: Alignment) -> Self {
        let mut bytes = BytesMut::zeroed((len * size_of::<T>()) + *alignment);
        bytes.advance(bytes.as_ptr().align_offset(*alignment));
        unsafe { bytes.set_len(len * size_of::<T>()) };
        Self {
            bytes,
            length: len,
            alignment,
            _marker: Default::default(),
        }
    }

    /// Create a new empty `BufferMut` with the provided alignment.
    pub fn empty() -> Self {
        Self::empty_aligned(Alignment::of::<T>())
    }

    /// Create a new empty `BufferMut` with the provided alignment.
    pub fn empty_aligned(alignment: Alignment) -> Self {
        BufferMut::with_capacity_aligned(0, alignment)
    }

    /// Create a new full `BufferMut` with the given value.
    pub fn full(item: T, len: usize) -> Self
    where
        T: Copy,
    {
        let mut buffer = BufferMut::<T>::with_capacity(len);
        buffer.push_n(item, len);
        buffer
    }

    /// Create a mutable scalar buffer by copying the contents of the slice.
    pub fn copy_from(other: impl AsRef<[T]>) -> Self {
        Self::copy_from_aligned(other, Alignment::of::<T>())
    }

    /// Create a mutable scalar buffer with the alignment by copying the contents of the slice.
    ///
    /// ## Panics
    ///
    /// Panics when the requested alignment isn't itself aligned to type T.
    pub fn copy_from_aligned(other: impl AsRef<[T]>, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!("Given alignment is not aligned to type T")
        }
        let other = other.as_ref();
        let mut buffer = Self::with_capacity_aligned(other.len(), alignment);
        buffer.extend_from_slice(other);
        debug_assert_eq!(buffer.alignment(), alignment);
        buffer
    }

    /// Get the alignment of the buffer.
    #[inline(always)]
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Returns the length of the buffer.
    #[inline(always)]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.length, self.bytes.len() / size_of::<T>());
        self.length
    }

    /// Returns whether the buffer is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.bytes.capacity() / size_of::<T>()
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let raw_slice = self.bytes.as_ref();
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(raw_slice.as_ptr().cast(), self.length) }
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let raw_slice = self.bytes.as_mut();
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts_mut(raw_slice.as_mut_ptr().cast(), self.length) }
    }

    /// Clear the buffer, retaining any existing capacity.
    #[inline]
    pub fn clear(&mut self) {
        unsafe { self.bytes.set_len(0) }
        self.length = 0;
    }

    /// Shortens the buffer, keeping the first `len` bytes and dropping the
    /// rest.
    ///
    /// If `len` is greater than the buffer's current length, this has no
    /// effect.
    ///
    /// Existing underlying capacity is preserved.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len <= self.len() {
            // SAFETY: Shrinking the buffer cannot expose uninitialized bytes.
            unsafe { self.set_len(len) };
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the buffer.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let additional_bytes = additional * size_of::<T>();
        if additional_bytes <= self.bytes.capacity() - self.bytes.len() {
            // We can fit the additional bytes in the remaining capacity. Nothing to do.
            return;
        }

        // Otherwise, reserve additional + alignment bytes in case we need to realign the buffer.
        self.reserve_allocate(additional);
    }

    /// A separate function so we can inline the reserve call's fast path. According to `BytesMut`
    /// this has significant performance implications.
    fn reserve_allocate(&mut self, additional: usize) {
        let new_capacity: usize = ((self.length + additional) * size_of::<T>()) + *self.alignment;
        // Make sure we at least double in size each time we re-allocate to amortize the cost
        let new_capacity = new_capacity.max(self.bytes.capacity() * 2);

        let mut bytes = BytesMut::with_capacity(new_capacity);
        bytes.align_empty(self.alignment);
        bytes.extend_from_slice(&self.bytes);
        self.bytes = bytes;
    }

    /// Returns the spare capacity of the buffer as a slice of `MaybeUninit<T>`.
    /// Has identical semantics to [`Vec::spare_capacity_mut`].
    ///
    /// The returned slice can be used to fill the buffer with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`set_len`] method.
    ///
    /// [`set_len`]: BufferMut::set_len
    /// [`Vec::spare_capacity_mut`]: Vec::spare_capacity_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_buffer::BufferMut;
    ///
    /// // Allocate vector big enough for 10 elements.
    /// let mut b = BufferMut::<u64>::with_capacity(10);
    ///
    /// // Fill in the first 3 elements.
    /// let uninit = b.spare_capacity_mut();
    /// uninit[0].write(0);
    /// uninit[1].write(1);
    /// uninit[2].write(2);
    ///
    /// // Mark the first 3 elements of the vector as being initialized.
    /// unsafe {
    ///     b.set_len(3);
    /// }
    ///
    /// assert_eq!(b.as_slice(), &[0u64, 1, 2]);
    /// ```
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let dst = self.bytes.spare_capacity_mut().as_mut_ptr();
        unsafe {
            std::slice::from_raw_parts_mut(
                dst as *mut MaybeUninit<T>,
                self.capacity() - self.length,
            )
        }
    }

    /// # Safety
    /// The caller must ensure that the buffer was properly initialized up to `len`.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        unsafe { self.bytes.set_len(len * size_of::<T>()) };
        self.length = len;
    }

    /// Appends a scalar to the buffer.
    #[inline]
    pub fn push(&mut self, value: T) {
        self.reserve(1);
        unsafe { self.push_unchecked(value) }
    }

    /// Appends a scalar to the buffer without checking for sufficient capacity.
    ///
    /// ## Safety
    ///
    /// The caller must ensure there is sufficient capacity in the array.
    #[inline]
    pub unsafe fn push_unchecked(&mut self, item: T) {
        // SAFETY: the caller ensures we have sufficient capacity
        unsafe {
            let dst: *mut T = self.bytes.spare_capacity_mut().as_mut_ptr().cast();
            dst.write(item);
            self.bytes.set_len(self.bytes.len() + size_of::<T>())
        }
        self.length += 1;
    }

    /// Appends n scalars to the buffer.
    ///
    /// This function is slightly more optimized than `extend(iter::repeat_n(item, b))`.
    #[inline]
    pub fn push_n(&mut self, item: T, n: usize)
    where
        T: Copy,
    {
        self.reserve(n);
        unsafe { self.push_n_unchecked(item, n) }
    }

    /// Appends n scalars to the buffer.
    ///
    /// ## Safety
    ///
    /// The caller must ensure there is sufficient capacity in the array.
    #[inline]
    pub unsafe fn push_n_unchecked(&mut self, item: T, n: usize)
    where
        T: Copy,
    {
        let mut dst: *mut T = self.bytes.spare_capacity_mut().as_mut_ptr().cast();
        // SAFETY: we checked the capacity in the reserve call
        unsafe {
            for _ in 0..n {
                dst.write(item);
                dst = dst.add(1);
            }
            self.bytes.set_len(self.bytes.len() + (n * size_of::<T>()));
        }
        self.length += n;
    }

    /// Appends a slice of type `T`, growing the internal buffer as needed.
    ///
    /// # Example:
    ///
    /// ```
    /// # use vortex_buffer::BufferMut;
    ///
    /// let mut builder = BufferMut::<u16>::with_capacity(10);
    /// builder.extend_from_slice(&[42, 44, 46]);
    ///
    /// assert_eq!(builder.len(), 3);
    /// ```
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        self.reserve(slice.len());
        let raw_slice: &[u8] =
            unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), size_of_val(slice)) };
        self.bytes.extend_from_slice(raw_slice);
        self.length += slice.len();
    }

    /// Freeze the `BufferMut` into a `Buffer`.
    pub fn freeze(self) -> Buffer<T> {
        Buffer {
            bytes: self.bytes.freeze(),
            length: self.length,
            alignment: self.alignment,
            _marker: Default::default(),
        }
    }

    /// Map each element of the buffer with a closure.
    pub fn map_each<R, F>(self, mut f: F) -> BufferMut<R>
    where
        T: Copy,
        F: FnMut(T) -> R,
    {
        assert_eq!(
            size_of::<T>(),
            size_of::<R>(),
            "Size of T and R do not match"
        );
        // SAFETY: we have checked that `size_of::<T>` == `size_of::<R>`.
        let mut buf: BufferMut<R> = unsafe { std::mem::transmute(self) };
        buf.iter_mut()
            .for_each(|item| *item = f(unsafe { std::mem::transmute_copy(item) }));
        buf
    }

    /// Return a `BufferMut<T>` with the given alignment. Where possible, this will be zero-copy.
    pub fn aligned(self, alignment: Alignment) -> Self {
        if self.as_ptr().align_offset(*alignment) == 0 {
            self
        } else {
            Self::copy_from_aligned(self, alignment)
        }
    }
}

impl<T> Clone for BufferMut<T> {
    fn clone(&self) -> Self {
        // NOTE(ngates): we cannot derive Clone since BytesMut copies on clone and the alignment
        //  might be messed up.
        let mut buffer = BufferMut::<T>::with_capacity_aligned(self.capacity(), self.alignment);
        buffer.extend_from_slice(self.as_slice());
        buffer
    }
}

impl<T: Debug> Debug for BufferMut<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("BufferMut<{}>", type_name::<T>()))
            .field("length", &self.length)
            .field("alignment", &self.alignment)
            .field("as_slice", &TruncatedDebug(self.as_slice()))
            .finish()
    }
}

impl<T> Default for BufferMut<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> Deref for BufferMut<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for BufferMut<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for BufferMut<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for BufferMut<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Extend<T> for BufferMut<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        <Self as SpecExtend<T, I::IntoIter>>::spec_extend(self, iter.into_iter())
    }
}

impl<T> FromIterator<T> for BufferMut<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        // We don't infer the capacity here and just let the first call to `extend` do it for us.
        let mut buffer = Self::with_capacity(0);
        buffer.extend(iter);
        debug_assert_eq!(buffer.alignment(), Alignment::of::<T>());
        buffer
    }
}

impl Buf for ByteBufferMut {
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

/// As per the BufMut implementation, we must support internal resizing when
/// asked to extend the buffer.
/// See: <https://github.com/tokio-rs/bytes/issues/131>
unsafe impl BufMut for ByteBufferMut {
    #[inline]
    fn remaining_mut(&self) -> usize {
        usize::MAX - self.len()
    }

    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        if !cnt.is_multiple_of(*self.alignment) {
            vortex_panic!(
                "Cannot advance buffer by {} items, resulting alignment is not {}",
                cnt,
                self.alignment
            );
        }
        unsafe { self.bytes.advance_mut(cnt) };
        self.length -= cnt;
    }

    #[inline]
    fn chunk_mut(&mut self) -> &mut UninitSlice {
        self.bytes.chunk_mut()
    }

    fn put<T: Buf>(&mut self, mut src: T)
    where
        Self: Sized,
    {
        while src.has_remaining() {
            let chunk = src.chunk();
            self.extend_from_slice(chunk);
            src.advance(chunk.len());
        }
    }

    #[inline]
    fn put_slice(&mut self, src: &[u8]) {
        self.extend_from_slice(src);
    }

    #[inline]
    fn put_bytes(&mut self, val: u8, cnt: usize) {
        self.push_n(val, cnt)
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
            vortex_panic!("ByteBufferMut must be empty");
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

#[cfg(test)]
mod test {
    use bytes::{Buf, BufMut};

    use crate::{Alignment, BufferMut, ByteBufferMut, buffer_mut};

    #[test]
    fn capacity() {
        let mut n = 57;
        let mut buf = BufferMut::<i32>::with_capacity_aligned(n, Alignment::new(1024));
        assert!(buf.capacity() >= 57);

        while n > 0 {
            buf.push(0);
            assert!(buf.capacity() >= n);
            n -= 1
        }

        assert_eq!(buf.alignment(), Alignment::new(1024));
    }

    #[test]
    fn from_iter() {
        let buf = BufferMut::from_iter([0, 10, 20, 30]);
        assert_eq!(buf.as_slice(), &[0, 10, 20, 30]);
    }

    #[test]
    fn extend() {
        let mut buf = BufferMut::empty();
        buf.extend([0i32, 10, 20, 30]);
        buf.extend([40, 50, 60]);
        assert_eq!(buf.as_slice(), &[0, 10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn push() {
        let mut buf = BufferMut::empty();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn push_n() {
        let mut buf = BufferMut::empty();
        buf.push_n(0, 100);
        assert_eq!(buf.as_slice(), &[0; 100]);
    }

    #[test]
    fn as_mut() {
        let mut buf = buffer_mut![0, 1, 2];
        // Uses DerefMut
        buf[1] = 0;
        // Uses as_mut
        buf.as_mut()[2] = 0;
        assert_eq!(buf.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn map_each() {
        let buf = buffer_mut![0i32, 1, 2];
        // Add one, and cast to an unsigned u32 in the same closure
        let buf = buf.map_each(|i| (i + 1) as u32);
        assert_eq!(buf.as_slice(), &[1u32, 2, 3]);
    }

    #[test]
    fn bytes_buf() {
        let mut buf = ByteBufferMut::copy_from("helloworld".as_bytes());
        assert_eq!(buf.remaining(), 10);
        assert_eq!(buf.chunk(), b"helloworld");

        Buf::advance(&mut buf, 5);
        assert_eq!(buf.remaining(), 5);
        assert_eq!(buf.as_slice(), b"world");
        assert_eq!(buf.chunk(), b"world");
    }

    #[test]
    fn bytes_buf_mut() {
        let mut buf = ByteBufferMut::copy_from("hello".as_bytes());
        assert_eq!(BufMut::remaining_mut(&buf), usize::MAX - 5);

        BufMut::put_slice(&mut buf, b"world");
        assert_eq!(buf.as_slice(), b"helloworld");
    }
}
