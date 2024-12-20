use std::ops::{Deref, DerefMut};

use bytes::{Buf, BytesMut};
use vortex_error::{vortex_panic, VortexExpect};

use crate::{Alignment, Buffer};

/// A mutable buffer that maintains a runtime-defined alignment through resizing operations.
#[derive(Clone, Debug, PartialEq, Eq)]
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

    /// Create a mutable scalar buffer by copying the contents of an immutable `Buffer`.
    pub fn copy_from(other: &Buffer<T>) -> Self {
        let mut buffer = Self::with_capacity_aligned(other.len(), other.alignment());
        buffer.extend_from_slice(other.as_slice());
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
        // FIXME(ngates): test whether this is correct, since capacity is not always divisble
        //  by the size of T due to over-allocating for alignment.
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

        // NOTE(ngates): this assumes the platform is little-endian. Currently enforced
        //  with a flag cfg(target_endian = "little")
        let raw_ptr = &value as *const T as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(raw_ptr, size_of::<T>()) };

        let dst = self.bytes.as_mut_ptr();
        // SAFETY: we checked the capacity in the reserve call
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, size_of::<T>());
            self.bytes.set_len(self.bytes.len() + size_of::<T>())
        }
        self.length += 1;
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
    pub fn map_each<R, F>(mut self, mut f: F) -> BufferMut<R>
    where
        F: FnMut(&T) -> R,
    {
        {
            let raw_src = self.as_ptr();
            let src = unsafe { std::slice::from_raw_parts(raw_src, self.len()) };

            let dst: &mut [R] = unsafe { std::mem::transmute(self.as_mut()) };
            dst.iter_mut().zip(src.iter()).for_each(|(d, s)| *d = f(s));
        }
        // SAFETY: we didn't change the length of the buffer or its alignment.
        unsafe { std::mem::transmute(self) }
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
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let mut iterator = iter.into_iter();

        // Attempt to reserve enough memory up-front, although this is only a lower bound.
        let (lower, _) = iterator.size_hint();
        self.reserve(lower);

        let item_size = size_of::<T>();

        let remaining = self.capacity() - self.len();
        let mut consumed = 0;
        let mut dst = unsafe { self.bytes.as_mut_ptr().add(self.len() * item_size) };

        while consumed < remaining {
            if let Some(item) = iterator.next() {
                // SAFETY: We know we have enough capacity to write the item.
                unsafe {
                    let raw_ptr = &item as *const T as *const u8;
                    let bytes = std::slice::from_raw_parts(raw_ptr, size_of::<T>());
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, item_size);
                    dst = dst.add(item_size);
                }
                consumed += 1;
            } else {
                break;
            }
        }

        self.length += consumed;
        unsafe { self.bytes.set_len(self.length * item_size) };

        iterator.for_each(|item| self.push(item));
    }
}

impl<T> FromIterator<T> for BufferMut<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut buffer = Self::with_capacity(0);
        buffer.extend(iter);
        buffer
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
