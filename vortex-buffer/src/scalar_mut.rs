use std::ops::{Deref, DerefMut};

use vortex_error::vortex_panic;

use crate::{AlignedBufferMut, Alignment, ScalarBuffer};

/// A mutable buffer of Vortex primitive scalars.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarBufferMut<T> {
    pub(crate) buffer: AlignedBufferMut,
    pub(crate) length: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

impl<T: Sized + Copy> ScalarBufferMut<T> {
    /// Create a new `ScalarBufferMut` with the requested alignment and capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_aligned(capacity, Alignment::of::<T>())
    }

    /// Create a new `ScalarBufferMut` with the requested alignment and capacity.
    pub fn with_capacity_aligned(capacity: usize, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must align to the scalar type's alignment {}",
                alignment,
                align_of::<T>()
            );
        }
        Self {
            buffer: AlignedBufferMut::with_capacity(capacity * size_of::<T>(), alignment),
            length: 0,
            _marker: Default::default(),
        }
    }

    /// Create a mutable scalar buffer by copying the contents of an immutable `ScalarBuffer`.
    pub fn copy_from(other: &ScalarBuffer<T>) -> Self {
        let mut buffer = Self::with_capacity_aligned(other.len(), other.alignment());
        buffer.extend_from_slice(other.as_slice());
        buffer
    }

    /// Get the alignment of the buffer.
    #[inline(always)]
    pub fn alignment(&self) -> Alignment {
        self.buffer.alignment()
    }

    /// Returns the length of the buffer.
    #[inline(always)]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.length, self.buffer.len() / size_of::<T>());
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
        self.buffer.capacity() / size_of::<T>()
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let raw_slice = self.buffer.as_slice();
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(raw_slice.as_ptr().cast(), self.length) }
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let raw_slice = self.buffer.as_mut_slice();
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts_mut(raw_slice.as_mut_ptr().cast(), self.length) }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the buffer.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.buffer.reserve(additional * size_of::<T>());
    }

    /// # Safety
    /// The caller must ensure that the buffer was properly initialized up to `len`.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        unsafe { self.buffer.set_len(len * size_of::<T>()) };
        self.length = len;
    }

    /// Appends a scalar to the buffer.
    pub fn push(&mut self, value: T) {
        // NOTE(ngates): this assumes the platform is little-endian. Currently enforced
        //  with a flag cfg(target_endian = "little")
        let raw_ptr = &value as *const T as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(raw_ptr, size_of::<T>()) };

        // The extend_from_slice function will reserve additional space if required.
        self.buffer.extend_from_slice(bytes);
        self.length += 1;
    }

    /// Appends a slice of type `T`, growing the internal buffer as needed.
    ///
    /// # Example:
    ///
    /// ```
    /// # use vortex_buffer::ScalarBufferMut;
    ///
    /// let mut builder = ScalarBufferMut::<u16>::with_capacity(10);
    /// builder.extend_from_slice(&[42, 44, 46]);
    ///
    /// assert_eq!(builder.len(), 3);
    /// ```
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        let raw_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr().cast(), slice.len() * size_of::<T>())
        };
        self.buffer.extend_from_slice(raw_slice);
        self.length += slice.len();
    }

    /// Freeze the `ScalarBufferMut` into a `ScalarBuffer`.
    pub fn freeze(self) -> ScalarBuffer<T> {
        ScalarBuffer::from(self.buffer.freeze())
    }
}

impl<T: Sized + Copy> Deref for ScalarBufferMut<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Sized + Copy> DerefMut for ScalarBufferMut<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Sized + Copy> Extend<T> for ScalarBufferMut<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        // TODO(ngates): check out the Arrow extend_from_slice optimizations
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        self.reserve(lower);
        for value in iter {
            self.push(value);
        }
    }
}

impl<T: Sized + Copy> FromIterator<T> for ScalarBufferMut<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut buffer = Self::with_capacity(0);
        buffer.extend(iter);
        buffer
    }
}
