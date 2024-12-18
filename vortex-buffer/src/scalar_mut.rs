use vortex_dtype::NativePType;
use vortex_error::vortex_panic;

use crate::{AlignedBufferMut, Alignment, ScalarBuffer};

/// A mutable buffer of Vortex primitive scalars.
pub struct ScalarBufferMut<T: NativePType> {
    pub(crate) buffer: AlignedBufferMut,
    pub(crate) length: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

impl<T: NativePType> ScalarBufferMut<T> {
    /// Create a new `ScalarBufferMut` with the requested alignment and capacity.
    pub fn new(capacity: usize) -> Self {
        Self::new_aligned(capacity, Alignment::of::<T>())
    }

    /// Create a new `ScalarBufferMut` with the requested alignment and capacity.
    pub fn new_aligned(capacity: usize, alignment: Alignment) -> Self {
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
        let mut buffer = Self::new_aligned(other.len(), other.alignment());
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

    /// Reserves capacity for at least `additional` more elements to be inserted in the buffer.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.buffer.reserve(additional * size_of::<T>());
    }

    /// Appends a scalar to the buffer.
    pub fn push(&mut self, value: T) {
        self.reserve(1);
        self.buffer.extend_from_slice(value.to_le_bytes());
        self.length += 1;
    }

    /// Appends a slice of type `T`, growing the internal buffer as needed.
    ///
    /// # Example:
    ///
    /// ```
    /// # use vortex_buffer::ScalarBufferMut;
    ///
    /// let mut builder = ScalarBufferMut::<u16>::new(10);
    /// builder.extend_from_slice(&[42, 44, 46]);
    ///
    /// assert_eq!(builder.len(), 3);
    /// ```
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        self.buffer.reserve(slice.len() * size_of::<T>());
        self.buffer
            .extend_from_slice(unsafe { std::mem::transmute(slice) });
        self.length += slice.len();
    }

    /// Freeze the `ScalarBufferMut` into a `ScalarBuffer`.
    pub fn freeze(self) -> ScalarBuffer<T> {
        ScalarBuffer::from(self.buffer.freeze())
    }
}
