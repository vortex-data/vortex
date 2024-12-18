use vortex_dtype::NativePType;
use vortex_error::vortex_panic;

use crate::{AlignedBufferMut, Alignment};

/// A mutable buffer of Vortex primitive scalars.
pub struct ScalarBufferMut<T: NativePType> {
    buffer: AlignedBufferMut,
    length: usize,
    _marker: std::marker::PhantomData<T>,
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
        self.buffer.extend_from_slice(value.as_bytes());
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
    /// builder.append_slice(&[42, 44, 46]);
    ///
    /// assert_eq!(builder.len(), 3);
    /// ```
    #[inline]
    pub fn append_slice(&mut self, slice: &[T]) {
        self.buffer.reserve(slice.len() * size_of::<T>());
        self.buffer.extend_from_slice(unsafe { std::mem::transmute(slice) };
        self.length += slice.len();
    }
}

impl<T: NativePType> Extend<T> for ScalarBufferMut<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.buffer.extend(iter.into_iter().inspect(|_| {
            self.length += 1;
        }))
    }
}
