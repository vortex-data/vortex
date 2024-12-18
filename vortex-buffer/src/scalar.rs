use vortex_dtype::NativePType;
use vortex_error::vortex_panic;

use crate::{AlignedBuffer, AlignedBufferMut};

/// A buffer of Vortex primitive scalars.
pub struct ScalarBuffer<T: NativePType> {
    /// The underlying aligned buffer.
    /// We hold an `AlignedBuffer` instead of a `Bytes` to allow defining a wider alignment than
    /// the scalar type's alignment.
    buffer: AlignedBuffer,
    length: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: NativePType> ScalarBuffer<T> {
    /// Returns a slice over the buffer of elements of type T.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        let raw_slice = self.buffer.as_slice();
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(raw_slice.as_ptr().cast(), self.length) }
    }

    /// Returns the underlying aligned buffer.
    pub fn into_inner(self) -> AlignedBuffer {
        self.buffer
    }

    /// Returns a new `ScalarBuffer<T>` copied from the provided `Vec<T>`.
    ///
    /// Due to our underlying usage of `bytes::Bytes`, we are unable to take zero-copy ownership
    /// of the provided `Vec<T>` while maintaining the ability to convert it back into a mutable
    /// buffer. We could fix this by forking `Bytes`, or in many other complex ways, but for now
    /// callers should prefer to construct `ScalarBuffer<T>` from a `ScalarBufferMut<T>`.
    pub fn copy_from_vec(vec: Vec<T>) -> Self {
        let mut buffer =
            AlignedBufferMut::with_capacity(vec.len() * size_of::<T>(), align_of::<T>().into());
        buffer.extend_from_slice(unsafe { std::mem::transmute(vec.as_slice()) });
        Self {
            buffer: buffer.freeze(),
            length: vec.len(),
            _marker: Default::default(),
        }
    }
}

impl<T: NativePType> From<AlignedBuffer> for ScalarBuffer<T> {
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
