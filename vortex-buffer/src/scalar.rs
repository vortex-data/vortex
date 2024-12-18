use bytes::Bytes;
use vortex_dtype::NativePType;
use vortex_error::vortex_panic;

use crate::AlignedBuffer;

/// A buffer of Vortex primitive scalars.
pub struct ScalarBuffer<T: NativePType> {
    /// The underlying aligned buffer. We hold an `AlignedBuffer` instead of a `Bytes` to allow
    /// defining a wider alignment than the scalar type's alignment.
    inner: AlignedBuffer,
    _marker: std::marker::PhantomData<T>,
}

impl<T: NativePType> ScalarBuffer<T> {
    pub fn into_aligned_buffer(self) -> AlignedBuffer {
        self.inner
    }

    pub fn into_bytes(self) -> Bytes {
        self.inner.into_inner()
    }
}

impl<T: NativePType> From<Vec<T>> for ScalarBuffer<T> {
    fn from(value: Vec<T>) -> Self {
        Self {
            inner: AlignedBuffer::new::<T>(Bytes::from_owner(value)),
            _marker: Default::default(),
        }
    }
}

impl<T: NativePType> From<AlignedBuffer> for ScalarBuffer<T> {
    fn from(value: AlignedBuffer) -> Self {
        if !value.alignment().is_multiple_of(align_of::<T>()) {
            vortex_panic!("Alignment must be a multiple of the scalar type's alignment");
        }
        Self {
            inner: value,
            _marker: Default::default(),
        }
    }
}
