use vortex_dtype::NativePType;
use vortex_error::vortex_panic;

use crate::AlignedBuffer;

/// A buffer of Vortex primitive scalars.
pub struct ScalarBuffer<T: NativePType> {
    /// The underlying aligned buffer.
    /// We hold an `AlignedBuffer` instead of a `Bytes` to allow defining a wider alignment than
    /// the scalar type's alignment.
    buffer: AlignedBuffer,
    _marker: std::marker::PhantomData<T>,
}

impl<T: NativePType> ScalarBuffer<T> {
    /// Returns the underlying aligned buffer.
    pub fn into_buffer(self) -> AlignedBuffer {
        self.buffer
    }
}

impl<T: NativePType> From<AlignedBuffer> for ScalarBuffer<T> {
    fn from(buffer: AlignedBuffer) -> Self {
        if !buffer.alignment().is_multiple_of(align_of::<T>()) {
            vortex_panic!("Alignment must be a multiple of the scalar type's alignment");
        }
        Self {
            buffer,
            _marker: Default::default(),
        }
    }
}
