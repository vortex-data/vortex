use bytes::BytesMut;
use vortex_dtype::NativePType;
use vortex_error::vortex_panic;

use crate::Alignment;

/// A mutable buffer of Vortex primitive scalars.
pub struct ScalarBufferMut<T: NativePType> {
    bytes: BytesMut,
    alignment: Alignment,
    _marker: std::marker::PhantomData<T>,
}

impl<T: NativePType> ScalarBufferMut<T> {
    pub fn new() -> Self {
        Self::with_capacity_aligned(0, align_of::<T>().into())
    }

    pub fn new_aligned(alignment: Alignment) -> Self {
        Self::with_capacity_aligned(0, alignment)
    }

    pub fn with_capacity_aligned(capacity: usize, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must align to the scalar type's alignment {}",
                alignment,
                align_of::<T>()
            );
        }

        /// We allocate extra so we can shift the starting position to the next aligned address.
        let bytes = BytesMut::with_capacity((capacity * size_of::<T>()) + *alignment);

        Self {
            bytes: BytesMut::with_capacity(capacity * size_of::<T>()),
            alignment,
            _marker: Default::default(),
        }
    }
}
