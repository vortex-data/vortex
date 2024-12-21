use std::ops::Deref;

use vortex_error::{vortex_bail, VortexError};

use crate::{Alignment, Buffer};

/// A buffer of items of `T` with a compile-time alignment.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct ConstBuffer<T, const A: usize> {
    buffer: Buffer<T>,
    alignment: Alignment,
}

impl<T, const A: usize> ConstBuffer<T, A> {
    /// Unwrap the inner buffer.
    pub fn into_inner(self) -> Buffer<T> {
        self.buffer
    }

    /// Returns the alignment of the buffer.
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }
}

impl<T, const A: usize> TryFrom<Buffer<T>> for ConstBuffer<T, A> {
    type Error = VortexError;

    fn try_from(buffer: Buffer<T>) -> Result<Self, Self::Error> {
        let alignment = Alignment::new(A);
        if !buffer.alignment().is_aligned_to(alignment) {
            vortex_bail!(
                "Cannot convert buffer with alignment {} to buffer with alignment {}",
                buffer.alignment(),
                A
            );
        }
        Ok(Self { buffer, alignment })
    }
}

impl<T, const A: usize> AsRef<Buffer<T>> for ConstBuffer<T, A> {
    fn as_ref(&self) -> &Buffer<T> {
        &self.buffer
    }
}

impl<T, const A: usize> Deref for ConstBuffer<T, A> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.buffer.as_slice()
    }
}
