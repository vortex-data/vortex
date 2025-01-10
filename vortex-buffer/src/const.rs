use std::ops::Deref;

use vortex_error::{vortex_bail, VortexError};

use crate::{Alignment, Buffer};

/// A buffer of items of `T` with a compile-time alignment.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct ConstBuffer<T, const A: usize>(Buffer<T>);

impl<T, const A: usize> ConstBuffer<T, A> {
    /// Returns the alignment of the buffer.
    pub const fn alignment() -> Alignment {
        Alignment::new(A)
    }

    /// Align the given buffer (possibly with a copy) and return a new `ConstBuffer`.
    pub fn align_from<B: Into<Buffer<T>>>(buf: B) -> Self {
        Self(buf.into().aligned(Self::alignment()))
    }

    /// Unwrap the inner buffer.
    pub fn into_inner(self) -> Buffer<T> {
        self.0
    }
}

impl<T, const A: usize> TryFrom<Buffer<T>> for ConstBuffer<T, A> {
    type Error = VortexError;

    fn try_from(value: Buffer<T>) -> Result<Self, Self::Error> {
        if !value.alignment().is_aligned_to(Alignment::new(A)) {
            vortex_bail!(
                "Cannot convert buffer with alignment {} to buffer with alignment {}",
                value.alignment(),
                A
            );
        }
        Ok(Self(value))
    }
}

impl<T, const A: usize> AsRef<Buffer<T>> for ConstBuffer<T, A> {
    fn as_ref(&self) -> &Buffer<T> {
        &self.0
    }
}

impl<T, const A: usize> Deref for ConstBuffer<T, A> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}
