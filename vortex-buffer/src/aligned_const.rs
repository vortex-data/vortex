use std::ops::Deref;

use vortex_error::{vortex_bail, VortexError};

use crate::{AlignedBuffer, Alignment};

/// An aligned buffer with compile-time alignment.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct ConstAlignedBuffer<const A: usize>(AlignedBuffer);

impl<const A: usize> ConstAlignedBuffer<A> {
    /// Unwrap the inner aligned buffer.
    pub fn into_inner(self) -> AlignedBuffer {
        self.0
    }
}

impl<const A: usize> TryFrom<AlignedBuffer> for ConstAlignedBuffer<A> {
    type Error = VortexError;

    fn try_from(value: AlignedBuffer) -> Result<Self, Self::Error> {
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

impl<const A: usize> AsRef<AlignedBuffer> for ConstAlignedBuffer<A> {
    fn as_ref(&self) -> &AlignedBuffer {
        &self.0
    }
}

impl<const A: usize> Deref for ConstAlignedBuffer<A> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
