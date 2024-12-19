use std::ops::Deref;

use vortex_error::{vortex_bail, VortexError};

use crate::{Alignment, ScalarBuffer};

/// An aligned buffer with compile-time alignment.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct ConstBuffer<T, const A: usize>(ScalarBuffer<T>);

impl<T, const A: usize> ConstBuffer<T, A> {
    /// Unwrap the inner buffer.
    pub fn into_inner(self) -> ScalarBuffer<T> {
        self.0
    }
}

impl<T, const A: usize> TryFrom<ScalarBuffer<T>> for ConstBuffer<T, A> {
    type Error = VortexError;

    fn try_from(value: ScalarBuffer<T>) -> Result<Self, Self::Error> {
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

impl<T, const A: usize> AsRef<ScalarBuffer<T>> for ConstBuffer<T, A> {
    fn as_ref(&self) -> &ScalarBuffer<T> {
        &self.0
    }
}

impl<T, const A: usize> Deref for ConstBuffer<T, A> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}
