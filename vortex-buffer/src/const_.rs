// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use vortex_error::VortexError;
use vortex_error::vortex_bail;

use crate::Alignment;
use crate::Buffer;

/// A buffer of items of `T` with a compile-time alignment.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Hash)]
pub struct ConstBuffer<T, const A: usize>(Buffer<T>);

impl<T, const A: usize> ConstBuffer<T, A> {
    /// Returns the alignment of the buffer.
    pub const fn alignment() -> Alignment {
        Alignment::new(A)
    }

    /// Align the given buffer (possibly with a copy) and return a new `ConstBuffer`.
    pub fn align_from<B: Into<Buffer<T>>>(buf: B) -> Self
    where
        T: Copy,
    {
        Self(buf.into().aligned(Self::alignment()))
    }

    /// Create a new [`ConstBuffer`] with a copy from the provided slice.
    pub fn copy_from<B: AsRef<[T]>>(buf: B) -> Self
    where
        T: Copy,
    {
        Self(Buffer::<T>::copy_from_aligned(buf, Self::alignment()))
    }

    /// Returns a slice over the buffer of elements of type T.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }

    /// Unwrap the inner buffer.
    pub fn inner(&self) -> &Buffer<T> {
        &self.0
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

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::Alignment;
    use crate::ByteBuffer;
    use crate::ConstBuffer;

    #[test]
    fn try_from_aligned_succeeds_and_misaligned_errors() -> VortexResult<()> {
        let aligned = ByteBuffer::copy_from_aligned([1u8, 2, 3, 4], Alignment::new(16));
        let ptr = aligned.as_ptr();
        let const_buf = ConstBuffer::<u8, 16>::try_from(aligned)?;
        assert_eq!(const_buf.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(const_buf.inner().as_ptr(), ptr);
        assert_eq!(const_buf.into_inner().alignment(), Alignment::new(16));

        let unaligned = ByteBuffer::copy_from_aligned([1u8, 2, 3, 4], Alignment::new(1));
        let err = ConstBuffer::<u8, 16>::try_from(unaligned).unwrap_err();
        assert!(
            err.to_string()
                .contains("Cannot convert buffer with alignment 1 to buffer with alignment 16"),
            "{err}"
        );
        Ok(())
    }

    #[test]
    fn align_from_realigns_misaligned_slice() {
        let misaligned =
            ByteBuffer::copy_from_aligned([0u8, 1, 2, 3, 4, 5, 6, 7], Alignment::new(8))
                .slice_unaligned(1..5);
        let const_buf = ConstBuffer::<u8, 8>::align_from(misaligned);
        assert_eq!(const_buf.as_slice(), &[1, 2, 3, 4]);
        assert!(const_buf.inner().is_aligned(Alignment::new(8)));
        assert_eq!(const_buf.inner().alignment(), Alignment::new(8));
    }
}
