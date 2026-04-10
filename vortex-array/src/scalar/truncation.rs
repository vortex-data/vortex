// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Produce lower on upper bounds of scalars via truncation.

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar::StringLike;

/// A trait for truncating [`Scalar`]s to a given length in bytes.
#[expect(clippy::len_without_is_empty)]
pub trait ScalarTruncation: Send + Sized {
    /// Unwrap a Scalar into a ScalarTruncation object
    ///
    /// # Errors
    /// If the scalar doesn't match the truncations dtype.
    fn from_scalar(value: Scalar) -> VortexResult<Option<Self>>;

    /// The length of the value in bytes.
    fn len(&self) -> usize;

    /// Convert the value into a [`Scalar`] with the given nullability.
    fn into_scalar(self, nullability: Nullability) -> Scalar;

    /// Constructs the next [`Scalar`] at most `max_length` bytes that's lexicographically greater
    /// than this.
    ///
    /// Returns `None` if the value is null or if constructing a greater value would overflow.
    fn upper_bound(self, max_length: usize) -> Option<Self>;

    /// Construct a [`ByteBuffer`] at most `max_length` in size that's less than or equal to
    /// ourselves.
    fn lower_bound(self, max_length: usize) -> Self;
}

impl ScalarTruncation for ByteBuffer {
    fn from_scalar(value: Scalar) -> VortexResult<Option<Self>> {
        vortex_ensure!(
            value.dtype().is_binary(),
            "Expected binary scalar, got {}",
            value.dtype()
        );
        Ok(value.into_value().map(|b| b.into_binary()))
    }

    fn len(&self) -> usize {
        ByteBuffer::len(self)
    }

    fn into_scalar(self, nullability: Nullability) -> Scalar {
        Scalar::binary(self, nullability)
    }

    fn upper_bound(self, max_length: usize) -> Option<Self> {
        let sliced = self.slice(0..max_length);
        let mut sliced_mut = sliced.into_mut();
        for b in sliced_mut.iter_mut().rev() {
            let (incr, overflow) = b.overflowing_add(1);
            *b = incr;
            if !overflow {
                return Some(sliced_mut.freeze());
            }
        }
        None
    }

    fn lower_bound(self, max_length: usize) -> Self {
        self.slice(0..max_length)
    }
}

impl ScalarTruncation for BufferString {
    fn from_scalar(value: Scalar) -> VortexResult<Option<Self>> {
        vortex_ensure!(
            value.dtype().is_utf8(),
            "Expected utf8 scalar, got {}",
            value.dtype()
        );
        Ok(value.into_value().map(|b| b.into_utf8()))
    }

    fn len(&self) -> usize {
        self.inner().len()
    }

    fn into_scalar(self, nullability: Nullability) -> Scalar {
        Scalar::utf8(self, nullability)
    }

    /// Constructs the next [`BufferString`] at most `max_length` bytes that's lexicographically greater
    /// than this.
    ///
    /// Returns `None` if the value is null or if constructing a greater value would overflow.
    fn upper_bound(self, max_length: usize) -> Option<Self> {
        let utf8_split_pos = (max_length.saturating_sub(3)..=max_length)
            .rfind(|p| self.is_char_boundary(*p))
            .vortex_expect("Failed to find utf8 character boundary");

        // SAFETY: we slice to a char boundary so the sliced range contains valid UTF-8.
        let sliced =
            unsafe { BufferString::new_unchecked(self.into_inner().slice(..utf8_split_pos)) };
        sliced.increment().ok()
    }

    /// Construct a [`BufferString`] at most `max_length` in size that's less than or equal to
    /// ourselves.
    fn lower_bound(self, max_length: usize) -> Self {
        // UTF-8 characters are at most 4 bytes. Since we know that `BufferString` is
        // valid UTF-8, we must have a valid character boundary.
        let utf8_split_pos = (max_length.saturating_sub(3)..=max_length)
            .rfind(|p| self.is_char_boundary(*p))
            .vortex_expect("Failed to find utf8 character boundary");

        unsafe { BufferString::new_unchecked(self.into_inner().slice(..utf8_split_pos)) }
    }
}

/// Truncate the value to be less than max_length in bytes and be lexicographically smaller than the value itself
pub fn lower_bound(
    value: Option<impl ScalarTruncation>,
    max_length: usize,
    nullability: Nullability,
) -> Option<(Scalar, bool)> {
    let value = value?;
    if value.len() > max_length {
        Some((value.lower_bound(max_length).into_scalar(nullability), true))
    } else {
        Some((value.into_scalar(nullability), false))
    }
}

/// Truncate the value to be less than max_length in bytes and be lexicographically greater than the value itself
pub fn upper_bound(
    value: Option<impl ScalarTruncation>,
    max_length: usize,
    nullability: Nullability,
) -> Option<(Scalar, bool)> {
    let value = value?;
    if value.len() > max_length {
        value
            .upper_bound(max_length)
            .map(|v| (v.into_scalar(nullability), true))
    } else {
        Some((value.into_scalar(nullability), false))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferString;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;

    use crate::dtype::Nullability;
    use crate::scalar::truncation::ScalarTruncation;
    use crate::scalar::truncation::lower_bound;
    use crate::scalar::truncation::upper_bound;

    #[test]
    fn binary_lower_bound() {
        let binary = buffer![0u8, 5, 47, 33, 129];
        let expected = buffer![0u8, 5];
        assert_eq!(binary.lower_bound(2), expected,);
    }

    #[test]
    fn binary_upper_bound() {
        let binary = buffer![0u8, 5, 255, 234, 23];
        let expected = buffer![0u8, 6, 0];
        assert_eq!(binary.upper_bound(3).unwrap(), expected,);
    }

    #[test]
    fn binary_upper_bound_overflow() {
        let binary = buffer![255u8, 255, 255];
        assert!(binary.upper_bound(2).is_none());
    }

    #[test]
    fn binary_upper_bound_null() {
        assert!(upper_bound(Option::<ByteBuffer>::None, 10, Nullability::Nullable).is_none());
    }

    #[test]
    fn binary_lower_bound_null() {
        assert!(lower_bound(Option::<ByteBuffer>::None, 10, Nullability::Nullable).is_none());
    }

    #[test]
    fn utf8_lower_bound() {
        let utf8 = BufferString::from("snowman⛄️snowman");
        let expected = BufferString::from("snowman");
        assert_eq!(utf8.lower_bound(9), expected);
    }

    #[test]
    fn utf8_upper_bound() {
        let utf8 = BufferString::from("char🪩");
        let expected = BufferString::from("chas");
        assert_eq!(utf8.upper_bound(5).unwrap(), expected);
    }

    #[test]
    fn utf8_upper_bound_overflow() {
        let utf8 = BufferString::from("🂑🂒🂓");
        assert!(utf8.upper_bound(2).is_none());
    }

    #[test]
    fn utf8_upper_bound_null() {
        assert!(upper_bound(Option::<BufferString>::None, 10, Nullability::Nullable).is_none());
    }

    #[test]
    fn utf8_lower_bound_null() {
        assert!(lower_bound(Option::<BufferString>::None, 10, Nullability::Nullable).is_none());
    }
}
