// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bound the min and max statistics of scalars to a byte budget, so that stored statistics stay
//! small.

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar::StringLike;

/// A trait for truncating [`Scalar`]s to a given length in bytes.
trait ScalarTruncation: Send + Sized {
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

/// Whether the min and max of `dtype` can be bounded within `max_bytes` per value.
///
/// UTF-8 and binary values are truncated to fit. A fixed-size list, whose width grows with its
/// list size, cannot be shortened without changing its dtype, so one wider than `max_bytes` has no
/// bound. Extension types follow their storage, and other values are kept as they are.
pub fn can_bound_extrema(dtype: &DType, max_bytes: usize) -> bool {
    match dtype {
        DType::FixedSizeList(..) => dtype.element_size().is_some_and(|bytes| bytes <= max_bytes),
        DType::Extension(ext_dtype) => can_bound_extrema(ext_dtype.storage_dtype(), max_bytes),
        _ => true,
    }
}

/// A value at most `max_bytes` long that is less than or equal to `value`, and whether it was
/// truncated.
///
/// Returns `None` for a null value or one whose dtype fails [`can_bound_extrema`].
pub fn lower_bound(value: Scalar, max_bytes: usize) -> Option<(Scalar, bool)> {
    let nullability = value.dtype().nullability();
    match value.dtype() {
        DType::Utf8(_) => truncated_lower_bound(
            BufferString::from_scalar(value).vortex_expect("utf8 scalar"),
            max_bytes,
            nullability,
        ),
        DType::Binary(_) => truncated_lower_bound(
            ByteBuffer::from_scalar(value).vortex_expect("binary scalar"),
            max_bytes,
            nullability,
        ),
        _ => untruncated_bound(value, max_bytes),
    }
}

/// A value at most `max_bytes` long that is greater than or equal to `value`, and whether it was
/// truncated.
///
/// Returns `None` for a null value, one whose dtype fails [`can_bound_extrema`], or one with no
/// greater value within `max_bytes`, such as a string of `0xFF` bytes.
pub fn upper_bound(value: Scalar, max_bytes: usize) -> Option<(Scalar, bool)> {
    let nullability = value.dtype().nullability();
    match value.dtype() {
        DType::Utf8(_) => truncated_upper_bound(
            BufferString::from_scalar(value).vortex_expect("utf8 scalar"),
            max_bytes,
            nullability,
        ),
        DType::Binary(_) => truncated_upper_bound(
            ByteBuffer::from_scalar(value).vortex_expect("binary scalar"),
            max_bytes,
            nullability,
        ),
        _ => untruncated_bound(value, max_bytes),
    }
}

fn untruncated_bound(value: Scalar, max_bytes: usize) -> Option<(Scalar, bool)> {
    (!value.is_null() && can_bound_extrema(value.dtype(), max_bytes)).then_some((value, false))
}

fn truncated_lower_bound(
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

fn truncated_upper_bound(
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
    use rstest::rstest;
    use vortex_buffer::BufferString;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtId;
    use crate::dtype::extension::ForeignExtDType;
    use crate::scalar::Scalar;
    use crate::scalar::truncation::ScalarTruncation;
    use crate::scalar::truncation::can_bound_extrema;
    use crate::scalar::truncation::lower_bound;
    use crate::scalar::truncation::upper_bound;

    fn byte_list<const N: usize>(bytes: [u8; N]) -> Scalar {
        let element_dtype = DType::Primitive(PType::U8, NonNullable);
        let elements = bytes
            .into_iter()
            .map(|byte| Scalar::primitive(byte, NonNullable))
            .collect();
        Scalar::fixed_size_list(element_dtype, elements, NonNullable)
    }

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

    #[rstest]
    #[case::utf8(DType::Utf8(Nullable))]
    #[case::binary(DType::Binary(Nullable))]
    #[case::primitive(DType::Primitive(PType::I32, Nullable))]
    fn null_values_have_no_bound(#[case] dtype: DType) {
        assert_eq!(lower_bound(Scalar::null(dtype.clone()), 10), None);
        assert_eq!(upper_bound(Scalar::null(dtype), 10), None);
    }

    #[test]
    fn utf8_scalar_bounds_are_truncated() {
        let value = Scalar::utf8("char🪩", NonNullable);
        assert_eq!(
            lower_bound(value.clone(), 5),
            Some((Scalar::utf8("char", NonNullable), true))
        );
        assert_eq!(
            upper_bound(value, 5),
            Some((Scalar::utf8("chas", NonNullable), true))
        );
    }

    /// A fixed-size list cannot be truncated, so it is its own bound or has none.
    #[rstest]
    #[case::fits(4, true)]
    #[case::too_wide(3, false)]
    fn fixed_size_list_is_bounded_whole_or_not_at_all(
        #[case] max_bytes: usize,
        #[case] bounded: bool,
    ) {
        let value = byte_list([1, 2, 3, 4]);
        let expected = bounded.then(|| (value.clone(), false));
        assert_eq!(can_bound_extrema(value.dtype(), max_bytes), bounded);
        assert_eq!(lower_bound(value.clone(), max_bytes), expected);
        assert_eq!(upper_bound(value, max_bytes), expected);
    }

    /// Extension types follow their storage, so a UUID-like type over 16 bytes needs 16.
    #[rstest]
    #[case::fits(16, true)]
    #[case::too_wide(15, false)]
    fn extension_is_bounded_by_its_storage(
        #[case] max_bytes: usize,
        #[case] bounded: bool,
    ) -> VortexResult<()> {
        let storage = byte_list([7; 16]);
        #[expect(clippy::disallowed_methods, reason = "test-only id")]
        let id = ExtId::new("test.bytes16");
        let ext_dtype = ForeignExtDType::from_parts(id, Vec::new(), storage.dtype().clone())?;
        let value = Scalar::extension_ref(ext_dtype, storage);
        let expected = bounded.then(|| (value.clone(), false));
        assert_eq!(can_bound_extrema(value.dtype(), max_bytes), bounded);
        assert_eq!(lower_bound(value.clone(), max_bytes), expected);
        assert_eq!(upper_bound(value, max_bytes), expected);
        Ok(())
    }
}
