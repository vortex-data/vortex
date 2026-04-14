// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`Utf8Scalar`] typed view implementation.

use std::cmp;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_buffer::BufferString;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::StringEscape;

use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A scalar value representing a UTF-8 encoded string.
///
/// This type provides a view into a UTF-8 string scalar value, which can be either
/// a valid UTF-8 string or null.
#[derive(Debug, Clone, Hash, Eq)]
pub struct Utf8Scalar<'a> {
    /// The data type of this scalar.
    dtype: &'a DType,
    /// The string value, or [`None`] if null.
    value: Option<&'a BufferString>,
}

impl Display for Utf8Scalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.value {
            None => write!(f, "null"),
            Some(v) => write!(f, "\"{}\"", StringEscape(v.as_str())),
        }
    }
}

impl PartialEq for Utf8Scalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.value == other.value
    }
}

impl PartialOrd for Utf8Scalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Utf8Scalar<'_> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<'a> Utf8Scalar<'a> {
    /// Creates a UTF-8 scalar from a data type and scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not a UTF-8 type.
    pub fn try_new(dtype: &'a DType, value: Option<&'a ScalarValue>) -> VortexResult<Self> {
        if !matches!(dtype, DType::Utf8(..)) {
            vortex_bail!("Can only construct utf8 scalar from utf8 dtype, found {dtype}")
        }

        Ok(Self {
            dtype,
            value: value.map(|value| value.as_utf8()),
        })
    }

    /// Returns the data type of this UTF-8 scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns a reference to the string value, or None if null.
    /// This avoids cloning the underlying BufferString.
    pub fn value(&self) -> Option<&'a BufferString> {
        self.value
    }

    /// Casts this scalar to the given `dtype`.
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if !matches!(dtype, DType::Utf8(..)) {
            vortex_bail!(
                "Cannot cast utf8 to {dtype}: UTF-8 scalars can only be cast to UTF-8 types with different nullability"
            )
        }
        Scalar::try_new(
            dtype.clone(),
            Some(ScalarValue::Utf8(
                self.value()
                    .cloned()
                    .vortex_expect("nullness handled in Scalar::cast"),
            )),
        )
    }

    /// Length of the scalar value or None if value is null
    pub fn len(&self) -> Option<usize> {
        self.value.as_ref().map(|v| v.len())
    }

    /// Returns whether its value is non-null and empty, otherwise `None`.
    pub fn is_empty(&self) -> Option<bool> {
        self.value.as_ref().map(|v| v.is_empty())
    }
}

/// Types that can hold a valid UTF-8 string.
pub trait StringLike: private::Sealed + Sized {
    /// Replace the last codepoint in the string with the next codepoint.
    ///
    /// This operation will attempt to reuse the original memory.
    ///
    /// If incrementing the last char fails, or if the string is empty,
    /// we return an Err with the original unmodified string.
    /// # Errors
    ///
    /// Returns `Err(self)` if the string is empty or if incrementing the last char overflows.
    fn increment(self) -> Result<Self, Self>;
}

/// Sealed trait implementation module for [`StringLike`].
mod private {
    use vortex_buffer::BufferString;

    use crate::scalar::StringLike;

    /// Prevents external implementations of [`StringLike`].
    pub trait Sealed {}

    impl Sealed for String {}

    impl StringLike for String {
        fn increment(mut self) -> Result<String, String> {
            let Some(last_char) = self.pop() else {
                return Ok(self);
            };

            if let Some(next_char) = char::from_u32(last_char as u32 + 1) {
                self.push(next_char);
                Ok(self)
            } else {
                // Return the original string
                self.push(last_char);
                Err(self)
            }
        }
    }

    impl Sealed for BufferString {}

    impl StringLike for BufferString {
        #[expect(clippy::unwrap_in_result, clippy::expect_used)]
        fn increment(self) -> Result<BufferString, BufferString> {
            if self.is_empty() {
                return Err(self);
            }

            // Chop off the last char and return it here.
            let (last_idx, last_char) = self.char_indices().last().expect("non-empty");
            if let Some(next_char) = char::from_u32(last_char as u32 + 1)
                && next_char.len_utf8() == last_char.len_utf8()
            {
                // Because the next char has the same byte width as the last char, we can overwrite
                // the memory directly.
                let mut bytes = self.into_inner().into_mut();
                next_char.encode_utf8(&mut bytes.as_mut()[last_idx..]);

                // SAFETY: we overwrite the last valid char with new valid char, so
                //  the buffer continues to hold valid UTF-8 data.
                unsafe { Ok(BufferString::new_unchecked(bytes.freeze())) }
            } else {
                Err(self)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rstest::rstest;

    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::scalar::Utf8Scalar;

    #[rstest]
    #[case("hello", "hello", true)]
    #[case("hello", "world", false)]
    #[case("", "", true)]
    #[case("abc", "ABC", false)]
    fn test_utf8_scalar_equality(#[case] str1: &str, #[case] str2: &str, #[case] expected: bool) {
        let scalar1 = Scalar::utf8(str1, Nullability::NonNullable);
        let scalar2 = Scalar::utf8(str2, Nullability::NonNullable);

        let utf8_scalar1 = scalar1.as_utf8();
        let utf8_scalar2 = scalar2.as_utf8();

        assert_eq!(utf8_scalar1 == utf8_scalar2, expected);
    }

    #[rstest]
    #[case("apple", "banana", Ordering::Less)]
    #[case("banana", "apple", Ordering::Greater)]
    #[case("apple", "apple", Ordering::Equal)]
    #[case("", "a", Ordering::Less)]
    #[case("z", "aa", Ordering::Greater)]
    fn test_utf8_scalar_ordering(
        #[case] str1: &str,
        #[case] str2: &str,
        #[case] expected: Ordering,
    ) {
        let scalar1 = Scalar::utf8(str1, Nullability::NonNullable);
        let scalar2 = Scalar::utf8(str2, Nullability::NonNullable);

        let utf8_scalar1 = scalar1.as_utf8();
        let utf8_scalar2 = scalar2.as_utf8();

        assert_eq!(utf8_scalar1.partial_cmp(&utf8_scalar2), Some(expected));
    }

    #[test]
    fn test_utf8_null_value() {
        let null_utf8 = Scalar::null(crate::dtype::DType::Utf8(Nullability::Nullable));
        let scalar = null_utf8.as_utf8();

        assert!(scalar.value().is_none());
        assert!(scalar.value().is_none());
        assert!(scalar.len().is_none());
        assert!(scalar.is_empty().is_none());
    }

    #[test]
    fn test_utf8_len_and_empty() {
        let empty = Scalar::utf8("", Nullability::NonNullable);
        let non_empty = Scalar::utf8("hello", Nullability::NonNullable);

        let empty_scalar = empty.as_utf8();
        assert_eq!(empty_scalar.len(), Some(0));
        assert_eq!(empty_scalar.is_empty(), Some(true));

        let non_empty_scalar = non_empty.as_utf8();
        assert_eq!(non_empty_scalar.len(), Some(5));
        assert_eq!(non_empty_scalar.is_empty(), Some(false));
    }

    #[test]
    fn test_utf8_value_ref() {
        let data = "test string";
        let utf8 = Scalar::utf8(data, Nullability::NonNullable);
        let scalar = utf8.as_utf8();

        // value_ref should not clone
        let value_ref = scalar.value().unwrap();
        assert_eq!(value_ref.as_str(), data);

        // value should clone
        let value = scalar.value().unwrap();
        assert_eq!(value.as_str(), data);
    }

    #[test]
    fn test_utf8_cast_to_utf8() {
        use crate::dtype::DType;
        use crate::dtype::Nullability;

        let utf8 = Scalar::utf8("test", Nullability::NonNullable);
        let scalar = utf8.as_utf8();

        // Cast to nullable utf8
        let result = scalar.cast(&DType::Utf8(Nullability::Nullable)).unwrap();
        assert_eq!(result.dtype(), &DType::Utf8(Nullability::Nullable));

        let casted = result.as_utf8();
        assert_eq!(casted.value().unwrap().as_str(), "test");
    }

    #[test]
    fn test_utf8_cast_to_non_utf8_fails() {
        use crate::dtype::DType;
        use crate::dtype::Nullability;
        use crate::dtype::PType;

        let utf8 = Scalar::utf8("test", Nullability::NonNullable);
        let scalar = utf8.as_utf8();

        let result = scalar.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_try_new_non_utf8_dtype() {
        use crate::dtype::DType;
        use crate::dtype::Nullability;
        use crate::dtype::PType;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let value = crate::scalar::ScalarValue::Primitive(crate::scalar::PValue::I32(42));

        let result = Utf8Scalar::try_new(&dtype, Some(&value));
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_non_utf8_scalar() {
        use crate::dtype::Nullability;

        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        assert!(scalar.as_utf8_opt().is_none());
    }

    #[test]
    fn test_from_str() {
        let data = "hello world";
        let scalar: Scalar = data.into();

        assert_eq!(
            scalar.dtype(),
            &crate::dtype::DType::Utf8(Nullability::NonNullable)
        );
        let utf8 = scalar.as_utf8();
        assert_eq!(utf8.value().unwrap().as_str(), data);
    }

    #[test]
    fn test_from_string() {
        let data = String::from("hello world");
        let scalar: Scalar = data.into();

        assert_eq!(
            scalar.dtype(),
            &crate::dtype::DType::Utf8(Nullability::NonNullable)
        );
        let utf8 = scalar.as_utf8();
        assert_eq!(utf8.value().unwrap().as_str(), "hello world");
    }

    #[test]
    fn test_from_buffer_string() {
        use vortex_buffer::BufferString;

        let data = BufferString::from("test");
        let scalar: Scalar = data.into();

        assert_eq!(
            scalar.dtype(),
            &crate::dtype::DType::Utf8(Nullability::NonNullable)
        );
        let utf8 = scalar.as_utf8();
        assert_eq!(utf8.value().unwrap().as_str(), "test");
    }

    #[test]
    fn test_try_from_scalar_to_string() {
        let data = "test string";
        let scalar = Scalar::utf8(data, Nullability::NonNullable);

        // Try from &Scalar to String
        let string: String = (&scalar).try_into().unwrap();
        assert_eq!(string, data);
    }

    #[test]
    fn test_try_from_scalar_to_buffer_string() {
        use vortex_buffer::BufferString;

        let data = "test data";
        let scalar = Scalar::utf8(data, Nullability::NonNullable);

        // Try from &Scalar
        let buffer: BufferString = (&scalar).try_into().unwrap();
        assert_eq!(buffer.as_str(), data);

        // Try from Scalar (owned)
        let scalar2 = Scalar::utf8(data, Nullability::NonNullable);
        let buffer2: BufferString = scalar2.try_into().unwrap();
        assert_eq!(buffer2.as_str(), data);
    }

    #[test]
    fn test_try_from_scalar_to_option_buffer_string() {
        use vortex_buffer::BufferString;

        // Non-null case
        let data = "test";
        let scalar = Scalar::utf8(data, Nullability::Nullable);
        let buffer: Option<BufferString> = (&scalar).try_into().unwrap();
        assert_eq!(buffer.unwrap().as_str(), data);

        // Null case
        let null_scalar = Scalar::null(crate::dtype::DType::Utf8(Nullability::Nullable));
        let null_buffer: Option<BufferString> = (&null_scalar).try_into().unwrap();
        assert!(null_buffer.is_none());
    }

    #[test]
    fn test_try_from_non_utf8_to_buffer_string() {
        use vortex_buffer::BufferString;

        use crate::dtype::Nullability;

        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);

        let result: Result<BufferString, _> = (&scalar).try_into();
        assert!(result.is_err());

        let result2: Result<Option<BufferString>, _> = (&scalar).try_into();
        assert!(result2.is_err());
    }

    #[test]
    fn test_scalar_value_from_str() {
        let data = "test";
        let value: crate::scalar::ScalarValue = data.into();

        let scalar = Scalar::new(
            crate::dtype::DType::Utf8(Nullability::NonNullable),
            Some(value),
        );
        let utf8 = scalar.as_utf8();
        assert_eq!(utf8.value().unwrap().as_str(), data);
    }

    #[test]
    fn test_scalar_value_from_string() {
        let data = String::from("test");
        let value: crate::scalar::ScalarValue = data.clone().into();

        let scalar = Scalar::new(
            crate::dtype::DType::Utf8(Nullability::NonNullable),
            Some(value),
        );
        let utf8 = scalar.as_utf8();
        assert_eq!(utf8.value().unwrap().as_str(), &data);
    }

    #[test]
    fn test_scalar_value_from_buffer_string() {
        use vortex_buffer::BufferString;

        let data = BufferString::from("test");
        let value: crate::scalar::ScalarValue = data.into();

        let scalar = Scalar::new(
            crate::dtype::DType::Utf8(Nullability::NonNullable),
            Some(value),
        );
        let utf8 = scalar.as_utf8();
        assert_eq!(utf8.value().unwrap().as_str(), "test");
    }

    #[test]
    fn test_utf8_with_emoji() {
        let emoji_str = "Hello 👋 World 🌍!";
        let scalar = Scalar::utf8(emoji_str, Nullability::NonNullable);
        let utf8_scalar = scalar.as_utf8();

        assert_eq!(utf8_scalar.value().unwrap().as_str(), emoji_str);
        assert!(utf8_scalar.len().unwrap() > emoji_str.chars().count()); // Byte length > char count
    }

    #[test]
    fn test_partial_ord_null() {
        let null_scalar = Scalar::null(crate::dtype::DType::Utf8(Nullability::Nullable));
        let non_null_scalar = Scalar::utf8("test", Nullability::Nullable);

        let null = null_scalar.as_utf8();
        let non_null = non_null_scalar.as_utf8();

        // Null < Some("test")
        assert!(null < non_null);
        assert!(non_null > null);
    }
}
