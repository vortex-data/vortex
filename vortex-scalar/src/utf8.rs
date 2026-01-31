// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferString;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::Nullability::NonNullable;

use crate::Scalar;
use crate::ScalarValue;

/// Types that can hold a valid UTF-8 string.
pub trait StringLike: private::Sealed + Sized {
    /// Replace the last codepoint in the string with the next codepoint.
    ///
    /// This operation will attempt to reuse the original memory.
    ///
    /// If incrementing the last char fails, or if the string is empty,
    /// we return an Err with the original unmodified string.
    fn increment(self) -> Result<Self, Self>;
}

mod private {
    use vortex_buffer::BufferString;

    use crate::StringLike;

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
        #[allow(clippy::unwrap_in_result, clippy::expect_used)]
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

/// A scalar value representing a UTF-8 encoded string.
///
/// This type provides a view into a UTF-8 string scalar value, which can be either
/// a valid UTF-8 string or null.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Utf8Scalar<'a> {
    pub(super) nullability: Nullability,
    pub(super) value: Option<&'a BufferString>,
}

impl<'a> Utf8Scalar<'a> {
    /// Returns the string value, or None if null.
    pub fn value(&self) -> Option<&BufferString> {
        self.value
    }
    //
    // /// Constructs the next scalar at most `max_length` bytes that's lexicographically greater than
    // /// this.
    // ///
    // /// Returns None if constructing a greater value would overflow.
    // pub fn upper_bound(self, max_length: usize) -> Option<Self> {
    //     if let Some(value) = self.value {
    //         if value.len() > max_length {
    //             let utf8_split_pos = (max_length.saturating_sub(3)..=max_length)
    //                 .rfind(|p| value.is_char_boundary(*p))
    //                 .vortex_expect("Failed to find utf8 character boundary");
    //
    //             let sliced = value.inner().slice(..utf8_split_pos);
    //             drop(value);
    //
    //             // SAFETY: we slice to a char boundary so the sliced range contains valid UTF-8.
    //             let sliced_buf = unsafe { BufferString::new_unchecked(sliced) };
    //             let incremented = sliced_buf.increment().ok()?;
    //             Some(Self {
    //                 dtype: self.dtype,
    //                 value: Some(Arc::new(incremented)),
    //             })
    //         } else {
    //             Some(Self {
    //                 dtype: self.dtype,
    //                 value: Some(value),
    //             })
    //         }
    //     } else {
    //         Some(self)
    //     }
    // }
    //
    // /// Construct a value at most `max_length` in size that's less than ourselves.
    // pub fn lower_bound(self, max_length: usize) -> Self {
    //     if let Some(value) = self.value {
    //         if value.len() > max_length {
    //             // UTF8 characters are at most 4 bytes, since we know that BufferString is UTF8 we must have a valid character boundary
    //             let utf8_split_pos = (max_length.saturating_sub(3)..=max_length)
    //                 .rfind(|p| value.is_char_boundary(*p))
    //                 .vortex_expect("Failed to find utf8 character boundary");
    //
    //             Self {
    //                 dtype: self.dtype,
    //                 value: Some(Arc::new(unsafe {
    //                     BufferString::new_unchecked(value.inner().slice(0..utf8_split_pos))
    //                 })),
    //             }
    //         } else {
    //             Self {
    //                 dtype: self.dtype,
    //                 value: Some(value),
    //             }
    //         }
    //     } else {
    //         self
    //     }
    // }
    //
    // pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
    //     if !matches!(dtype, DType::Utf8(..)) {
    //         vortex_bail!(
    //             "Cannot cast utf8 to {dtype}: UTF-8 scalars can only be cast to UTF-8 types with different nullability"
    //         )
    //     }
    //     Ok(Scalar::new(
    //         dtype.clone(),
    //         ScalarValue(InnerScalarValue::BufferString(
    //             self.value
    //                 .as_ref()
    //                 .vortex_expect("nullness handled in Scalar::cast")
    //                 .clone(),
    //         )),
    //     ))
    // }

    /// Length of the scalar value or None if value is null
    pub fn len(&self) -> Option<usize> {
        self.value.map(|v| v.len())
    }

    /// Returns whether its value is non-null and empty, otherwise `None`.
    pub fn is_empty(&self) -> Option<bool> {
        self.value.map(|v| v.is_empty())
    }
}

impl Scalar {
    /// Creates a new UTF-8 scalar from a string-like value.
    ///
    /// # Panics
    ///
    /// Panics if the input cannot be converted to a valid UTF-8 string.
    pub fn utf8<B>(str: B, nullability: Nullability) -> Self
    where
        B: Into<BufferString>,
    {
        unsafe { Self::new_unchecked(DType::Utf8(nullability), ScalarValue::Utf8(str.into())) }
    }

    /// Tries to create a new UTF-8 scalar from a string-like value.
    ///
    /// # Errors
    ///
    /// Returns an error if the input cannot be converted to a valid UTF-8 string.
    pub fn try_utf8<B>(
        str: B,
        nullability: Nullability,
    ) -> Result<Self, <B as TryInto<BufferString>>::Error>
    where
        B: TryInto<BufferString>,
    {
        Ok(unsafe {
            Self::new_unchecked(DType::Utf8(nullability), ScalarValue::Utf8(str.try_into()?))
        })
    }
}

impl From<&str> for Scalar {
    fn from(value: &str) -> Self {
        Self::utf8(value, NonNullable)
    }
}

impl From<String> for Scalar {
    fn from(value: String) -> Self {
        Self::utf8(value, NonNullable)
    }
}

impl From<BufferString> for Scalar {
    fn from(value: BufferString) -> Self {
        Self::utf8(value, NonNullable)
    }
}

impl From<&str> for ScalarValue {
    fn from(value: &str) -> Self {
        ScalarValue::Utf8(BufferString::from(value))
    }
}

impl From<String> for ScalarValue {
    fn from(value: String) -> Self {
        ScalarValue::Utf8(BufferString::from(value))
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rstest::rstest;
    use vortex_dtype::Nullability;
    use vortex_error::VortexExpect;

    use crate::Scalar;
    use crate::Utf8Scalar;

    #[test]
    fn lower_bound() {
        let utf8 = Scalar::utf8("snowman⛄️snowman", Nullability::NonNullable);
        let expected = Scalar::utf8("snowman", Nullability::NonNullable);
        assert_eq!(
            Utf8Scalar::try_from(&utf8)
                .vortex_expect("utf8 scalar conversion should succeed")
                .lower_bound(9),
            Utf8Scalar::try_from(&expected).vortex_expect("utf8 scalar conversion should succeed")
        );
    }

    #[test]
    fn upper_bound() {
        let utf8 = Scalar::utf8("char🪩", Nullability::NonNullable);
        let expected = Scalar::utf8("chas", Nullability::NonNullable);
        assert_eq!(
            Utf8Scalar::try_from(&utf8)
                .vortex_expect("utf8 scalar conversion should succeed")
                .upper_bound(5)
                .vortex_expect("must have upper bound"),
            Utf8Scalar::try_from(&expected).vortex_expect("utf8 scalar conversion should succeed")
        );
    }

    #[test]
    fn upper_bound_overflow() {
        let utf8 = Scalar::utf8("🂑🂒🂓", Nullability::NonNullable);

        assert!(
            Utf8Scalar::try_from(&utf8)
                .vortex_expect("utf8 scalar conversion should succeed")
                .upper_bound(2)
                .is_none()
        );
    }

    #[rstest]
    #[case("hello", "hello", true)]
    #[case("hello", "world", false)]
    #[case("", "", true)]
    #[case("abc", "ABC", false)]
    fn test_utf8_scalar_equality(#[case] str1: &str, #[case] str2: &str, #[case] expected: bool) {
        let scalar1 = Scalar::utf8(str1, Nullability::NonNullable);
        let scalar2 = Scalar::utf8(str2, Nullability::NonNullable);

        let utf8_scalar1 = Utf8Scalar::try_from(&scalar1).unwrap();
        let utf8_scalar2 = Utf8Scalar::try_from(&scalar2).unwrap();

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

        let utf8_scalar1 = Utf8Scalar::try_from(&scalar1).unwrap();
        let utf8_scalar2 = Utf8Scalar::try_from(&scalar2).unwrap();

        assert_eq!(utf8_scalar1.partial_cmp(&utf8_scalar2), Some(expected));
    }

    #[test]
    fn test_utf8_null_value() {
        let null_utf8 = Scalar::null(vortex_dtype::DType::Utf8(Nullability::Nullable));
        let scalar = Utf8Scalar::try_from(&null_utf8).unwrap();

        assert!(scalar.value().is_none());
        assert!(scalar.value_ref().is_none());
        assert!(scalar.len().is_none());
        assert!(scalar.is_empty().is_none());
    }

    #[test]
    fn test_utf8_len_and_empty() {
        let empty = Scalar::utf8("", Nullability::NonNullable);
        let non_empty = Scalar::utf8("hello", Nullability::NonNullable);

        let empty_scalar = Utf8Scalar::try_from(&empty).unwrap();
        assert_eq!(empty_scalar.len(), Some(0));
        assert_eq!(empty_scalar.is_empty(), Some(true));

        let non_empty_scalar = Utf8Scalar::try_from(&non_empty).unwrap();
        assert_eq!(non_empty_scalar.len(), Some(5));
        assert_eq!(non_empty_scalar.is_empty(), Some(false));
    }

    #[test]
    fn test_utf8_value_ref() {
        let data = "test string";
        let utf8 = Scalar::utf8(data, Nullability::NonNullable);
        let scalar = Utf8Scalar::try_from(&utf8).unwrap();

        // value_ref should not clone
        let value_ref = scalar.value_ref().unwrap();
        assert_eq!(value_ref.as_str(), data);

        // value should clone
        let value = scalar.value().unwrap();
        assert_eq!(value.as_str(), data);
    }

    #[test]
    fn test_utf8_cast_to_utf8() {
        use vortex_dtype::DType;
        use vortex_dtype::Nullability;

        let utf8 = Scalar::utf8("test", Nullability::NonNullable);
        let scalar = Utf8Scalar::try_from(&utf8).unwrap();

        // Cast to nullable utf8
        let result = scalar.cast(&DType::Utf8(Nullability::Nullable)).unwrap();
        assert_eq!(result.dtype(), &DType::Utf8(Nullability::Nullable));

        let casted = Utf8Scalar::try_from(&result).unwrap();
        assert_eq!(casted.value().unwrap().as_str(), "test");
    }

    #[test]
    fn test_utf8_cast_to_non_utf8_fails() {
        use vortex_dtype::DType;
        use vortex_dtype::Nullability;
        use vortex_dtype::PType;

        let utf8 = Scalar::utf8("test", Nullability::NonNullable);
        let scalar = Utf8Scalar::try_from(&utf8).unwrap();

        let result = scalar.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_from_scalar_value_non_utf8_dtype() {
        use vortex_dtype::DType;
        use vortex_dtype::Nullability;
        use vortex_dtype::PType;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let value = crate::ScalarValue(crate::InnerScalarValue::Primitive(crate::PValue::I32(42)));

        let result = Utf8Scalar::from_scalar_value(&dtype, value);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_non_utf8_scalar() {
        use vortex_dtype::Nullability;

        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        let result = Utf8Scalar::try_from(&scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_upper_bound_null() {
        let null_utf8 = Scalar::null(vortex_dtype::DType::Utf8(Nullability::Nullable));
        let scalar = Utf8Scalar::try_from(&null_utf8).unwrap();

        let result = scalar.upper_bound(10);
        assert!(result.is_some());
        assert!(result.unwrap().value().is_none());
    }

    #[test]
    fn test_lower_bound_null() {
        let null_utf8 = Scalar::null(vortex_dtype::DType::Utf8(Nullability::Nullable));
        let scalar = Utf8Scalar::try_from(&null_utf8).unwrap();

        let result = scalar.lower_bound(10);
        assert!(result.value().is_none());
    }

    #[test]
    fn test_upper_bound_exact_length() {
        let utf8 = Scalar::utf8("abc", Nullability::NonNullable);
        let scalar = Utf8Scalar::try_from(&utf8).unwrap();

        let result = scalar.upper_bound(3);
        assert!(result.is_some());
        let upper = result.unwrap();
        assert_eq!(upper.value().unwrap().as_str(), "abc");
    }

    #[test]
    fn test_lower_bound_exact_length() {
        let utf8 = Scalar::utf8("abc", Nullability::NonNullable);
        let scalar = Utf8Scalar::try_from(&utf8).unwrap();

        let result = scalar.lower_bound(3);
        assert_eq!(result.value().unwrap().as_str(), "abc");
    }

    #[test]
    fn test_from_str() {
        let data = "hello world";
        let scalar: Scalar = data.into();

        assert_eq!(
            scalar.dtype(),
            &vortex_dtype::DType::Utf8(Nullability::NonNullable)
        );
        let utf8 = Utf8Scalar::try_from(&scalar).unwrap();
        assert_eq!(utf8.value().unwrap().as_str(), data);
    }

    #[test]
    fn test_from_string() {
        let data = String::from("hello world");
        let scalar: Scalar = data.into();

        assert_eq!(
            scalar.dtype(),
            &vortex_dtype::DType::Utf8(Nullability::NonNullable)
        );
        let utf8 = Utf8Scalar::try_from(&scalar).unwrap();
        assert_eq!(utf8.value().unwrap().as_str(), "hello world");
    }

    #[test]
    fn test_from_buffer_string() {
        use vortex_buffer::BufferString;

        let data = BufferString::from("test");
        let scalar: Scalar = data.into();

        assert_eq!(
            scalar.dtype(),
            &vortex_dtype::DType::Utf8(Nullability::NonNullable)
        );
        let utf8 = Utf8Scalar::try_from(&scalar).unwrap();
        assert_eq!(utf8.value().unwrap().as_str(), "test");
    }

    #[test]
    fn test_from_arc_buffer_string() {
        use std::sync::Arc;

        use vortex_buffer::BufferString;

        let data = Arc::new(BufferString::from("test"));
        let scalar: Scalar = data.into();

        assert_eq!(
            scalar.dtype(),
            &vortex_dtype::DType::Utf8(Nullability::NonNullable)
        );
        let utf8 = Utf8Scalar::try_from(&scalar).unwrap();
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
        let null_scalar = Scalar::null(vortex_dtype::DType::Utf8(Nullability::Nullable));
        let null_buffer: Option<BufferString> = (&null_scalar).try_into().unwrap();
        assert!(null_buffer.is_none());
    }

    #[test]
    fn test_try_from_non_utf8_to_buffer_string() {
        use vortex_buffer::BufferString;
        use vortex_dtype::Nullability;

        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);

        let result: Result<BufferString, _> = (&scalar).try_into();
        assert!(result.is_err());

        let result2: Result<Option<BufferString>, _> = (&scalar).try_into();
        assert!(result2.is_err());
    }

    #[test]
    fn test_scalar_value_from_str() {
        let data = "test";
        let value: crate::ScalarValue = data.into();

        let scalar = Scalar::new(vortex_dtype::DType::Utf8(Nullability::NonNullable), value);
        let utf8 = Utf8Scalar::try_from(&scalar).unwrap();
        assert_eq!(utf8.value().unwrap().as_str(), data);
    }

    #[test]
    fn test_scalar_value_from_string() {
        let data = String::from("test");
        let value: crate::ScalarValue = data.clone().into();

        let scalar = Scalar::new(vortex_dtype::DType::Utf8(Nullability::NonNullable), value);
        let utf8 = Utf8Scalar::try_from(&scalar).unwrap();
        assert_eq!(utf8.value().unwrap().as_str(), &data);
    }

    #[test]
    fn test_scalar_value_from_buffer_string() {
        use vortex_buffer::BufferString;

        let data = BufferString::from("test");
        let value: crate::ScalarValue = data.into();

        let scalar = Scalar::new(vortex_dtype::DType::Utf8(Nullability::NonNullable), value);
        let utf8 = Utf8Scalar::try_from(&scalar).unwrap();
        assert_eq!(utf8.value().unwrap().as_str(), "test");
    }

    #[test]
    fn test_utf8_with_emoji() {
        let emoji_str = "Hello 👋 World 🌍!";
        let scalar = Scalar::utf8(emoji_str, Nullability::NonNullable);
        let utf8_scalar = Utf8Scalar::try_from(&scalar).unwrap();

        assert_eq!(utf8_scalar.value().unwrap().as_str(), emoji_str);
        assert!(utf8_scalar.len().unwrap() > emoji_str.chars().count()); // Byte length > char count
    }

    #[test]
    fn test_partial_ord_null() {
        let null_scalar = Scalar::null(vortex_dtype::DType::Utf8(Nullability::Nullable));
        let non_null_scalar = Scalar::utf8("test", Nullability::Nullable);

        let null = Utf8Scalar::try_from(&null_scalar).unwrap();
        let non_null = Utf8Scalar::try_from(&non_null_scalar).unwrap();

        // Null < Some("test")
        assert!(null < non_null);
        assert!(non_null > null);
    }
}
