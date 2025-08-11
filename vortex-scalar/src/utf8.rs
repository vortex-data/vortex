// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};
use std::sync::Arc;

use vortex_buffer::BufferString;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_bail, vortex_err};

use crate::{InnerScalarValue, Scalar, ScalarValue};

/// A scalar value representing a UTF-8 encoded string.
///
/// This type provides a view into a UTF-8 string scalar value, which can be either
/// a valid UTF-8 string or null.
#[derive(Debug, Hash, Eq)]
pub struct Utf8Scalar<'a> {
    dtype: &'a DType,
    value: Option<Arc<BufferString>>,
}

impl Display for Utf8Scalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.value {
            None => write!(f, "null"),
            Some(v) => write!(f, "\"{}\"", v.as_str()),
        }
    }
}

impl PartialEq for Utf8Scalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.value == other.value
    }
}

impl PartialOrd for Utf8Scalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Utf8Scalar<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<'a> Utf8Scalar<'a> {
    /// Creates a UTF-8 scalar from a data type and scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not a UTF-8 type.
    pub fn from_scalar_value(dtype: &'a DType, value: ScalarValue) -> VortexResult<Self> {
        if !matches!(dtype, DType::Utf8(..)) {
            vortex_bail!("Can only construct utf8 scalar from utf8 dtype, found {dtype}")
        }
        Ok(Self {
            dtype,
            value: value.as_buffer_string()?,
        })
    }

    /// Returns the data type of this UTF-8 scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the string value, or None if null.
    pub fn value(&self) -> Option<BufferString> {
        self.value.as_ref().map(|v| v.as_ref().clone())
    }

    /// Returns a reference to the string value, or None if null.
    /// This avoids cloning the underlying BufferString.
    pub fn value_ref(&self) -> Option<&BufferString> {
        self.value.as_ref().map(|v| v.as_ref())
    }

    /// Constructs a value at most `max_length` in size that's greater than this value.
    ///
    /// Returns None if constructing a greater value would overflow.
    pub fn upper_bound(self, max_length: usize) -> Option<Self> {
        if let Some(value) = self.value {
            if value.len() > max_length {
                let utf8_split_pos = (max_length.saturating_sub(3)..=max_length)
                    .rfind(|p| value.is_char_boundary(*p))
                    .vortex_expect("Failed to find utf8 character boundary");

                let utf8_mut = value
                    .get(..utf8_split_pos)
                    .vortex_expect("Slicing with existing index");

                for (idx, original_char) in utf8_mut.char_indices().rev() {
                    let original_len = original_char.len_utf8();
                    if let Some(next_char) = char::from_u32(original_char as u32 + 1) {
                        // do not allow increasing byte width of incremented char
                        if next_char.len_utf8() == original_len {
                            let sliced = value.inner().slice(0..idx + original_len);
                            drop(value);
                            let mut result = sliced.into_mut();
                            next_char.encode_utf8(&mut result[idx..]);
                            return Some(Self {
                                dtype: self.dtype,
                                value: Some(Arc::new(unsafe {
                                    BufferString::new_unchecked(result.freeze())
                                })),
                            });
                        }
                    }
                }
                None
            } else {
                Some(Self {
                    dtype: self.dtype,
                    value: Some(value),
                })
            }
        } else {
            Some(self)
        }
    }

    /// Construct a value at most `max_length` in size that's less than ourselves.
    pub fn lower_bound(self, max_length: usize) -> Self {
        if let Some(value) = self.value {
            if value.len() > max_length {
                // UTF8 characters are at most 4 bytes, since we know that BufferString is UTF8 we must have a valid character boundary
                let utf8_split_pos = (max_length.saturating_sub(3)..=max_length)
                    .rfind(|p| value.is_char_boundary(*p))
                    .vortex_expect("Failed to find utf8 character boundary");

                Self {
                    dtype: self.dtype,
                    value: Some(Arc::new(unsafe {
                        BufferString::new_unchecked(value.inner().slice(0..utf8_split_pos))
                    })),
                }
            } else {
                Self {
                    dtype: self.dtype,
                    value: Some(value),
                }
            }
        } else {
            self
        }
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if !matches!(dtype, DType::Utf8(..)) {
            vortex_bail!(
                "Cannot cast utf8 to {dtype}: UTF-8 scalars can only be cast to UTF-8 types with different nullability"
            )
        }
        Ok(Scalar::new(
            dtype.clone(),
            ScalarValue(InnerScalarValue::BufferString(
                self.value
                    .as_ref()
                    .vortex_expect("nullness handled in Scalar::cast")
                    .clone(),
            )),
        ))
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
        Self::try_utf8(str, nullability).unwrap()
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
        Ok(Self {
            dtype: DType::Utf8(nullability),
            value: ScalarValue(InnerScalarValue::BufferString(Arc::new(str.try_into()?))),
        })
    }
}

impl<'a> TryFrom<&'a Scalar> for Utf8Scalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        if !matches!(value.dtype(), DType::Utf8(_)) {
            vortex_bail!("Expected utf8 scalar, found {}", value.dtype())
        }
        Ok(Self {
            dtype: value.dtype(),
            value: value.value.as_buffer_string()?,
        })
    }
}

impl<'a> TryFrom<&'a Scalar> for String {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        Ok(BufferString::try_from(value)?.to_string())
    }
}

impl From<&str> for Scalar {
    fn from(value: &str) -> Self {
        Self {
            dtype: DType::Utf8(NonNullable),
            value: ScalarValue(InnerScalarValue::BufferString(Arc::new(
                value.to_string().into(),
            ))),
        }
    }
}

impl From<String> for Scalar {
    fn from(value: String) -> Self {
        Self {
            dtype: DType::Utf8(NonNullable),
            value: ScalarValue(InnerScalarValue::BufferString(Arc::new(value.into()))),
        }
    }
}

impl From<BufferString> for Scalar {
    fn from(value: BufferString) -> Self {
        Self {
            dtype: DType::Utf8(NonNullable),
            value: ScalarValue(InnerScalarValue::BufferString(Arc::new(value))),
        }
    }
}

impl From<Arc<BufferString>> for Scalar {
    fn from(value: Arc<BufferString>) -> Self {
        Self {
            dtype: DType::Utf8(NonNullable),
            value: ScalarValue(InnerScalarValue::BufferString(value)),
        }
    }
}

impl<'a> TryFrom<&'a Scalar> for BufferString {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> VortexResult<Self> {
        <Option<BufferString>>::try_from(scalar)?
            .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
    }
}

impl TryFrom<Scalar> for BufferString {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        Self::try_from(&scalar)
    }
}

impl<'a> TryFrom<&'a Scalar> for Option<BufferString> {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> Result<Self, Self::Error> {
        Ok(Utf8Scalar::try_from(scalar)?.value())
    }
}

impl TryFrom<Scalar> for Option<BufferString> {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        Self::try_from(&scalar)
    }
}

impl From<&str> for ScalarValue {
    fn from(value: &str) -> Self {
        ScalarValue(InnerScalarValue::BufferString(Arc::new(
            value.to_string().into(),
        )))
    }
}

impl From<String> for ScalarValue {
    fn from(value: String) -> Self {
        ScalarValue(InnerScalarValue::BufferString(Arc::new(value.into())))
    }
}

impl From<BufferString> for ScalarValue {
    fn from(value: BufferString) -> Self {
        ScalarValue(InnerScalarValue::BufferString(Arc::new(value)))
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rstest::rstest;
    use vortex_dtype::Nullability;
    use vortex_error::{VortexExpect, VortexUnwrap};

    use crate::{Scalar, Utf8Scalar};

    #[test]
    fn lower_bound() {
        let utf8 = Scalar::utf8("snowman⛄️snowman", Nullability::NonNullable);
        let expected = Scalar::utf8("snowman", Nullability::NonNullable);
        assert_eq!(
            Utf8Scalar::try_from(&utf8).vortex_unwrap().lower_bound(9),
            Utf8Scalar::try_from(&expected).vortex_unwrap()
        );
    }

    #[test]
    fn upper_bound() {
        let utf8 = Scalar::utf8("char🪩", Nullability::NonNullable);
        let expected = Scalar::utf8("chas", Nullability::NonNullable);
        assert_eq!(
            Utf8Scalar::try_from(&utf8)
                .vortex_unwrap()
                .upper_bound(5)
                .vortex_expect("must have upper bound"),
            Utf8Scalar::try_from(&expected).vortex_unwrap()
        );
    }

    #[test]
    fn upper_bound_overflow() {
        let utf8 = Scalar::utf8("🂑🂒🂓", Nullability::NonNullable);
        assert!(
            Utf8Scalar::try_from(&utf8)
                .vortex_unwrap()
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
        use vortex_dtype::{DType, Nullability};

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
        use vortex_dtype::{DType, Nullability, PType};

        let utf8 = Scalar::utf8("test", Nullability::NonNullable);
        let scalar = Utf8Scalar::try_from(&utf8).unwrap();

        let result = scalar.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_from_scalar_value_non_utf8_dtype() {
        use vortex_dtype::{DType, Nullability, PType};

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
