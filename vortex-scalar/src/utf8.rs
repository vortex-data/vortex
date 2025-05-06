use std::fmt::{Display, Formatter};
use std::sync::Arc;

use vortex_buffer::BufferString;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_bail, vortex_err};

use crate::{InnerScalarValue, Scalar, ScalarValue};

#[derive(Debug, Hash)]
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

impl Eq for Utf8Scalar<'_> {}

impl PartialOrd for Utf8Scalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.value.cmp(&other.value))
    }
}

impl Ord for Utf8Scalar<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<'a> Utf8Scalar<'a> {
    pub fn from_scalar_value(dtype: &'a DType, value: ScalarValue) -> VortexResult<Self> {
        if !matches!(dtype, DType::Utf8(..)) {
            vortex_bail!("Can only construct utf8 scalar from utf8 dtype, found {dtype}")
        }
        Ok(Self {
            dtype,
            value: value.as_buffer_string()?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    pub fn value(&self) -> Option<BufferString> {
        self.value.as_ref().map(|v| v.as_ref().clone())
    }

    /// Construct a value at most `max_length` in size that's greater than ourselves.
    ///
    /// Will return None if constructing greater value overflows
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
            vortex_bail!("Can't cast utf8 to {}", dtype)
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

    /// Convert typed scalar into ScalarValue
    pub fn into_value(self) -> ScalarValue {
        ScalarValue(
            self.value
                .map(InnerScalarValue::BufferString)
                .unwrap_or_else(|| InnerScalarValue::Null),
        )
    }
}

impl Scalar {
    pub fn utf8<B>(str: B, nullability: Nullability) -> Self
    where
        B: Into<BufferString>,
    {
        Self::try_utf8(str, nullability).unwrap()
    }

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

#[cfg(test)]
mod tests {
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
}
