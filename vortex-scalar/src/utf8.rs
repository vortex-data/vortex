use std::sync::Arc;

use vortex_buffer::BufferString;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_bail, vortex_err};

use crate::{InnerScalarValue, Scalar, ScalarValue};

#[derive(Debug, Hash)]
pub struct Utf8Scalar<'a> {
    dtype: &'a DType,
    value: Option<BufferString>,
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
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    pub fn value(&self) -> Option<BufferString> {
        self.value.as_ref().cloned()
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if !matches!(dtype, DType::Utf8(..)) {
            vortex_bail!("Can't cast utf8 to {}", dtype)
        }
        Ok(Scalar::new(
            dtype.clone(),
            ScalarValue(InnerScalarValue::BufferString(Arc::new(
                self.value
                    .as_ref()
                    .vortex_expect("nullness handled in Scalar::cast")
                    .clone(),
            ))),
        ))
    }

    /// Returns whether its value is non-null and empty, otherwise `None`.
    pub fn is_empty(&self) -> Option<bool> {
        self.value.as_ref().map(|v| v.is_empty())
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
