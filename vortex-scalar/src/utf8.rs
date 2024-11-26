use vortex_buffer::BufferString;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::value::ScalarValue;
use crate::{InnerScalarValue, Scalar};

pub struct Utf8Scalar<'a> {
    dtype: &'a DType,
    value: Option<BufferString>,
}

impl<'a> Utf8Scalar<'a> {
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    pub fn value(&self) -> Option<BufferString> {
        self.value.as_ref().cloned()
    }

    pub fn cast(&self, _dtype: &DType) -> VortexResult<Scalar> {
        todo!()
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
            value: ScalarValue(InnerScalarValue::BufferString(str.try_into()?)),
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
            value: ScalarValue(InnerScalarValue::BufferString(value.to_string().into())),
        }
    }
}

impl From<String> for Scalar {
    fn from(value: String) -> Self {
        Self {
            dtype: DType::Utf8(NonNullable),
            value: ScalarValue(InnerScalarValue::BufferString(value.into())),
        }
    }
}

impl From<BufferString> for Scalar {
    fn from(value: BufferString) -> Self {
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
