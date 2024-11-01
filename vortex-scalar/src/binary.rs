use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::value::ScalarValue;
use crate::Scalar;

pub struct BinaryScalar<'a> {
    dtype: &'a DType,
    value: Option<Buffer>,
}

impl<'a> BinaryScalar<'a> {
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    pub fn value(&self) -> Option<Buffer> {
        self.value.as_ref().cloned()
    }

    pub fn cast(&self, _dtype: &DType) -> VortexResult<Scalar> {
        todo!()
    }
}

impl Scalar {
    pub fn binary(buffer: Buffer, nullability: Nullability) -> Self {
        Self {
            dtype: DType::Binary(nullability),
            value: ScalarValue::Buffer(buffer),
        }
    }
}

impl<'a> TryFrom<&'a Scalar> for BinaryScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        if !matches!(value.dtype(), DType::Binary(_)) {
            vortex_bail!("Expected binary scalar, found {}", value.dtype())
        }
        Ok(Self {
            dtype: value.dtype(),
            value: value.value.as_buffer()?,
        })
    }
}

impl<'a> TryFrom<&'a Scalar> for Buffer {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> VortexResult<Self> {
        Buffer::try_from(scalar.value())
    }
}

impl TryFrom<Scalar> for Buffer {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> VortexResult<Self> {
        Buffer::try_from(&scalar)
    }
}

impl TryFrom<&ScalarValue> for Buffer {
    type Error = VortexError;

    fn try_from(value: &ScalarValue) -> Result<Self, Self::Error> {
        Option::<Buffer>::try_from(value)?
            .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
    }
}

impl TryFrom<ScalarValue> for Buffer {
    type Error = VortexError;

    fn try_from(value: ScalarValue) -> Result<Self, Self::Error> {
        Buffer::try_from(&value)
    }
}

impl TryFrom<&ScalarValue> for Option<Buffer> {
    type Error = VortexError;

    fn try_from(value: &ScalarValue) -> Result<Self, Self::Error> {
        value.as_buffer()
    }
}

impl TryFrom<ScalarValue> for Option<Buffer> {
    type Error = VortexError;

    fn try_from(value: ScalarValue) -> Result<Self, Self::Error> {
        Option::<Buffer>::try_from(&value)
    }
}
