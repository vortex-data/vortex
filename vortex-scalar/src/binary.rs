use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::value::{InnerScalarValue, ScalarValue};
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
            value: ScalarValue(InnerScalarValue::Buffer(buffer)),
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
        let binary = scalar
            .as_binary_opt()
            .ok_or_else(|| vortex_err!("Cannot extract buffer from non-buffer scalar"))?;

        binary
            .value()
            .ok_or_else(|| vortex_err!("Cannot extract present value from null scalar"))
    }
}

impl<'a> TryFrom<&'a Scalar> for Option<Buffer> {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> VortexResult<Self> {
        Ok(scalar
            .as_binary_opt()
            .ok_or_else(|| vortex_err!("Cannot extract buffer from non-buffer scalar"))?
            .value())
    }
}

impl TryFrom<Scalar> for Buffer {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> VortexResult<Self> {
        Self::try_from(&scalar)
    }
}

impl TryFrom<Scalar> for Option<Buffer> {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> VortexResult<Self> {
        Self::try_from(&scalar)
    }
}

impl From<&[u8]> for Scalar {
    fn from(value: &[u8]) -> Self {
        Scalar::from(Buffer::from(value))
    }
}

impl From<bytes::Bytes> for Scalar {
    fn from(value: bytes::Bytes) -> Self {
        Scalar::from(Buffer::from(value))
    }
}

impl From<Buffer> for Scalar {
    fn from(value: Buffer) -> Self {
        Self {
            dtype: DType::Binary(Nullability::NonNullable),
            value: ScalarValue(InnerScalarValue::Buffer(value)),
        }
    }
}
