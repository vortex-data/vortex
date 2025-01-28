use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect as _, VortexResult};

use crate::value::{InnerScalarValue, ScalarValue};
use crate::Scalar;

#[derive(Debug, Hash)]
pub struct BinaryScalar<'a> {
    dtype: &'a DType,
    value: Option<ByteBuffer>,
}

impl PartialEq for BinaryScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype && self.value == other.value
    }
}

impl Eq for BinaryScalar<'_> {}

/// Ord is not implemented since it's undefined for different nullability
impl PartialOrd for BinaryScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<'a> BinaryScalar<'a> {
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    pub fn value(&self) -> Option<ByteBuffer> {
        self.value.as_ref().cloned()
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if !matches!(dtype, DType::Binary(..)) {
            vortex_bail!("Can't cast binary to {}", dtype)
        }
        Ok(Scalar::new(
            dtype.clone(),
            ScalarValue(InnerScalarValue::Buffer(Arc::new(
                self.value
                    .as_ref()
                    .vortex_expect("nullness handled in Scalar::cast")
                    .clone(),
            ))),
        ))
    }
}

impl Scalar {
    pub fn binary(buffer: impl Into<Arc<ByteBuffer>>, nullability: Nullability) -> Self {
        Self {
            dtype: DType::Binary(nullability),
            value: ScalarValue(InnerScalarValue::Buffer(buffer.into())),
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

impl<'a> TryFrom<&'a Scalar> for ByteBuffer {
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

impl<'a> TryFrom<&'a Scalar> for Option<ByteBuffer> {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> VortexResult<Self> {
        Ok(scalar
            .as_binary_opt()
            .ok_or_else(|| vortex_err!("Cannot extract buffer from non-buffer scalar"))?
            .value())
    }
}

impl TryFrom<Scalar> for ByteBuffer {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> VortexResult<Self> {
        Self::try_from(&scalar)
    }
}

impl TryFrom<Scalar> for Option<ByteBuffer> {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> VortexResult<Self> {
        Self::try_from(&scalar)
    }
}

impl From<&[u8]> for Scalar {
    fn from(value: &[u8]) -> Self {
        Scalar::from(ByteBuffer::from(value.to_vec()))
    }
}

impl From<ByteBuffer> for Scalar {
    fn from(value: ByteBuffer) -> Self {
        Self {
            dtype: DType::Binary(Nullability::NonNullable),
            value: ScalarValue(InnerScalarValue::Buffer(Arc::new(value))),
        }
    }
}

impl From<Arc<ByteBuffer>> for Scalar {
    fn from(value: Arc<ByteBuffer>) -> Self {
        Self {
            dtype: DType::Binary(Nullability::NonNullable),
            value: ScalarValue(InnerScalarValue::Buffer(value)),
        }
    }
}
