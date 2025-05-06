use std::fmt::{Display, Formatter};
use std::sync::Arc;

use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_bail, vortex_err};

use crate::{InnerScalarValue, Scalar, ScalarValue};

#[derive(Debug, Hash)]
pub struct BinaryScalar<'a> {
    dtype: &'a DType,
    value: Option<Arc<ByteBuffer>>,
}

impl Display for BinaryScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.value {
            None => write!(f, "null"),
            Some(v) => write!(
                f,
                "\"{}\"",
                v.as_slice().iter().map(|b| format!("{b:x}")).format(" ")
            ),
        }
    }
}

impl PartialEq for BinaryScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.value == other.value
    }
}

impl Eq for BinaryScalar<'_> {}

impl PartialOrd for BinaryScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.value.cmp(&other.value))
    }
}

impl Ord for BinaryScalar<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<'a> BinaryScalar<'a> {
    pub fn from_scalar_value(dtype: &'a DType, value: ScalarValue) -> VortexResult<Self> {
        if !matches!(dtype, DType::Binary(..)) {
            vortex_bail!("Can only construct binary scalar from binary dtype, found {dtype}")
        }
        Ok(Self {
            dtype,
            value: value.as_buffer()?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    pub fn value(&self) -> Option<ByteBuffer> {
        self.value.as_ref().map(|v| v.as_ref().clone())
    }

    /// Construct a value at most `max_length` in size that's greater than ourselves.
    ///
    /// Will return None if constructing greater value overflows
    pub fn upper_bound(self, max_length: usize) -> Option<Self> {
        if let Some(value) = self.value {
            if value.len() > max_length {
                let sliced = value.slice(0..max_length);
                drop(value);
                let mut sliced_mut = sliced.into_mut();
                for b in sliced_mut.iter_mut().rev() {
                    let (incr, overflow) = b.overflowing_add(1);
                    *b = incr;
                    if !overflow {
                        return Some(Self {
                            dtype: self.dtype,
                            value: Some(Arc::new(sliced_mut.freeze())),
                        });
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
                Self {
                    dtype: self.dtype,
                    value: Some(Arc::new(value.slice(0..max_length))),
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
        if !matches!(dtype, DType::Binary(..)) {
            vortex_bail!("Can't cast binary to {}", dtype)
        }
        Ok(Scalar::new(
            dtype.clone(),
            ScalarValue(InnerScalarValue::Buffer(
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

    /// Extract value as a ScalarValue
    pub fn into_value(self) -> ScalarValue {
        ScalarValue(
            self.value
                .map(InnerScalarValue::Buffer)
                .unwrap_or_else(|| InnerScalarValue::Null),
        )
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

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_error::{VortexExpect, VortexUnwrap};

    use crate::{BinaryScalar, Scalar};

    #[test]
    fn lower_bound() {
        let binary = Scalar::binary(buffer![0u8, 5, 47, 33, 129], Nullability::NonNullable);
        let expected = Scalar::binary(buffer![0u8, 5], Nullability::NonNullable);
        assert_eq!(
            BinaryScalar::try_from(&binary)
                .vortex_unwrap()
                .lower_bound(2),
            BinaryScalar::try_from(&expected).vortex_unwrap()
        );
    }

    #[test]
    fn upper_bound() {
        let binary = Scalar::binary(buffer![0u8, 5, 255, 234, 23], Nullability::NonNullable);
        let expected = Scalar::binary(buffer![0u8, 6, 0], Nullability::NonNullable);
        assert_eq!(
            BinaryScalar::try_from(&binary)
                .vortex_unwrap()
                .upper_bound(3)
                .vortex_expect("must have upper bound"),
            BinaryScalar::try_from(&expected).vortex_unwrap()
        );
    }

    #[test]
    fn upper_bound_overflow() {
        let binary = Scalar::binary(buffer![255u8, 255, 255], Nullability::NonNullable);
        assert!(
            BinaryScalar::try_from(&binary)
                .vortex_unwrap()
                .upper_bound(2)
                .is_none()
        );
    }
}
