// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};
use std::sync::Arc;

use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_bail, vortex_err};

use crate::{InnerScalarValue, Scalar, ScalarValue};

/// A scalar value representing binary data.
///
/// This type provides a view into a binary scalar value, which can be either
/// a valid byte buffer or null.
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
        Some(self.cmp(other))
    }
}

impl Ord for BinaryScalar<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<'a> BinaryScalar<'a> {
    /// Creates a binary scalar from a data type and scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not a binary type.
    pub fn from_scalar_value(dtype: &'a DType, value: ScalarValue) -> VortexResult<Self> {
        if !matches!(dtype, DType::Binary(..)) {
            vortex_bail!("Can only construct binary scalar from binary dtype, found {dtype}")
        }
        Ok(Self {
            dtype,
            value: value.as_buffer()?,
        })
    }

    /// Returns the data type of this binary scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the binary value as a byte buffer, or None if null.
    pub fn value(&self) -> Option<ByteBuffer> {
        self.value.as_ref().map(|v| v.as_ref().clone())
    }

    /// Returns a reference to the binary value, or None if null.
    /// This avoids cloning the underlying ByteBuffer.
    pub fn value_ref(&self) -> Option<&ByteBuffer> {
        self.value.as_ref().map(|v| v.as_ref())
    }

    /// Constructs a value at most `max_length` in size that's greater than this value.
    ///
    /// Returns None if constructing a greater value would overflow.
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
            vortex_bail!(
                "Cannot cast binary to {dtype}: binary scalars can only be cast to binary types with different nullability"
            )
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
}

impl Scalar {
    /// Creates a new binary scalar from a byte buffer.
    pub fn binary(buffer: impl Into<ByteBuffer>, nullability: Nullability) -> Self {
        Self::new(
            DType::Binary(nullability),
            ScalarValue(InnerScalarValue::Buffer(Arc::new(buffer.into()))),
        )
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
            value: value.value().as_buffer()?,
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
        Self::new(DType::Binary(Nullability::NonNullable), value.into())
    }
}

impl From<Arc<ByteBuffer>> for Scalar {
    fn from(value: Arc<ByteBuffer>) -> Self {
        Self::new(
            DType::Binary(Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Buffer(value)),
        )
    }
}

impl From<&[u8]> for ScalarValue {
    fn from(value: &[u8]) -> Self {
        ScalarValue::from(ByteBuffer::from(value.to_vec()))
    }
}

impl From<ByteBuffer> for ScalarValue {
    fn from(value: ByteBuffer) -> Self {
        ScalarValue(InnerScalarValue::Buffer(Arc::new(value)))
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rstest::rstest;
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

    #[rstest]
    #[case(&[1u8, 2, 3], &[1u8, 2, 3], true)]
    #[case(&[1u8, 2, 3], &[1u8, 2, 4], false)]
    #[case(&[], &[], true)]
    #[case(&[255u8], &[255u8], true)]
    fn test_binary_scalar_equality(
        #[case] data1: &[u8],
        #[case] data2: &[u8],
        #[case] expected: bool,
    ) {
        let binary1 = Scalar::binary(data1.to_vec(), Nullability::NonNullable);
        let binary2 = Scalar::binary(data2.to_vec(), Nullability::NonNullable);

        let scalar1 = BinaryScalar::try_from(&binary1).unwrap();
        let scalar2 = BinaryScalar::try_from(&binary2).unwrap();

        assert_eq!(scalar1 == scalar2, expected);
    }

    #[rstest]
    #[case(&[1u8, 2, 3], &[1u8, 2, 4], Ordering::Less)]
    #[case(&[1u8, 2, 4], &[1u8, 2, 3], Ordering::Greater)]
    #[case(&[1u8, 2, 3], &[1u8, 2, 3], Ordering::Equal)]
    #[case(&[], &[1u8], Ordering::Less)]
    #[case(&[2u8, 0, 0], &[1u8, 255, 255], Ordering::Greater)]
    fn test_binary_scalar_ordering(
        #[case] data1: &[u8],
        #[case] data2: &[u8],
        #[case] expected: Ordering,
    ) {
        let binary1 = Scalar::binary(data1.to_vec(), Nullability::NonNullable);
        let binary2 = Scalar::binary(data2.to_vec(), Nullability::NonNullable);

        let scalar1 = BinaryScalar::try_from(&binary1).unwrap();
        let scalar2 = BinaryScalar::try_from(&binary2).unwrap();

        assert_eq!(scalar1.partial_cmp(&scalar2), Some(expected));
    }

    #[test]
    fn test_binary_null_value() {
        let null_binary = Scalar::null(vortex_dtype::DType::Binary(Nullability::Nullable));
        let scalar = BinaryScalar::try_from(&null_binary).unwrap();

        assert!(scalar.value().is_none());
        assert!(scalar.value_ref().is_none());
        assert!(scalar.len().is_none());
        assert!(scalar.is_empty().is_none());
    }

    #[test]
    fn test_binary_len_and_empty() {
        use vortex_buffer::ByteBuffer;

        let empty = Scalar::binary(ByteBuffer::empty(), Nullability::NonNullable);
        let non_empty = Scalar::binary(buffer![1u8, 2, 3], Nullability::NonNullable);

        let empty_scalar = BinaryScalar::try_from(&empty).unwrap();
        assert_eq!(empty_scalar.len(), Some(0));
        assert_eq!(empty_scalar.is_empty(), Some(true));

        let non_empty_scalar = BinaryScalar::try_from(&non_empty).unwrap();
        assert_eq!(non_empty_scalar.len(), Some(3));
        assert_eq!(non_empty_scalar.is_empty(), Some(false));
    }

    #[test]
    fn test_binary_value_ref() {
        use vortex_buffer::ByteBuffer;

        let data = vec![1u8, 2, 3, 4, 5];
        let binary = Scalar::binary(ByteBuffer::from(data.clone()), Nullability::NonNullable);
        let scalar = BinaryScalar::try_from(&binary).unwrap();

        // value_ref should not clone
        let value_ref = scalar.value_ref().unwrap();
        assert_eq!(value_ref.as_slice(), &data);

        // value should clone
        let value = scalar.value().unwrap();
        assert_eq!(value.as_slice(), &data);
    }

    #[test]
    fn test_binary_cast_to_binary() {
        use vortex_dtype::{DType, Nullability};

        let binary = Scalar::binary(buffer![1u8, 2, 3], Nullability::NonNullable);
        let scalar = BinaryScalar::try_from(&binary).unwrap();

        // Cast to nullable binary
        let result = scalar.cast(&DType::Binary(Nullability::Nullable)).unwrap();
        assert_eq!(result.dtype(), &DType::Binary(Nullability::Nullable));

        let casted = BinaryScalar::try_from(&result).unwrap();
        assert_eq!(casted.value().unwrap().as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_binary_cast_to_non_binary_fails() {
        use vortex_dtype::{DType, Nullability, PType};

        let binary = Scalar::binary(buffer![1u8, 2, 3], Nullability::NonNullable);
        let scalar = BinaryScalar::try_from(&binary).unwrap();

        let result = scalar.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_from_scalar_value_non_binary_dtype() {
        use vortex_dtype::{DType, Nullability, PType};

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let value = crate::ScalarValue(crate::InnerScalarValue::Primitive(crate::PValue::I32(42)));

        let result = BinaryScalar::from_scalar_value(&dtype, value);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_non_binary_scalar() {
        use vortex_dtype::Nullability;

        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        let result = BinaryScalar::try_from(&scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_upper_bound_null() {
        let null_binary = Scalar::null(vortex_dtype::DType::Binary(Nullability::Nullable));
        let scalar = BinaryScalar::try_from(&null_binary).unwrap();

        let result = scalar.upper_bound(10);
        assert!(result.is_some());
        assert!(result.unwrap().value().is_none());
    }

    #[test]
    fn test_lower_bound_null() {
        let null_binary = Scalar::null(vortex_dtype::DType::Binary(Nullability::Nullable));
        let scalar = BinaryScalar::try_from(&null_binary).unwrap();

        let result = scalar.lower_bound(10);
        assert!(result.value().is_none());
    }

    #[test]
    fn test_upper_bound_exact_length() {
        let binary = Scalar::binary(buffer![1u8, 2, 3], Nullability::NonNullable);
        let scalar = BinaryScalar::try_from(&binary).unwrap();

        let result = scalar.upper_bound(3);
        assert!(result.is_some());
        let upper = result.unwrap();
        assert_eq!(upper.value().unwrap().as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_lower_bound_exact_length() {
        let binary = Scalar::binary(buffer![1u8, 2, 3], Nullability::NonNullable);
        let scalar = BinaryScalar::try_from(&binary).unwrap();

        let result = scalar.lower_bound(3);
        assert_eq!(result.value().unwrap().as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_from_slice() {
        let data: &[u8] = &[1u8, 2, 3, 4];
        let scalar: Scalar = data.into();

        assert_eq!(
            scalar.dtype(),
            &vortex_dtype::DType::Binary(Nullability::NonNullable)
        );
        let binary = BinaryScalar::try_from(&scalar).unwrap();
        assert_eq!(binary.value().unwrap().as_slice(), data);
    }

    #[test]
    fn test_try_from_scalar_to_bytebuffer() {
        use vortex_buffer::ByteBuffer;

        let data = vec![5u8, 6, 7];
        let scalar = Scalar::binary(ByteBuffer::from(data.clone()), Nullability::NonNullable);

        // Try from &Scalar
        let buffer: ByteBuffer = (&scalar).try_into().unwrap();
        assert_eq!(buffer.as_slice(), &data);

        // Try from Scalar (owned)
        let data2 = vec![5u8, 6, 7];
        let scalar2 = Scalar::binary(ByteBuffer::from(data2.clone()), Nullability::NonNullable);
        let buffer2: ByteBuffer = scalar2.try_into().unwrap();
        assert_eq!(buffer2.as_slice(), &data2);
    }

    #[test]
    fn test_try_from_scalar_to_option_bytebuffer() {
        use vortex_buffer::ByteBuffer;

        // Non-null case
        let data = vec![5u8, 6, 7];
        let scalar = Scalar::binary(ByteBuffer::from(data.clone()), Nullability::Nullable);
        let buffer: Option<ByteBuffer> = (&scalar).try_into().unwrap();
        assert_eq!(buffer.unwrap().as_slice(), &data);

        // Null case
        let null_scalar = Scalar::null(vortex_dtype::DType::Binary(Nullability::Nullable));
        let null_buffer: Option<ByteBuffer> = (&null_scalar).try_into().unwrap();
        assert!(null_buffer.is_none());
    }

    #[test]
    fn test_try_from_non_binary_to_bytebuffer() {
        use vortex_buffer::ByteBuffer;
        use vortex_dtype::Nullability;

        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);

        let result: Result<ByteBuffer, _> = (&scalar).try_into();
        assert!(result.is_err());

        let result2: Result<Option<ByteBuffer>, _> = (&scalar).try_into();
        assert!(result2.is_err());
    }

    #[test]
    fn test_from_arc_bytebuffer() {
        use std::sync::Arc;

        use vortex_buffer::ByteBuffer;

        let data = vec![10u8, 20, 30];
        let buffer = Arc::new(ByteBuffer::from(data.clone()));
        let scalar: Scalar = buffer.into();

        assert_eq!(
            scalar.dtype(),
            &vortex_dtype::DType::Binary(Nullability::NonNullable)
        );
        let binary = BinaryScalar::try_from(&scalar).unwrap();
        assert_eq!(binary.value().unwrap().as_slice(), &data);
    }

    #[test]
    fn test_scalar_value_from_slice() {
        let data: &[u8] = &[100u8, 200];
        let value: crate::ScalarValue = data.into();

        let scalar = Scalar::new(vortex_dtype::DType::Binary(Nullability::NonNullable), value);
        let binary = BinaryScalar::try_from(&scalar).unwrap();
        assert_eq!(binary.value().unwrap().as_slice(), data);
    }

    #[test]
    fn test_scalar_value_from_bytebuffer() {
        use vortex_buffer::ByteBuffer;

        let data = vec![111u8, 222];
        let buffer = ByteBuffer::from(data.clone());
        let value: crate::ScalarValue = buffer.into();

        let scalar = Scalar::new(vortex_dtype::DType::Binary(Nullability::NonNullable), value);
        let binary = BinaryScalar::try_from(&scalar).unwrap();
        assert_eq!(binary.value().unwrap().as_slice(), &data);
    }
}
