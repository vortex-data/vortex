// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`BinaryScalar`] typed view implementation.

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A scalar value representing binary data.
///
/// This type provides a view into a binary scalar value, which can be either
/// a valid byte buffer or null.
#[derive(Debug, Clone, Hash)]
pub struct BinaryScalar<'a> {
    /// The data type of this scalar.
    dtype: &'a DType,
    /// The binary value, or [`None`] if null.
    value: Option<&'a ByteBuffer>,
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
    pub fn try_new(dtype: &'a DType, value: Option<&'a ScalarValue>) -> VortexResult<Self> {
        if !matches!(dtype, DType::Binary(..)) {
            vortex_bail!("Can only construct binary scalar from binary dtype, found {dtype}")
        }

        Ok(Self {
            dtype,
            value: value.map(|value| value.as_binary()),
        })
    }

    /// Returns the data type of this binary scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns a reference to the binary value, or None if null.
    /// This avoids cloning the underlying ByteBuffer.
    pub fn value(&self) -> Option<&'a ByteBuffer> {
        self.value
    }

    /// Casts this scalar to the given `dtype`.
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if !matches!(dtype, DType::Binary(..)) {
            vortex_bail!(
                "Cannot cast binary to {dtype}: binary scalars can only be cast to binary types with different nullability"
            )
        }
        Scalar::try_new(
            dtype.clone(),
            Some(ScalarValue::Binary(
                self.value()
                    .cloned()
                    .vortex_expect("nullness handled in Scalar::cast"),
            )),
        )
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

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::dtype::Nullability;
    use crate::scalar::BinaryScalar;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

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

        let scalar1 = binary1.as_binary();
        let scalar2 = binary2.as_binary();

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

        let scalar1 = binary1.as_binary();
        let scalar2 = binary2.as_binary();

        assert_eq!(scalar1.partial_cmp(&scalar2), Some(expected));
    }

    #[test]
    fn test_binary_null_value() {
        let null_binary = Scalar::null(crate::dtype::DType::Binary(Nullability::Nullable));
        let scalar = null_binary.as_binary();

        assert!(scalar.value().is_none());
        assert!(scalar.value().is_none());
        assert!(scalar.len().is_none());
        assert!(scalar.is_empty().is_none());
    }

    #[test]
    fn test_binary_len_and_empty() {
        use vortex_buffer::ByteBuffer;

        let empty = Scalar::binary(ByteBuffer::empty(), Nullability::NonNullable);
        let non_empty = Scalar::binary(buffer![1u8, 2, 3], Nullability::NonNullable);

        let empty_scalar = empty.as_binary();
        assert_eq!(empty_scalar.len(), Some(0));
        assert_eq!(empty_scalar.is_empty(), Some(true));

        let non_empty_scalar = non_empty.as_binary();
        assert_eq!(non_empty_scalar.len(), Some(3));
        assert_eq!(non_empty_scalar.is_empty(), Some(false));
    }

    #[test]
    fn test_binary_value_ref() {
        use vortex_buffer::ByteBuffer;

        let data = vec![1u8, 2, 3, 4, 5];
        let binary = Scalar::binary(ByteBuffer::from(data.clone()), Nullability::NonNullable);
        let scalar = binary.as_binary();

        // value_ref should not clone
        let value_ref = scalar.value().unwrap();
        assert_eq!(value_ref.as_slice(), &data);

        // to_value should clone
        let value = scalar.value().unwrap();
        assert_eq!(value.as_slice(), &data);
    }

    #[test]
    fn test_binary_cast_to_binary() {
        use crate::dtype::DType;
        use crate::dtype::Nullability;

        let binary = Scalar::binary(buffer![1u8, 2, 3], Nullability::NonNullable);
        let scalar = binary.as_binary();

        // Cast to nullable binary
        let result = scalar.cast(&DType::Binary(Nullability::Nullable)).unwrap();
        assert_eq!(result.dtype(), &DType::Binary(Nullability::Nullable));

        let casted = result.as_binary();
        assert_eq!(casted.value().unwrap().as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_binary_cast_to_non_binary_fails() {
        use crate::dtype::DType;
        use crate::dtype::Nullability;
        use crate::dtype::PType;

        let binary = Scalar::binary(buffer![1u8, 2, 3], Nullability::NonNullable);
        let scalar = binary.as_binary();

        let result = scalar.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_from_scalar_value_non_binary_dtype() {
        use crate::dtype::DType;
        use crate::dtype::Nullability;
        use crate::dtype::PType;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let value = ScalarValue::Primitive(PValue::I32(42));

        let result = BinaryScalar::try_new(&dtype, Some(&value));
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_non_binary_scalar() {
        use crate::dtype::Nullability;

        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        assert!(scalar.as_binary_opt().is_none());
    }

    #[test]
    fn test_from_slice() {
        let data: &[u8] = &[1u8, 2, 3, 4];
        let scalar: Scalar = data.into();

        assert_eq!(
            scalar.dtype(),
            &crate::dtype::DType::Binary(Nullability::NonNullable)
        );
        let binary = scalar.as_binary();
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
        let null_scalar = Scalar::null(crate::dtype::DType::Binary(Nullability::Nullable));
        let null_buffer: Option<ByteBuffer> = (&null_scalar).try_into().unwrap();
        assert!(null_buffer.is_none());
    }

    #[test]
    fn test_try_from_non_binary_to_bytebuffer() {
        use vortex_buffer::ByteBuffer;

        use crate::dtype::Nullability;

        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);

        let result: Result<ByteBuffer, _> = (&scalar).try_into();
        assert!(result.is_err());

        let result2: Result<Option<ByteBuffer>, _> = (&scalar).try_into();
        assert!(result2.is_err());
    }

    #[test]
    fn test_from_arc_bytebuffer() {
        use vortex_buffer::ByteBuffer;

        let data = vec![10u8, 20, 30];
        let buffer = ByteBuffer::from(data.clone());
        let scalar: Scalar = buffer.into();

        assert_eq!(
            scalar.dtype(),
            &crate::dtype::DType::Binary(Nullability::NonNullable)
        );
        let binary = scalar.as_binary();
        assert_eq!(binary.value().unwrap().as_slice(), &data);
    }

    #[test]
    fn test_scalar_value_from_slice() {
        let data: &[u8] = &[100u8, 200];
        let value: ScalarValue = data.into();

        let scalar = Scalar::new(
            crate::dtype::DType::Binary(Nullability::NonNullable),
            Some(value),
        );
        let binary = scalar.as_binary();
        assert_eq!(binary.value().unwrap().as_slice(), data);
    }

    #[test]
    fn test_scalar_value_from_bytebuffer() {
        use vortex_buffer::ByteBuffer;

        let data = vec![111u8, 222];
        let buffer = ByteBuffer::from(data.clone());
        let value: ScalarValue = buffer.into();

        let scalar = Scalar::new(
            crate::dtype::DType::Binary(Nullability::NonNullable),
            Some(value),
        );
        let binary = scalar.as_binary();
        assert_eq!(binary.value().unwrap().as_slice(), &data);
    }
}
