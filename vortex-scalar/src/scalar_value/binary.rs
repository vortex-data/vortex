// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexResult, vortex_err};

use crate::ScalarValue;
use crate::scalar_value::InnerScalarValue;

impl<'a> TryFrom<&'a ScalarValue> for ByteBuffer {
    type Error = VortexError;

    fn try_from(scalar: &'a ScalarValue) -> VortexResult<Self> {
        scalar
            .as_buffer()?
            .ok_or_else(|| vortex_err!("Can't convert null scalar into a byte buffer"))
            .map(|b| b.as_ref().clone())
    }
}

impl<'a> TryFrom<&'a ScalarValue> for Option<ByteBuffer> {
    type Error = VortexError;

    fn try_from(scalar: &'a ScalarValue) -> VortexResult<Self> {
        Ok(scalar.as_buffer()?.as_ref().map(|b| b.as_ref().clone()))
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
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_value_from_byte_slice() {
        let data = b"hello world";
        let scalar = ScalarValue::from(&data[..]);

        // Verify we can convert back
        let buffer: ByteBuffer = ByteBuffer::try_from(&scalar).unwrap();
        assert_eq!(buffer.as_ref(), data);
    }

    #[test]
    fn test_scalar_value_from_byte_buffer() {
        let data = vec![1u8, 2, 3, 4, 5];
        let byte_buffer = ByteBuffer::from(data.clone());
        let scalar = ScalarValue::from(byte_buffer);

        // Verify we can convert back
        let recovered: ByteBuffer = ByteBuffer::try_from(&scalar).unwrap();
        assert_eq!(recovered.as_ref(), &data[..]);
    }

    #[test]
    fn test_try_from_scalar_to_byte_buffer() {
        let data = vec![255u8, 128, 64, 32, 16, 8, 4, 2, 1];
        let byte_buffer = ByteBuffer::from(data.clone());
        let scalar = ScalarValue::from(byte_buffer);

        let result: Result<ByteBuffer, _> = ByteBuffer::try_from(&scalar);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_ref(), &data[..]);
    }

    #[test]
    fn test_try_from_scalar_to_option_byte_buffer() {
        let data = b"test data";
        let scalar = ScalarValue::from(&data[..]);

        let result: Result<Option<ByteBuffer>, _> = Option::<ByteBuffer>::try_from(&scalar);
        assert!(result.is_ok());
        let option_buffer = result.unwrap();
        assert!(option_buffer.is_some());
        assert_eq!(option_buffer.unwrap().as_ref(), data);
    }

    #[test]
    fn test_null_scalar_to_byte_buffer_fails() {
        use crate::InnerScalarValue;

        let null_scalar = ScalarValue(InnerScalarValue::Null);

        // Direct conversion should return an error (after fixing the bug)
        let result: Result<ByteBuffer, _> = ByteBuffer::try_from(&null_scalar);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Can't convert null scalar into a byte buffer")
        );
    }

    #[test]
    fn test_null_scalar_to_option_byte_buffer() {
        use crate::InnerScalarValue;

        let null_scalar = ScalarValue(InnerScalarValue::Null);

        // Option conversion should succeed with None
        let result: Result<Option<ByteBuffer>, _> = Option::<ByteBuffer>::try_from(&null_scalar);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_empty_byte_slice() {
        let empty = b"";
        let scalar = ScalarValue::from(&empty[..]);

        let buffer: ByteBuffer = ByteBuffer::try_from(&scalar).unwrap();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_large_byte_buffer() {
        let large_data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let byte_buffer = ByteBuffer::from(large_data.clone());
        let scalar = ScalarValue::from(byte_buffer);

        let recovered: ByteBuffer = ByteBuffer::try_from(&scalar).unwrap();
        assert_eq!(recovered.len(), 10000);
        assert_eq!(recovered.as_ref(), &large_data[..]);
    }

    #[test]
    fn test_wrong_scalar_type_to_byte_buffer() {
        use crate::{InnerScalarValue, PValue};

        // Try with a boolean scalar
        let bool_scalar = ScalarValue(InnerScalarValue::Bool(true));
        let result: Result<ByteBuffer, _> = ByteBuffer::try_from(&bool_scalar);
        assert!(result.is_err());

        // Try with a primitive scalar
        let int_scalar = ScalarValue(InnerScalarValue::Primitive(PValue::I32(42)));
        let result2: Result<ByteBuffer, _> = ByteBuffer::try_from(&int_scalar);
        assert!(result2.is_err());
    }

    #[test]
    fn test_byte_buffer_round_trip() {
        let original_data = b"round trip test";

        // Create from slice
        let scalar1 = ScalarValue::from(&original_data[..]);

        // Convert to ByteBuffer
        let buffer1: ByteBuffer = ByteBuffer::try_from(&scalar1).unwrap();

        // Create new scalar from ByteBuffer
        let scalar2 = ScalarValue::from(buffer1.clone());

        // Convert back to ByteBuffer
        let buffer2: ByteBuffer = ByteBuffer::try_from(&scalar2).unwrap();

        // Should be equal
        assert_eq!(buffer1.as_ref(), buffer2.as_ref());
        assert_eq!(buffer2.as_ref(), original_data);
    }

    #[test]
    fn test_arc_sharing() {
        let data = vec![1, 2, 3, 4, 5];
        let byte_buffer = ByteBuffer::from(data);
        let scalar = ScalarValue::from(byte_buffer);

        // Clone the scalar - should share the Arc
        let scalar_clone = scalar.clone();

        // Both should convert to the same data
        let buffer1: ByteBuffer = ByteBuffer::try_from(&scalar).unwrap();
        let buffer2: ByteBuffer = ByteBuffer::try_from(&scalar_clone).unwrap();

        assert_eq!(buffer1.as_ref(), buffer2.as_ref());
    }
}
