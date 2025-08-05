// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Additional trait implementations for decimal types to ensure consistency.

use vortex_dtype::{DType, DecimalDType, Nullability};
use vortex_error::{VortexError, vortex_err};

use crate::{DecimalScalar, DecimalValue, Scalar};

// Add From<DecimalValue> for Scalar to match other types
impl From<DecimalValue> for Scalar {
    fn from(value: DecimalValue) -> Self {
        // Default to a reasonable precision and scale
        // This matches how primitive types work - they get a default nullability
        let dtype = match &value {
            DecimalValue::I8(_) => DecimalDType::new(3, 0),
            DecimalValue::I16(_) => DecimalDType::new(5, 0),
            DecimalValue::I32(_) => DecimalDType::new(10, 0),
            DecimalValue::I64(_) => DecimalDType::new(19, 0),
            DecimalValue::I128(_) => DecimalDType::new(38, 0),
            DecimalValue::I256(_) => DecimalDType::new(76, 0),
        };
        Scalar::decimal(value, dtype, Nullability::NonNullable)
    }
}

// Add TryFrom<&Scalar> for DecimalValue
impl TryFrom<&Scalar> for DecimalValue {
    type Error = VortexError;

    fn try_from(scalar: &Scalar) -> Result<Self, Self::Error> {
        let decimal_scalar = DecimalScalar::try_from(scalar)?;
        decimal_scalar
            .decimal_value()
            .as_ref()
            .cloned()
            .ok_or_else(|| vortex_err!("Cannot extract DecimalValue from null decimal"))
    }
}

// Add TryFrom<Scalar> for DecimalValue (delegates to &Scalar)
impl TryFrom<Scalar> for DecimalValue {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        DecimalValue::try_from(&scalar)
    }
}

// Add TryFrom<&Scalar> for Option<DecimalValue>
impl TryFrom<&Scalar> for Option<DecimalValue> {
    type Error = VortexError;

    fn try_from(scalar: &Scalar) -> Result<Self, Self::Error> {
        let decimal_scalar = DecimalScalar::try_from(scalar)?;
        Ok(decimal_scalar.decimal_value().clone())
    }
}

// Add TryFrom<Scalar> for Option<DecimalValue> (delegates to &Scalar)
impl TryFrom<Scalar> for Option<DecimalValue> {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        Option::<DecimalValue>::try_from(&scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimal_value_from_scalar() {
        let value = DecimalValue::I32(12345);
        let scalar = Scalar::from(value.clone());
        
        // Test extraction
        let extracted: DecimalValue = DecimalValue::try_from(&scalar).unwrap();
        assert_eq!(extracted, value);
        
        // Test owned extraction
        let extracted_owned: DecimalValue = DecimalValue::try_from(scalar.clone()).unwrap();
        assert_eq!(extracted_owned, value);
    }

    #[test]
    fn test_decimal_value_option_from_scalar() {
        // Non-null case
        let value = DecimalValue::I64(999999);
        let scalar = Scalar::from(value.clone());
        
        let extracted: Option<DecimalValue> = Option::try_from(&scalar).unwrap();
        assert_eq!(extracted, Some(value));
        
        // Null case
        let null_scalar = Scalar::null(DType::Decimal(
            DecimalDType::new(10, 2),
            Nullability::Nullable,
        ));
        
        let extracted_null: Option<DecimalValue> = Option::try_from(&null_scalar).unwrap();
        assert_eq!(extracted_null, None);
    }

    #[test]
    fn test_decimal_value_from_conversion() {
        // Test that From<DecimalValue> creates reasonable defaults
        let values = vec![
            DecimalValue::I8(127),
            DecimalValue::I16(32767),
            DecimalValue::I32(1000000),
            DecimalValue::I64(1000000000000),
            DecimalValue::I128(123456789012345678901234567890),
            DecimalValue::I256(crate::i256::from_i128(987654321)),
        ];
        
        for value in values {
            let scalar = Scalar::from(value.clone());
            assert!(!scalar.is_null());
            
            // Verify we can extract it back
            let extracted: DecimalValue = DecimalValue::try_from(&scalar).unwrap();
            assert_eq!(extracted, value);
        }
    }
}