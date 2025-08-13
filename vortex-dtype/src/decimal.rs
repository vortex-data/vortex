// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};

use num_traits::ToPrimitive;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_panic};

use crate::DType;

/// Maximum precision for a Decimal128 type from Arrow
pub const DECIMAL128_MAX_PRECISION: u8 = 38;

/// Maximum precision for a Decimal256 type from Arrow
pub const DECIMAL256_MAX_PRECISION: u8 = 76;

/// Maximum scale for a Decimal128 type from Arrow
pub const DECIMAL128_MAX_SCALE: i8 = 38;

/// Maximum scale for a Decimal256 type from Arrow
pub const DECIMAL256_MAX_SCALE: i8 = 76;

const MAX_PRECISION: u8 = DECIMAL256_MAX_PRECISION;
const MAX_SCALE: i8 = DECIMAL256_MAX_SCALE;

/// Parameters that define the precision and scale of a decimal type.
///
/// Decimal types allow real numbers with a similar precision and scale to be represented exactly.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecimalDType {
    precision: u8,
    scale: i8,
}

impl DecimalDType {
    /// Fallible constructor for a `DecimalDType`.
    ///
    /// # Errors
    ///
    /// Returns an error if precision exceeds MAX_PRECISION or scale is outside [MIN_SCALE, MAX_SCALE].
    pub fn try_new(precision: u8, scale: i8) -> VortexResult<Self> {
        if precision == 0 {
            vortex_bail!(
                "decimal precision must be between 1 and {} (inclusive)",
                MAX_PRECISION
            );
        }
        if precision > MAX_PRECISION {
            vortex_bail!(
                "decimal precision {} exceeds MAX_PRECISION {}",
                precision,
                MAX_PRECISION
            );
        }

        if scale > MAX_SCALE {
            vortex_bail!("decimal scale {} exceeds MAX_SCALE {}", scale, MAX_SCALE);
        }

        if scale > 0 && scale as u8 > precision {
            vortex_bail!(
                "decimal scale {} is greater than precision {}",
                scale,
                precision
            );
        }

        Ok(Self { precision, scale })
    }

    /// Unchecked constructor for a `DecimalDType`.
    ///
    /// # Panics
    ///
    /// Attempting to build a new instance with invalid precision or scale values will panic.
    /// Prefer using `try_new` for fallible construction.
    pub fn new(precision: u8, scale: i8) -> Self {
        Self::try_new(precision, scale)
            .unwrap_or_else(|e| vortex_panic!(e, "Failed to create DecimalDType"))
    }

    /// The precision is the number of significant figures that the decimal tracks.
    pub fn precision(&self) -> u8 {
        self.precision
    }

    /// The scale is the maximum number of digits relative to the decimal point.
    ///
    /// Positive scale means digits after decimal point, negative scale means number of
    /// zeros before the decimal point.
    pub fn scale(&self) -> i8 {
        self.scale
    }

    /// Return the max number of bits required to fit a decimal with `precision` in.
    pub fn required_bit_width(&self) -> usize {
        (self.precision as f32 * 10.0f32.log(2.0))
            .ceil()
            .to_usize()
            .vortex_expect("too many bits required")
    }
}

impl Display for DecimalDType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "decimal({},{})", self.precision, self.scale)
    }
}

impl TryFrom<&DType> for DecimalDType {
    type Error = VortexError;

    fn try_from(value: &DType) -> Result<Self, Self::Error> {
        match value {
            DType::Decimal(dt, _) => Ok(*dt),
            _ => vortex_bail!("Cannot convert DType {value} into DecimalType"),
        }
    }
}

impl TryFrom<DType> for DecimalDType {
    type Error = VortexError;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Nullability};

    #[test]
    fn test_decimal_valid_construction() {
        let decimal = DecimalDType::try_new(10, 2).unwrap();
        assert_eq!(decimal.precision(), 10);
        assert_eq!(decimal.scale(), 2);
    }

    #[test]
    fn test_decimal_new_deprecated() {
        let decimal = DecimalDType::try_new(10, 2).unwrap();
        assert_eq!(decimal.precision(), 10);
        assert_eq!(decimal.scale(), 2);
    }

    #[test]
    fn test_decimal_max_precision() {
        let decimal = DecimalDType::try_new(MAX_PRECISION, 0).unwrap();
        assert_eq!(decimal.precision(), MAX_PRECISION);
    }

    #[test]
    fn test_decimal_max_scale() {
        // MAX_SCALE only works when precision >= scale
        let decimal = DecimalDType::try_new(MAX_PRECISION, MAX_SCALE).unwrap();
        assert_eq!(decimal.scale(), MAX_SCALE);
        assert_eq!(decimal.precision(), MAX_PRECISION);
    }

    #[test]
    fn test_decimal_negative_scale() {
        // Negative scale is valid - represents zeros before decimal point
        let decimal = DecimalDType::try_new(10, -5).unwrap();
        assert_eq!(decimal.scale(), -5);

        // Negative scale doesn't need to be less than precision
        let decimal2 = DecimalDType::try_new(5, -10).unwrap();
        assert_eq!(decimal2.scale(), -10);
        assert_eq!(decimal2.precision(), 5);
    }

    #[test]
    fn test_decimal_zero_precision() {
        // Zero precision is not allowed
        let result = DecimalDType::try_new(0, 0);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must be between 1 and")
        );
    }

    #[test]
    fn test_decimal_scale_greater_than_precision() {
        // When scale is positive, it must be <= precision
        let result = DecimalDType::try_new(5, 6);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("scale 6 is greater than precision 5")
        );

        // Edge case: scale == precision should work
        let decimal = DecimalDType::try_new(5, 5).unwrap();
        assert_eq!(decimal.precision(), 5);
        assert_eq!(decimal.scale(), 5);
    }

    #[test]
    fn test_decimal_exceeds_max_precision() {
        let result = DecimalDType::try_new(MAX_PRECISION + 1, 0);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exceeds MAX_PRECISION")
        );
    }

    #[test]
    fn test_decimal_exceeds_max_scale() {
        let result = DecimalDType::try_new(MAX_PRECISION, MAX_SCALE + 1);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exceeds MAX_SCALE")
        );
    }

    #[test]
    fn test_decimal_precision_scale_edge_cases() {
        // Precision 1 with scale 0 (single digit integer)
        let decimal = DecimalDType::try_new(1, 0).unwrap();
        assert_eq!(decimal.precision(), 1);
        assert_eq!(decimal.scale(), 0);

        // Precision 1 with scale 1 (0.X format)
        let decimal = DecimalDType::try_new(1, 1).unwrap();
        assert_eq!(decimal.precision(), 1);
        assert_eq!(decimal.scale(), 1);

        // Scale 0 is valid for any precision
        let decimal = DecimalDType::try_new(10, 0).unwrap();
        assert_eq!(decimal.precision(), 10);
        assert_eq!(decimal.scale(), 0);

        // Negative scale with small precision is valid
        let decimal = DecimalDType::try_new(1, -5).unwrap();
        assert_eq!(decimal.precision(), 1);
        assert_eq!(decimal.scale(), -5);
    }

    #[test]
    fn test_decimal128_boundaries() {
        let decimal = DecimalDType::new(DECIMAL128_MAX_PRECISION, DECIMAL128_MAX_SCALE);
        assert_eq!(decimal.precision(), 38);
        assert_eq!(decimal.scale(), 38);
    }

    #[test]
    fn test_decimal256_boundaries() {
        let decimal = DecimalDType::new(DECIMAL256_MAX_PRECISION, DECIMAL256_MAX_SCALE);
        assert_eq!(decimal.precision(), 76);
        assert_eq!(decimal.scale(), 76);
    }

    #[test]
    fn test_required_bit_width() {
        // Test common decimal precisions
        let decimal_9 = DecimalDType::new(9, 2);
        assert!(decimal_9.required_bit_width() <= 32); // fits in 32 bits

        let decimal_18 = DecimalDType::new(18, 4);
        assert!(decimal_18.required_bit_width() <= 64); // fits in 64 bits

        let decimal_38 = DecimalDType::new(38, 10);
        assert!(decimal_38.required_bit_width() <= 128); // fits in 128 bits

        let decimal_76 = DecimalDType::new(76, 20);
        assert!(decimal_76.required_bit_width() <= 256); // fits in 256 bits
    }

    #[test]
    fn test_required_bit_width_edge_cases() {
        // Precision 1 should require at least 4 bits (to store 0-9)
        let decimal_1 = DecimalDType::new(1, 0);
        assert!(decimal_1.required_bit_width() >= 4);

        // Maximum precision
        let decimal_max = DecimalDType::new(MAX_PRECISION, 0);
        let bits = decimal_max.required_bit_width();
        assert!(bits > 0 && bits <= 256);
    }

    #[test]
    fn test_try_from_dtype() {
        let decimal = DecimalDType::try_new(10, 2).unwrap();
        let dtype = DType::Decimal(decimal, Nullability::NonNullable);

        let converted = DecimalDType::try_from(&dtype).unwrap();
        assert_eq!(converted.precision(), 10);
        assert_eq!(converted.scale(), 2);
    }

    #[test]
    fn test_try_from_dtype_owned() {
        let decimal = DecimalDType::try_new(10, 2).unwrap();
        let dtype = DType::Decimal(decimal, Nullability::Nullable);

        let converted = DecimalDType::try_from(dtype).unwrap();
        assert_eq!(converted.precision(), 10);
        assert_eq!(converted.scale(), 2);
    }

    #[test]
    fn test_try_from_dtype_wrong_type() {
        let dtype = DType::Bool(Nullability::NonNullable);
        let result = DecimalDType::try_from(&dtype);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot convert DType")
        );
    }
}
