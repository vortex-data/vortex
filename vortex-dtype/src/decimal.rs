use vortex_error::{VortexError, vortex_err};

use crate::DType;

const MAX_PRECISION: u8 = 76;
const MAX_SCALE: i8 = 76;

/// Maximum precision for a Decimal128 type from Arrow
pub const DECIMAL128_MAX_PRECISION: u8 = 38;

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
    /// Checked constructor for a `DecimalDType`.
    ///
    /// # Panics
    ///
    /// Attempting to build a new instance with invalid precision or scale values will panic.
    pub fn new(precision: u8, scale: i8) -> Self {
        assert!(
            precision <= MAX_PRECISION,
            "decimal precision {} exceeds MAX_PRECISION",
            precision
        );

        assert!(
            scale <= MAX_SCALE,
            "decimal scale {} exceeds MAX_SCALE",
            scale
        );

        Self { precision, scale }
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
}

impl TryFrom<&DType> for DecimalDType {
    type Error = VortexError;

    fn try_from(value: &DType) -> Result<Self, Self::Error> {
        match value {
            DType::Decimal(dt, _) => Ok(*dt),
            _ => Err(vortex_err!(
                "Cannot convert DType {} into DecimalType",
                value
            )),
        }
    }
}

impl TryFrom<DType> for DecimalDType {
    type Error = VortexError;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}
