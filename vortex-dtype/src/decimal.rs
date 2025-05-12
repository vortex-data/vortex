use std::fmt::{Display, Formatter};

use num_traits::ToPrimitive;
use vortex_error::{VortexError, VortexExpect, vortex_bail};

use crate::DType;

const MAX_PRECISION: u8 = 76;
const MAX_SCALE: i8 = 76;

/// Maximum precision for a Decimal128 type from Arrow
pub const DECIMAL128_MAX_PRECISION: u8 = 38;

/// Maximum precision for a Decimal256 type from Arrow
pub const DECIMAL256_MAX_PRECISION: u8 = 76;

/// Maximum sacle for a Decimal128 type from Arrow
pub const DECIMAL128_MAX_SCALE: i8 = 38;

/// Maximum sacle for a Decimal256 type from Arrow
pub const DECIMAL256_MAX_SCALE: i8 = 76;

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
