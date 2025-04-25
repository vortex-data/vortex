use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};

use vortex_dtype::{DType, DecimalDType, Nullability};
use vortex_error::{VortexError, VortexResult};

use crate::scalarvalue::InnerScalarValue;
use crate::{Scalar, ScalarValue, i256};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd)]
pub enum DecimalValue {
    I128(i128),
    I256(i256),
}

impl Display for DecimalValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DecimalValue::I128(v128) => write!(f, "decimal128({})", v128),
            DecimalValue::I256(v256) => write!(f, "decimal256({})", v256),
        }
    }
}

impl Scalar {
    pub fn decimal(
        value: DecimalValue,
        decimal_type: DecimalDType,
        nullability: Nullability,
    ) -> Self {
        Self::new(
            DType::Decimal(decimal_type, nullability),
            ScalarValue(InnerScalarValue::Decimal(value)),
        )
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct DecimalScalar<'a> {
    dtype: &'a DType,
    decimal_type: DecimalDType,
    value: Option<DecimalValue>,
}

impl<'a> DecimalScalar<'a> {
    pub fn try_new(dtype: &'a DType, value: &ScalarValue) -> VortexResult<Self> {
        let decimal_type = DecimalDType::try_from(dtype)?;
        let value = value.as_decimal()?;

        Ok(Self {
            dtype,
            decimal_type,
            value,
        })
    }

    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    pub fn decimal_value(&self) -> &Option<DecimalValue> {
        &self.value
    }
}

impl<'a> TryFrom<&'a Scalar> for DecimalScalar<'a> {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> Result<Self, Self::Error> {
        DecimalScalar::try_new(&scalar.dtype, &scalar.value)
    }
}

impl Display for DecimalScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.value.as_ref() {
            Some(&dv) => {
                // Introduce some of the scale factors instead.
                match dv {
                    DecimalValue::I128(v) => write!(
                        f,
                        "decimal128({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I256(v) => write!(
                        f,
                        "decimal256({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                }
            }
            None => {
                write!(f, "null")
            }
        }
    }
}

impl PartialEq for DecimalScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.value == other.value
    }
}

impl Eq for DecimalScalar<'_> {}

/// Ord is not implemented since it's undefined for different PTypes
impl PartialOrd for DecimalScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self.dtype.eq_ignore_nullability(other.dtype) {
            return None;
        }
        self.value.partial_cmp(&other.value)
    }
}
