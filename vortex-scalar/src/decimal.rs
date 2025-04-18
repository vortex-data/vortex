use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};

use arrow_buffer::i256;
use vortex_dtype::{DType, DecimalDType, PType, match_each_native_ptype};
use vortex_error::{VortexError, VortexExpect, VortexResult};

use crate::{Scalar, ScalarValue};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd)]
pub enum DecimalValue {
    I128(i128),
    I256(i256),
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

    pub fn decimal_value(&self) -> &Option<DecimalValue> {
        &self.value
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        let decimal_type = DecimalDType::try_from(dtype)?;
        let value = self
            .decimal_value()
            .clone()
            .vortex_expect("nullness handled in Scalar::cast");
        // How does this work? I don't think we can safely cast, unless we're casting the nullability
        // from non-null to null when we have a non-null value.
        // Ok(match_each_native_ptype!(ptype, |$Q| {
        //     Scalar::primitive(
        //         pvalue
        //             .as_primitive::<$Q>()
        //             .map_err(|err| vortex_err!("Can't cast {} scalar {} to {} (cause: {})", self.ptype, pvalue, dtype, err))?,
        //         dtype.nullability()
        //     )
        // }))
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
