// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_bail, vortex_err};

use crate::{InnerScalarValue, Scalar, ScalarValue};

/// A scalar value representing a boolean.
///
/// This type provides a view into a boolean scalar value, which can be either
/// true, false, or null.
#[derive(Debug, Hash, Eq)]
pub struct BoolScalar<'a> {
    dtype: &'a DType,
    value: Option<bool>,
}

impl Display for BoolScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.value {
            None => write!(f, "null"),
            Some(v) => write!(f, "{v}"),
        }
    }
}

impl PartialEq for BoolScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.value == other.value
    }
}

impl PartialOrd for BoolScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BoolScalar<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl<'a> BoolScalar<'a> {
    /// Returns the data type of this boolean scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the boolean value, or None if null.
    pub fn value(&self) -> Option<bool> {
        self.value
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if !matches!(dtype, DType::Bool(..)) {
            vortex_bail!("Cannot cast bool to {}: unsupported conversion", dtype)
        }
        Ok(Scalar::bool(
            self.value.vortex_expect("nullness handled in Scalar::cast"),
            dtype.nullability(),
        ))
    }

    /// Returns a new boolean scalar with the inverted value.
    ///
    /// Null values remain null.
    pub fn invert(self) -> BoolScalar<'a> {
        BoolScalar {
            dtype: self.dtype,
            value: self.value.map(|v| !v),
        }
    }

    /// Converts this boolean scalar into a general scalar.
    pub fn into_scalar(self) -> Scalar {
        Scalar::new(
            self.dtype.clone(),
            self.value
                .map(|x| ScalarValue(InnerScalarValue::Bool(x)))
                .unwrap_or_else(|| ScalarValue(InnerScalarValue::Null)),
        )
    }
}

impl Scalar {
    /// Creates a new boolean scalar with the given value and nullability.
    pub fn bool(value: bool, nullability: Nullability) -> Self {
        Self::new(
            DType::Bool(nullability),
            ScalarValue(InnerScalarValue::Bool(value)),
        )
    }
}

impl<'a> TryFrom<&'a Scalar> for BoolScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        if !matches!(value.dtype(), DType::Bool(_)) {
            vortex_bail!("Expected bool scalar, found {}", value.dtype())
        }
        Ok(Self {
            dtype: value.dtype(),
            value: value.value.as_bool()?,
        })
    }
}

impl TryFrom<&Scalar> for bool {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> VortexResult<Self> {
        <Option<bool>>::try_from(value)?
            .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
    }
}

impl TryFrom<&Scalar> for Option<bool> {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> VortexResult<Self> {
        Ok(BoolScalar::try_from(value)?.value())
    }
}

impl TryFrom<Scalar> for bool {
    type Error = VortexError;

    fn try_from(value: Scalar) -> VortexResult<Self> {
        Self::try_from(&value)
    }
}

impl TryFrom<Scalar> for Option<bool> {
    type Error = VortexError;

    fn try_from(value: Scalar) -> VortexResult<Self> {
        Self::try_from(&value)
    }
}

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        Self::new(DType::Bool(NonNullable), value.into())
    }
}

impl From<bool> for ScalarValue {
    fn from(value: bool) -> Self {
        ScalarValue(InnerScalarValue::Bool(value))
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::Nullability::*;

    use super::*;

    #[test]
    fn into_from() {
        let scalar: Scalar = false.into();
        assert!(!bool::try_from(&scalar).unwrap());
    }

    #[test]
    fn equality() {
        assert_eq!(&Scalar::bool(true, Nullable), &Scalar::bool(true, Nullable));
        // Equality ignores nullability
        assert_eq!(
            &Scalar::bool(true, Nullable),
            &Scalar::bool(true, NonNullable)
        );
    }
}
