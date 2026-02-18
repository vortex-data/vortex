// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`BoolScalar`] typed view implementation.

use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A scalar value representing a boolean.
///
/// This type provides a view into a boolean scalar value, which can be either
/// true, false, or null.
#[derive(Debug, Clone, Hash, Eq)]
pub struct BoolScalar<'a> {
    /// The data type of this scalar.
    dtype: &'a DType,
    /// The boolean value, or [`None`] if null.
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
    /// Attempts to create a new [`BoolScalar`] from a [`DType`] and optional [`ScalarValue`].
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not a [`DType::Bool`].
    pub fn try_new(dtype: &'a DType, value: Option<&ScalarValue>) -> VortexResult<Self> {
        if !matches!(dtype, DType::Bool(_)) {
            vortex_bail!("Expected bool scalar, found {}", dtype)
        }
        Ok(Self {
            dtype,
            value: value.map(|v| v.as_bool()),
        })
    }

    /// Returns the data type of this boolean scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the boolean value, or None if null.
    pub fn value(&self) -> Option<bool> {
        self.value
    }

    /// Casts this scalar to the given `dtype`.
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if !matches!(dtype, DType::Bool(..)) {
            vortex_bail!(
                "Cannot cast bool to {dtype}: boolean scalars can only be cast to boolean types with different nullability"
            )
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
        // SAFETY: `BoolScalar` is already a valid `Scalar`.
        unsafe { Scalar::new_unchecked(self.dtype.clone(), self.value.map(ScalarValue::Bool)) }
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::Nullability::*;

    use super::*;

    #[test]
    fn into_from() {
        let scalar: Scalar = Some(false).into();
        assert!(!bool::try_from(&scalar).unwrap());
    }

    #[test]
    fn test_bool_scalar_ordering() {
        let false_scalar = Scalar::bool(false, NonNullable);
        let true_scalar = Scalar::bool(true, NonNullable);
        let null_scalar = Scalar::null(DType::Bool(Nullable));

        let false_bool = false_scalar.as_bool();
        let true_bool = true_scalar.as_bool();
        let null_bool = null_scalar.as_bool();

        // false < true
        assert!(false_bool < true_bool);
        assert!(true_bool > false_bool);

        // None < Some(false) < Some(true)
        assert!(null_bool < false_bool);
        assert!(null_bool < true_bool);
        assert!(false_bool > null_bool);
        assert!(true_bool > null_bool);
    }

    #[test]
    fn test_bool_invert() {
        let true_scalar = Scalar::bool(true, NonNullable);
        let false_scalar = Scalar::bool(false, NonNullable);
        let null_scalar = Scalar::null(DType::Bool(Nullable));

        let true_bool = true_scalar.as_bool();
        let false_bool = false_scalar.as_bool();
        let null_bool = null_scalar.as_bool();

        // Invert true -> false
        let inverted_true = true_bool.invert();
        assert_eq!(inverted_true.value(), Some(false));

        // Invert false -> true
        let inverted_false = false_bool.invert();
        assert_eq!(inverted_false.value(), Some(true));

        // Invert null -> null
        let inverted_null = null_bool.invert();
        assert_eq!(inverted_null.value(), None);
    }

    #[test]
    fn test_bool_into_scalar() {
        let bool_scalar = BoolScalar {
            dtype: &DType::Bool(NonNullable),
            value: Some(true),
        };

        let scalar = bool_scalar.into_scalar();
        assert_eq!(scalar.dtype(), &DType::Bool(NonNullable));
        assert!(bool::try_from(&scalar).unwrap());

        // Test null case
        let null_bool_scalar = BoolScalar {
            dtype: &DType::Bool(Nullable),
            value: None,
        };

        let null_scalar = null_bool_scalar.into_scalar();
        assert!(null_scalar.is_null());
    }

    #[test]
    fn test_bool_cast_to_bool() {
        let bool_scalar = Scalar::bool(true, NonNullable);
        let bool = bool_scalar.as_bool();

        // Cast to nullable bool
        let result = bool.cast(&DType::Bool(Nullable)).unwrap();
        assert_eq!(result.dtype(), &DType::Bool(Nullable));
        assert!(bool::try_from(&result).unwrap());

        // Cast to non-nullable bool
        let result = bool.cast(&DType::Bool(NonNullable)).unwrap();
        assert_eq!(result.dtype(), &DType::Bool(NonNullable));
        assert!(bool::try_from(&result).unwrap());
    }

    #[test]
    fn test_bool_cast_to_non_bool_fails() {
        use vortex_dtype::PType;

        let bool_scalar = Scalar::bool(true, NonNullable);
        let bool = bool_scalar.as_bool();

        let result = bool.cast(&DType::Primitive(PType::I32, NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_non_bool_scalar() {
        let int_scalar = Scalar::primitive(42i32, NonNullable);
        assert!(int_scalar.as_bool_opt().is_none());
    }

    #[test]
    fn test_try_from_null_scalar() {
        let null_scalar = Scalar::null(DType::Bool(Nullable));

        // Try to extract bool from null - should fail
        let result: Result<bool, _> = (&null_scalar).try_into();
        assert!(result.is_err());

        // Extract Option<bool> from null - should succeed with None
        let result: Result<Option<bool>, _> = (&null_scalar).try_into();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_try_from_owned_scalar() {
        // Test owned Scalar -> bool
        let scalar = Scalar::bool(true, NonNullable);
        let result: Result<bool, _> = scalar.try_into();
        assert!(result.is_ok());
        assert!(result.unwrap());

        // Test owned Scalar -> Option<bool>
        let scalar = Scalar::bool(false, Nullable);
        let result: Result<Option<bool>, _> = scalar.try_into();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(false));

        // Test owned null Scalar -> Option<bool>
        let null_scalar = Scalar::null(DType::Bool(Nullable));
        let result: Result<Option<bool>, _> = null_scalar.try_into();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_scalar_value_from_bool() {
        let value: ScalarValue = true.into();
        let scalar = Scalar::new(DType::Bool(NonNullable), Some(value));
        assert!(bool::try_from(&scalar).unwrap());

        let value: ScalarValue = false.into();
        let scalar = Scalar::new(DType::Bool(NonNullable), Some(value));
        assert!(!bool::try_from(&scalar).unwrap());
    }

    #[test]
    fn test_bool_partial_eq_different_values() {
        let true_scalar = Scalar::bool(true, NonNullable);
        let false_scalar = Scalar::bool(false, NonNullable);

        let true_bool = true_scalar.as_bool();
        let false_bool = false_scalar.as_bool();

        assert_ne!(true_bool, false_bool);
    }

    #[test]
    fn test_bool_partial_eq_null() {
        let null_scalar1 = Scalar::null(DType::Bool(Nullable));
        let null_scalar2 = Scalar::null(DType::Bool(Nullable));
        let non_null_scalar = Scalar::bool(true, Nullable);

        let null_bool1 = null_scalar1.as_bool();
        let null_bool2 = null_scalar2.as_bool();
        let non_null_bool = non_null_scalar.as_bool();

        // Two nulls are equal
        assert_eq!(null_bool1, null_bool2);

        // Null != non-null
        assert_ne!(null_bool1, non_null_bool);
    }

    #[test]
    fn test_bool_value_accessor() {
        let true_scalar = Scalar::bool(true, NonNullable);
        let false_scalar = Scalar::bool(false, NonNullable);
        let null_scalar = Scalar::null(DType::Bool(Nullable));

        let true_bool = true_scalar.as_bool();
        let false_bool = false_scalar.as_bool();
        let null_bool = null_scalar.as_bool();

        assert_eq!(true_bool.value(), Some(true));
        assert_eq!(false_bool.value(), Some(false));
        assert_eq!(null_bool.value(), None);
    }

    #[test]
    fn test_bool_dtype_accessor() {
        let nullable_scalar = Scalar::bool(true, Nullable);
        let non_nullable_scalar = Scalar::bool(false, NonNullable);

        let nullable_bool = nullable_scalar.as_bool();
        let non_nullable_bool = non_nullable_scalar.as_bool();

        assert_eq!(nullable_bool.dtype(), &DType::Bool(Nullable));
        assert_eq!(non_nullable_bool.dtype(), &DType::Bool(NonNullable));
    }

    #[test]
    fn test_bool_partial_cmp() {
        let false_scalar = Scalar::bool(false, NonNullable);
        let true_scalar = Scalar::bool(true, NonNullable);

        let false_bool = false_scalar.as_bool();
        let true_bool = true_scalar.as_bool();

        assert_eq!(false_bool.partial_cmp(&false_bool), Some(Ordering::Equal));
        assert_eq!(false_bool.partial_cmp(&true_bool), Some(Ordering::Less));
        assert_eq!(true_bool.partial_cmp(&false_bool), Some(Ordering::Greater));
    }
}
