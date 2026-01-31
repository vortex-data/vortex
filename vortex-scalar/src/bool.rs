// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex_dtype::DType;
use vortex_dtype::Nullability;

use crate::Scalar;
use crate::ScalarValue;

/// A scalar value representing a boolean.
///
/// This type provides a view into a boolean scalar value, which can be either
/// true, false, or null.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BoolScalar<'a> {
    pub(super) nullability: Nullability,
    pub(super) value: Option<bool>,
    // All other scalars carry a lifetime, so we do the same here for consistency.
    pub(super) _marker: PhantomData<&'a ()>,
}

impl<'a> BoolScalar<'a> {
    /// Returns the boolean value, or None if null.
    pub fn value(&self) -> Option<bool> {
        self.value
    }
    //
    // pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
    //     if !matches!(dtype, DType::Bool(..)) {
    //         vortex_bail!(
    //             "Cannot cast bool to {dtype}: boolean scalars can only be cast to boolean types with different nullability"
    //         )
    //     }
    //     Ok(Scalar::bool(
    //         self.value.vortex_expect("nullness handled in Scalar::cast"),
    //         dtype.nullability(),
    //     ))
    // }
    //
    // /// Returns a new boolean scalar with the inverted value.
    // ///
    // /// Null values remain null.
    // pub fn invert(self) -> BoolScalar<'a> {
    //     BoolScalar {
    //         dtype: self.dtype,
    //         value: self.value.map(|v| !v),
    //     }
    // }
    //
    // /// Converts this boolean scalar into a general scalar.
    // pub fn into_scalar(self) -> Scalar {
    //     Scalar::new(
    //         self.dtype.clone(),
    //         self.value
    //             .map(|x| ScalarValue(InnerScalarValue::Bool(x)))
    //             .unwrap_or_else(|| ScalarValue(InnerScalarValue::Null)),
    //     )
    // }
}

impl Scalar {
    /// Creates a new boolean scalar with the given value and nullability.
    pub fn bool(value: bool, nullability: Nullability) -> Self {
        unsafe { Scalar::new_unchecked(DType::Bool(nullability), ScalarValue::Bool(value)) }
    }
}

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        Self::bool(value, Nullability::NonNullable)
    }
}

impl From<bool> for ScalarValue {
    fn from(value: bool) -> Self {
        ScalarValue::Bool(value)
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::Nullability::*;

    use super::*;

    #[test]
    fn equality() {
        assert_eq!(&Scalar::bool(true, Nullable), &Scalar::bool(true, Nullable));
        // Equality ignores nullability
        assert_eq!(
            &Scalar::bool(true, Nullable),
            &Scalar::bool(true, NonNullable)
        );
    }

    #[test]
    fn test_bool_scalar_ordering() {
        let false_scalar = Scalar::bool(false, NonNullable);
        let true_scalar = Scalar::bool(true, NonNullable);
        let null_scalar = Scalar::null(DType::Bool(Nullable));

        let false_bool = BoolScalar::try_from(&false_scalar).unwrap();
        let true_bool = BoolScalar::try_from(&true_scalar).unwrap();
        let null_bool = BoolScalar::try_from(&null_scalar).unwrap();

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

        let true_bool = BoolScalar::try_from(&true_scalar).unwrap();
        let false_bool = BoolScalar::try_from(&false_scalar).unwrap();
        let null_bool = BoolScalar::try_from(&null_scalar).unwrap();

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
        let bool = BoolScalar::try_from(&bool_scalar).unwrap();

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
        let bool = BoolScalar::try_from(&bool_scalar).unwrap();

        let result = bool.cast(&DType::Primitive(PType::I32, NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_non_bool_scalar() {
        let int_scalar = Scalar::primitive(42i32, NonNullable);
        let result = BoolScalar::try_from(&int_scalar);
        assert!(result.is_err());
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
        let scalar = Scalar::new(DType::Bool(NonNullable), value);
        assert!(bool::try_from(&scalar).unwrap());

        let value: ScalarValue = false.into();
        let scalar = Scalar::new(DType::Bool(NonNullable), value);
        assert!(!bool::try_from(&scalar).unwrap());
    }

    #[test]
    fn test_bool_partial_eq_different_values() {
        let true_scalar = Scalar::bool(true, NonNullable);
        let false_scalar = Scalar::bool(false, NonNullable);

        let true_bool = BoolScalar::try_from(&true_scalar).unwrap();
        let false_bool = BoolScalar::try_from(&false_scalar).unwrap();

        assert_ne!(true_bool, false_bool);
    }

    #[test]
    fn test_bool_partial_eq_null() {
        let null_scalar1 = Scalar::null(DType::Bool(Nullable));
        let null_scalar2 = Scalar::null(DType::Bool(Nullable));
        let non_null_scalar = Scalar::bool(true, Nullable);

        let null_bool1 = BoolScalar::try_from(&null_scalar1).unwrap();
        let null_bool2 = BoolScalar::try_from(&null_scalar2).unwrap();
        let non_null_bool = BoolScalar::try_from(&non_null_scalar).unwrap();

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

        let true_bool = BoolScalar::try_from(&true_scalar).unwrap();
        let false_bool = BoolScalar::try_from(&false_scalar).unwrap();
        let null_bool = BoolScalar::try_from(&null_scalar).unwrap();

        assert_eq!(true_bool.value(), Some(true));
        assert_eq!(false_bool.value(), Some(false));
        assert_eq!(null_bool.value(), None);
    }

    #[test]
    fn test_bool_dtype_accessor() {
        let nullable_scalar = Scalar::bool(true, Nullable);
        let non_nullable_scalar = Scalar::bool(false, NonNullable);

        let nullable_bool = BoolScalar::try_from(&nullable_scalar).unwrap();
        let non_nullable_bool = BoolScalar::try_from(&non_nullable_scalar).unwrap();

        assert_eq!(nullable_bool.dtype(), &DType::Bool(Nullable));
        assert_eq!(non_nullable_bool.dtype(), &DType::Bool(NonNullable));
    }

    #[test]
    fn test_bool_partial_cmp() {
        let false_scalar = Scalar::bool(false, NonNullable);
        let true_scalar = Scalar::bool(true, NonNullable);

        let false_bool = BoolScalar::try_from(&false_scalar).unwrap();
        let true_bool = BoolScalar::try_from(&true_scalar).unwrap();

        assert_eq!(false_bool.partial_cmp(&false_bool), Some(Ordering::Equal));
        assert_eq!(false_bool.partial_cmp(&true_bool), Some(Ordering::Less));
        assert_eq!(true_bool.partial_cmp(&false_bool), Some(Ordering::Greater));
    }
}
