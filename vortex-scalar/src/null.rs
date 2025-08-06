// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Null scalar conversion implementations.
//!
//! This module provides conversions between Scalar values and the unit type `()`.
//!
//! # Conversion Behavior
//!
//! The `TryFrom<&Scalar>` implementation for `()` succeeds in two cases:
//!
//! 1. **Pure null scalars**: Scalars with `DType::Null` always convert successfully to `()`.
//!
//! 2. **Typed null scalars**: Scalars of any type (primitive, string, binary, etc.) that
//!    have a null value also convert successfully to `()`. This includes scalars created
//!    with methods like `Scalar::null_typed::<T>()`.
//!
//! This behavior means that typed null scalars (e.g., a null i32 or null string) are
//! treated equivalently to pure null scalars for the purpose of unit type conversion.
//!
//! # Examples
//!
//! ```ignore
//! use vortex_scalar::Scalar;
//! use vortex_dtype::DType;
//!
//! // Pure null scalar converts to ()
//! let null_scalar = Scalar::null(DType::Null);
//! let unit: () = null_scalar.try_into().unwrap();
//!
//! // Typed null scalar also converts to ()
//! let null_int = Scalar::null_typed::<i32>();
//! let unit: () = null_int.try_into().unwrap();
//!
//! // Non-null scalar fails conversion
//! let int_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
//! let result = <()>::try_from(&int_scalar);
//! assert!(result.is_err());
//! ```

use vortex_error::VortexError;

use crate::Scalar;

impl TryFrom<&Scalar> for () {
    type Error = VortexError;

    fn try_from(scalar: &Scalar) -> Result<Self, Self::Error> {
        scalar.value().as_null()
    }
}

impl TryFrom<Scalar> for () {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        <()>::try_from(&scalar)
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};

    use super::*;

    #[test]
    fn test_null_scalar_try_from_ref() {
        let null_scalar = Scalar::null(DType::Null);

        let result = <()>::try_from(&null_scalar);
        assert!(result.is_ok());
    }

    #[test]
    fn test_null_scalar_try_from_owned() {
        let null_scalar = Scalar::null(DType::Null);

        let result = <()>::try_from(null_scalar);
        assert!(result.is_ok());
    }

    #[test]
    fn test_non_null_scalar_fails_ref() {
        let int_scalar = Scalar::primitive(42i32, Nullability::NonNullable);

        let result = <()>::try_from(&int_scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_null_scalar_fails_owned() {
        let int_scalar = Scalar::primitive(42i32, Nullability::NonNullable);

        let result = <()>::try_from(int_scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_nullable_primitive_with_null_value() {
        let null_int = Scalar::null_typed::<i32>();

        // NOTE: Unexpected behavior - TryFrom succeeds for typed null scalars
        let result = <()>::try_from(&null_int);
        assert!(result.is_ok());
    }

    #[test]
    fn test_null_string() {
        let null_string = Scalar::null_typed::<String>();

        // NOTE: Unexpected behavior - TryFrom succeeds for typed null scalars
        let result = <()>::try_from(&null_string);
        assert!(result.is_ok());
    }

    #[test]
    fn test_null_bool() {
        let null_bool = Scalar::null_typed::<bool>();

        // NOTE: Unexpected behavior - TryFrom succeeds for typed null scalars
        let result = <()>::try_from(&null_bool);
        assert!(result.is_ok());
    }

    #[test]
    fn test_null_list() {
        use std::sync::Arc;

        use vortex_dtype::PType;

        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::Nullable));
        let null_list = Scalar::list_empty(element_dtype, Nullability::Nullable);

        // NOTE: Unexpected behavior - TryFrom succeeds for typed null scalars
        let result = <()>::try_from(&null_list);
        assert!(result.is_ok());
    }

    #[test]
    fn test_null_struct() {
        use vortex_dtype::{FieldDType, StructFields};

        let struct_dtype = DType::Struct(
            StructFields::from_iter([("field1", FieldDType::from(DType::Null))]),
            Nullability::Nullable,
        );

        let null_struct = Scalar::struct_(struct_dtype, vec![Scalar::null(DType::Null)]);

        // This should fail because it's a struct, not a pure null type
        let result = <()>::try_from(&null_struct);
        assert!(result.is_err());
    }

    #[test]
    fn test_null_binary() {
        let null_binary = Scalar::null(DType::Binary(Nullability::Nullable));

        // NOTE: Unexpected behavior - TryFrom succeeds for typed null scalars
        let result = <()>::try_from(&null_binary);
        assert!(result.is_ok());
    }
}
