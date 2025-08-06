// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests documenting current behavior that may be inconsistent or problematic.

#[cfg(test)]
mod tests {

    use vortex_dtype::Nullability;

    use crate::{BoolScalar, PrimitiveScalar, Scalar};

    // Demonstrates inconsistent null comparison behavior
    #[test]
    fn test_null_comparison_inconsistency() {
        // Test with primitive scalars
        let null_i32 = Scalar::null_typed::<i32>();
        let null_i64 = Scalar::null_typed::<i64>();

        let prim_i32 = PrimitiveScalar::try_from(&null_i32).unwrap();
        let prim_i64 = PrimitiveScalar::try_from(&null_i64).unwrap();

        // Primitive scalars check dtype compatibility first
        assert_eq!(prim_i32.partial_cmp(&prim_i64), None); // Different types => None

        // Test with boolean scalars with different nullability
        // We can't create nullable and non-nullable null bools, so test with non-null values
        let bool_nullable = Scalar::bool(true, Nullability::Nullable);
        let bool_non_nullable = Scalar::bool(true, Nullability::NonNullable);

        let bool1 = BoolScalar::try_from(&bool_nullable).unwrap();
        let bool2 = BoolScalar::try_from(&bool_non_nullable).unwrap();

        // Bool scalars should now check dtype compatibility but ignore nullability
        // So they should still compare as they have the same base type
        assert!(bool1.partial_cmp(&bool2).is_some()); // Same base type, different nullability -> Some
    }

    // Test that demonstrates potential issues with typed null conversions
    #[test]
    fn test_typed_null_unit_conversion_surprising() {
        // This behavior is documented but potentially surprising
        let typed_null = Scalar::null_typed::<i32>();

        // A typed null (i32 null) successfully converts to unit type
        let unit_result = <()>::try_from(&typed_null);
        assert!(unit_result.is_ok()); // This might be unexpected!

        // But a non-null value correctly fails
        let non_null = Scalar::primitive(42i32, Nullability::NonNullable);
        let unit_result = <()>::try_from(&non_null);
        assert!(unit_result.is_err()); // Expected
    }

    // Demonstrates that equality checking doesn't always consider nullability
    #[test]
    fn test_nullability_in_equality() {
        let nullable = Scalar::primitive(42i32, Nullability::Nullable);
        let non_nullable = Scalar::primitive(42i32, Nullability::NonNullable);

        // These have different dtypes (different nullability)
        assert_ne!(nullable.dtype(), non_nullable.dtype());

        // But they compare as equal in value
        // This might be correct behavior but could be surprising
        assert_eq!(nullable, non_nullable);
    }
}
