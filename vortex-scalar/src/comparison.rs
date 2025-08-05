// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Standardized comparison implementations for scalar types.
//! 
//! This module provides consistent comparison behavior across all scalar types.
//! The standard pattern is:
//! 1. Check dtype compatibility (ignoring nullability)
//! 2. If dtypes are incompatible, return None
//! 3. Otherwise, compare the values

use std::cmp::Ordering;

/// Standard partial comparison implementation for scalar types.
/// 
/// This macro ensures consistent comparison behavior:
/// - Returns None if dtypes are incompatible (ignoring nullability)
/// - Otherwise compares the values
#[macro_export]
macro_rules! impl_scalar_partial_ord {
    ($self:expr, $other:expr, $dtype_field:ident, $value_field:ident) => {{
        if !$self.$dtype_field.eq_ignore_nullability($other.$dtype_field) {
            None
        } else {
            $self.$value_field.partial_cmp(&$other.$value_field)
        }
    }};
    
    // Variant for types where we need to call a method to get the dtype
    ($self:expr, $other:expr, dtype_method: $dtype_method:ident, $value_field:ident) => {{
        if !$self.$dtype_method().eq_ignore_nullability($other.$dtype_method()) {
            None
        } else {
            $self.$value_field.partial_cmp(&$other.$value_field)
        }
    }};
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};
    
    use crate::Scalar;

    #[test]
    fn test_comparison_dtype_check() {
        // Test that all scalar types check dtype compatibility
        
        // Bool scalars should check dtype
        let bool1 = Scalar::bool(true, Nullability::Nullable);
        let bool2 = Scalar::bool(true, Nullability::NonNullable);
        // They have different nullability but same base type, so should compare
        assert!(bool1.partial_cmp(&bool2).is_some());
        
        // Different types should not compare
        let bool_scalar = Scalar::bool(true, Nullability::NonNullable);
        let int_scalar = Scalar::primitive(1i32, Nullability::NonNullable);
        assert_eq!(bool_scalar.partial_cmp(&int_scalar), None);
    }
}