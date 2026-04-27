// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comprehensive nullability testing for nested structures.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    #[rstest]
    fn null_can_cast_to_anything_nullable(
        #[values(
            DType::Null,
            DType::Bool(Nullability::Nullable),
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased()),
            DType::Extension(Timestamp::new(TimeUnit::Days, Nullability::Nullable).erased()),
        )]
        source_dtype: DType,
        #[values(
            DType::Null,
            DType::Bool(Nullability::Nullable),
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased()),
            DType::Extension(Timestamp::new(TimeUnit::Days, Nullability::Nullable).erased()),
        )]
        target_dtype: DType,
    ) {
        assert_eq!(
            Scalar::null(source_dtype)
                .cast(&target_dtype)
                .unwrap()
                .dtype(),
            &target_dtype
        );
    }

    #[test]
    fn test_list_cast_nullability_changes() {
        let list = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            Some(ScalarValue::Tuple(vec![Some(ScalarValue::Primitive(
                PValue::U16(6),
            ))])),
        );

        // Change element nullability from Nullable to NonNullable.
        let target_elem_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U16, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        assert_eq!(
            list.cast(&target_elem_nonnull).unwrap().dtype(),
            &target_elem_nonnull
        );

        // Change list nullability from Nullable to NonNullable.
        let target_list_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
            Nullability::NonNullable,
        );
        assert_eq!(
            list.cast(&target_list_nonnull).unwrap().dtype(),
            &target_list_nonnull
        );

        // Change both element and list nullability.
        let target_both_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U16, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        assert_eq!(
            list.cast(&target_both_nonnull).unwrap().dtype(),
            &target_both_nonnull
        );
    }

    #[test]
    fn test_list_cast_with_null_elements() {
        let list_with_null = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            Some(ScalarValue::Tuple(vec![
                Some(ScalarValue::Primitive(PValue::U16(6))),
                None,
                Some(ScalarValue::Primitive(PValue::U16(10))),
            ])),
        );

        // Cast to different element type with nullable elements - should succeed.
        let target_u8_nullable = DType::List(
            Arc::from(DType::Primitive(PType::U8, Nullability::Nullable)),
            Nullability::Nullable,
        );
        assert_eq!(
            list_with_null.cast(&target_u8_nullable).unwrap().dtype(),
            &target_u8_nullable
        );

        // Cast to non-nullable U16 elements should fail because we have null elements.
        let target_u16_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U16, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        let result = list_with_null.cast(&target_u16_nonnull);
        assert!(
            result.is_err(),
            "Expected cast to fail when casting list with null elements to non-nullable element type"
        );
    }

    #[test]
    fn test_scalar_nbytes_with_nulls() {
        // Test null string
        let null_utf8 = Scalar::null(DType::Utf8(Nullability::Nullable));
        assert_eq!(null_utf8.approx_nbytes(), 0);

        // Test null binary
        let null_binary = Scalar::null(DType::Binary(Nullability::Nullable));
        assert_eq!(null_binary.approx_nbytes(), 0);

        // Test struct with null fields
        let struct_with_null = Scalar::struct_(
            DType::struct_(
                [
                    ("a", DType::Primitive(PType::I32, Nullability::Nullable)),
                    ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ],
                Nullability::NonNullable,
            ),
            vec![
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::primitive(100i64, Nullability::NonNullable),
            ],
        );
        // Primitive null fields still count their byte width
        assert_eq!(struct_with_null.approx_nbytes(), 4 + 8);

        // Test list with null elements
        let list_with_null = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(1i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::primitive(3i32, Nullability::Nullable),
            ],
            Nullability::NonNullable,
        );
        // Primitive null elements still count their byte width
        assert_eq!(list_with_null.approx_nbytes(), 3 * 4); // 3 i32 values (including null)
    }

    #[test]
    fn test_scalar_is_valid_is_null() {
        let valid_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        assert!(valid_scalar.is_valid());
        assert!(!valid_scalar.is_null());

        let null_scalar = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        assert!(!null_scalar.is_valid());
        assert!(null_scalar.is_null());
    }

    #[test]
    fn test_scalar_into_nullable() {
        let non_nullable = Scalar::primitive(42i32, Nullability::NonNullable);
        assert_eq!(non_nullable.dtype().nullability(), Nullability::NonNullable);

        let nullable = non_nullable.into_nullable();
        assert_eq!(nullable.dtype().nullability(), Nullability::Nullable);
        assert_eq!(nullable.as_primitive().typed_value::<i32>(), Some(42));

        // Test with already nullable scalar
        let already_nullable = Scalar::primitive(42i32, Nullability::Nullable);
        let still_nullable = already_nullable.into_nullable();
        assert_eq!(still_nullable.dtype().nullability(), Nullability::Nullable);
    }

    #[test]
    fn test_list_cast_null_to_nonnull_error() {
        let list_with_null = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            Some(ScalarValue::Tuple(vec![
                Some(ScalarValue::Primitive(PValue::U16(6))),
                None,
            ])),
        );

        // Casting to non-nullable element type should fail.
        let target_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        assert!(list_with_null.cast(&target_nonnull).is_err());
    }

    #[test]
    fn test_fixed_size_list_null_elements() {
        // Create FixedSizeList with null elements.
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::Nullable));
        let fixed_list = Scalar::fixed_size_list(
            element_dtype,
            vec![
                Scalar::primitive(10i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::primitive(30i32, Nullability::Nullable),
            ],
            Nullability::NonNullable,
        );

        let list = fixed_list.as_list();
        assert_eq!(list.len(), 3);
        assert!(!list.is_null()); // The list itself is not null.

        // Check individual elements.
        assert!(!list.element(0).unwrap().is_null());
        assert!(list.element(1).unwrap().is_null());
        assert!(!list.element(2).unwrap().is_null());

        // Test casting with null elements.
        let target = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
            3,
            Nullability::NonNullable,
        );
        let casted = fixed_list.cast(&target).unwrap();
        assert!(casted.as_list().element(1).unwrap().is_null());
    }

    #[test]
    fn test_nested_nullability_propagation() {
        // Test null propagation through nested structures.
        // Create List of FixedSizeList[2] with nulls at different levels.

        let inner_dtype = Arc::new(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            2,
            Nullability::Nullable,
        ));

        // First inner list: non-null container with one null element.
        let inner_list1 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(1i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::Nullable,
        );

        // Second inner list: null container.
        let inner_list2 = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            2,
            Nullability::Nullable,
        ));

        // Third inner list: non-null container with non-null elements.
        let inner_list3 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(3i32, Nullability::Nullable),
                Scalar::primitive(4i32, Nullability::Nullable),
            ],
            Nullability::Nullable,
        );

        let outer_list = Scalar::list(
            inner_dtype,
            vec![inner_list1, inner_list2, inner_list3],
            Nullability::NonNullable,
        );

        let outer = outer_list.as_list();
        assert_eq!(outer.len(), 3);
        assert!(!outer.is_null());

        // First inner: not null, but has null element.
        let first = outer.element(0).unwrap();
        assert!(!first.is_null());
        assert!(first.as_list().element(1).unwrap().is_null());

        // Second inner: is null.
        let second = outer.element(1).unwrap();
        assert!(second.is_null());

        // Third inner: not null, no null elements.
        let third = outer.element(2).unwrap();
        assert!(!third.is_null());
        assert!(!third.as_list().element(0).unwrap().is_null());
        assert!(!third.as_list().element(1).unwrap().is_null());
    }

    #[test]
    fn test_struct_null_field_handling() {
        // Test struct with nullable fields including FixedSizeList.
        let struct_dtype = DType::struct_(
            [
                ("id", DType::Primitive(PType::I32, Nullability::NonNullable)),
                (
                    "values",
                    DType::FixedSizeList(
                        Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
                        2,
                        Nullability::Nullable,
                    ),
                ),
                ("name", DType::Utf8(Nullability::Nullable)),
            ],
            Nullability::NonNullable,
        );

        // Create struct with null FixedSizeList field.
        let struct_with_null_list = Scalar::struct_(
            struct_dtype.clone(),
            vec![
                Scalar::primitive(42i32, Nullability::NonNullable),
                Scalar::null(DType::FixedSizeList(
                    Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
                    2,
                    Nullability::Nullable,
                )),
                Scalar::utf8("test", Nullability::Nullable),
            ],
        );

        let struct_view = struct_with_null_list.as_struct();
        assert!(!struct_view.is_null());
        assert!(!struct_view.field("id").unwrap().is_null());
        assert!(struct_view.field("values").unwrap().is_null());
        assert!(!struct_view.field("name").unwrap().is_null());

        // Create struct with non-null FixedSizeList containing null elements.
        let fixed_list_with_nulls = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
            vec![
                Scalar::primitive(100i64, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
            ],
            Nullability::Nullable,
        );

        let struct_with_list_nulls = Scalar::struct_(
            struct_dtype,
            vec![
                Scalar::primitive(43i32, Nullability::NonNullable),
                fixed_list_with_nulls,
                Scalar::null(DType::Utf8(Nullability::Nullable)),
            ],
        );

        let struct_view2 = struct_with_list_nulls.as_struct();
        let values_field = struct_view2.field("values").unwrap();
        assert!(!values_field.is_null());
        assert!(values_field.as_list().element(1).unwrap().is_null());
        assert!(struct_view2.field("name").unwrap().is_null());
    }

    #[test]
    fn test_list_partial_null_elements() {
        // Test List with mixture of null and non-null elements.
        let element_dtype = Arc::new(DType::Primitive(PType::F32, Nullability::Nullable));
        let list_with_nulls = Scalar::list(
            element_dtype,
            vec![
                Scalar::primitive(1.5f32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::F32, Nullability::Nullable)),
                Scalar::primitive(2.5f32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::F32, Nullability::Nullable)),
                Scalar::primitive(3.5f32, Nullability::Nullable),
            ],
            Nullability::NonNullable,
        );

        let list = list_with_nulls.as_list();
        assert_eq!(list.len(), 5);
        assert!(!list.is_null());

        // Check null pattern.
        assert!(!list.element(0).unwrap().is_null());
        assert!(list.element(1).unwrap().is_null());
        assert!(!list.element(2).unwrap().is_null());
        assert!(list.element(3).unwrap().is_null());
        assert!(!list.element(4).unwrap().is_null());
    }

    #[test]
    fn test_cast_null_to_nonnull_errors() {
        // Test that casting null elements to non-nullable fails.

        // FixedSizeList with null elements can't cast to non-nullable elements.
        let fixed_list_with_nulls = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(1i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::NonNullable,
        );

        let target_nonnull_elems = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::NonNullable,
        );

        let result = fixed_list_with_nulls.cast(&target_nonnull_elems);
        assert!(result.is_err());

        // Null FixedSizeList can't cast to non-nullable container.
        let null_fixed_list = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            2,
            Nullability::Nullable,
        ));

        let target_nonnull_container = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            2,
            Nullability::NonNullable,
        );

        let result = null_fixed_list.cast(&target_nonnull_container);
        assert!(result.is_err());
    }

    #[test]
    fn test_nullable_to_nonnullable_valid_cast() {
        // Test that casting nullable types to non-nullable succeeds when no nulls present.
        let fixed_list_no_nulls = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(10i32, Nullability::Nullable),
                Scalar::primitive(20i32, Nullability::Nullable),
            ],
            Nullability::Nullable,
        );

        // Should succeed - no actual nulls.
        let target = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::NonNullable,
        );

        let casted = fixed_list_no_nulls.cast(&target).unwrap();
        assert_eq!(casted.dtype().nullability(), Nullability::NonNullable);
        match casted.dtype() {
            DType::FixedSizeList(elem_dtype, ..) => {
                assert_eq!(elem_dtype.nullability(), Nullability::NonNullable);
            }
            _ => panic!("Expected FixedSizeList"),
        }
    }

    #[test]
    fn test_is_null_vs_has_null_elements() {
        // Distinguish between a null container and a container with null elements.

        // Case 1: Null FixedSizeList.
        let null_list = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            3,
            Nullability::Nullable,
        ));
        assert!(null_list.is_null());

        // Case 2: Non-null FixedSizeList with all null elements.
        let list_all_nulls = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::Nullable,
        );
        assert!(!list_all_nulls.is_null()); // Container is not null.
        assert!(list_all_nulls.as_list().element(0).unwrap().is_null());
        assert!(list_all_nulls.as_list().element(1).unwrap().is_null());
        assert!(list_all_nulls.as_list().element(2).unwrap().is_null());

        // Case 3: Non-null FixedSizeList with some null elements.
        let list_some_nulls = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(1i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::primitive(3i32, Nullability::Nullable),
            ],
            Nullability::NonNullable,
        );
        assert!(!list_some_nulls.is_null());
        assert!(!list_some_nulls.as_list().element(0).unwrap().is_null());
        assert!(list_some_nulls.as_list().element(1).unwrap().is_null());
        assert!(!list_some_nulls.as_list().element(2).unwrap().is_null());
    }

    #[test]
    fn test_default_values_with_nullable_nested() {
        // Test default value creation for nullable nested structures.

        // Default value for nullable FixedSizeList should be null.
        let nullable_fixed_list_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            3,
            Nullability::Nullable,
        );
        let default_nullable_list = Scalar::default_value(&nullable_fixed_list_dtype);
        assert!(default_nullable_list.is_null());
        assert_eq!(default_nullable_list.dtype(), &nullable_fixed_list_dtype);

        // Default value for non-nullable FixedSizeList should be empty (if size=0) or have default elements.
        let nonnull_fixed_list_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::NonNullable,
        );
        let default_nonnull_list = Scalar::default_value(&nonnull_fixed_list_dtype);
        assert!(!default_nonnull_list.is_null());
        assert_eq!(default_nonnull_list.as_list().len(), 2);
        // Elements should be default values (0 for I32).
        assert_eq!(
            default_nonnull_list
                .as_list()
                .element(0)
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(0)
        );

        // Default value for struct with nullable FixedSizeList field.
        let struct_dtype = DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                (
                    "b",
                    DType::FixedSizeList(
                        Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
                        2,
                        Nullability::Nullable,
                    ),
                ),
            ],
            Nullability::NonNullable,
        );
        let default_struct = Scalar::default_value(&struct_dtype);
        let struct_view = default_struct.as_struct();
        assert_eq!(
            struct_view
                .field("a")
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(0)
        );
        assert!(struct_view.field("b").unwrap().is_null());
    }

    #[test]
    fn test_deeply_nested_nullability() {
        // Test nullability at 3+ levels of nesting.
        // FixedSizeList[2] of List of FixedSizeList[2] with nulls at various levels.

        let innermost_dtype = Arc::new(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            2,
            Nullability::Nullable,
        ));

        let middle_dtype = Arc::new(DType::List(
            Arc::clone(&innermost_dtype),
            Nullability::Nullable,
        ));

        // Create innermost FixedSizeLists with different null patterns.
        let inner1 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(1i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::Nullable,
        );

        let inner2 = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            2,
            Nullability::Nullable,
        ));

        let _inner3 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(3i32, Nullability::Nullable),
                Scalar::primitive(4i32, Nullability::Nullable),
            ],
            Nullability::Nullable,
        );

        // Create middle Lists.
        let middle1 = Scalar::list(
            Arc::clone(&innermost_dtype),
            vec![inner1, inner2],
            Nullability::Nullable,
        );

        let middle2 = Scalar::null(DType::List(innermost_dtype, Nullability::Nullable));

        // Create outer FixedSizeList.
        let outer = Scalar::fixed_size_list(
            middle_dtype,
            vec![middle1, middle2],
            Nullability::NonNullable,
        );

        assert!(!outer.is_null());

        let outer_list = outer.as_list();

        // First middle list is not null but contains a null element.
        let first_middle = outer_list.element(0).unwrap();
        assert!(!first_middle.is_null());
        assert!(first_middle.as_list().element(1).unwrap().is_null());

        // Second middle list is null.
        let second_middle = outer_list.element(1).unwrap();
        assert!(second_middle.is_null());
    }

    #[test]
    fn test_fixed_size_list_null_equality() {
        // Test equality comparisons with null FixedSizeLists.
        let null_list1 = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            2,
            Nullability::Nullable,
        ));

        let null_list2 = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            2,
            Nullability::Nullable,
        ));

        assert_eq!(null_list1, null_list2);

        // Different sizes should still be equal if both null.
        let null_list3 = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            3,
            Nullability::Nullable,
        ));

        // These have different types, so they should not be equal.
        assert_ne!(null_list1, null_list3);

        // Non-null list with null elements vs null list.
        let list_with_nulls = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::Nullable,
        );

        assert_ne!(null_list1, list_with_nulls);
    }

    #[test]
    fn test_struct_with_multiple_nullable_lists() {
        // Test struct with multiple nullable list fields.
        let struct_dtype = DType::struct_(
            [
                (
                    "fixed_list",
                    DType::FixedSizeList(
                        Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
                        2,
                        Nullability::Nullable,
                    ),
                ),
                (
                    "var_list",
                    DType::List(
                        Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
                        Nullability::Nullable,
                    ),
                ),
                (
                    "nested_fixed",
                    DType::FixedSizeList(
                        Arc::new(DType::FixedSizeList(
                            Arc::new(DType::Primitive(PType::U8, Nullability::Nullable)),
                            2,
                            Nullability::Nullable,
                        )),
                        2,
                        Nullability::Nullable,
                    ),
                ),
            ],
            Nullability::NonNullable,
        );

        // Create struct with mixed null patterns.
        let fixed_list_field = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(10i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::Nullable,
        );

        let var_list_field = Scalar::null(DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
            Nullability::Nullable,
        ));

        let inner_fixed1 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::U8, Nullability::Nullable)),
            vec![
                Scalar::primitive(1u8, Nullability::Nullable),
                Scalar::primitive(2u8, Nullability::Nullable),
            ],
            Nullability::Nullable,
        );

        let inner_fixed2 = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::Nullable)),
            2,
            Nullability::Nullable,
        ));

        let nested_fixed_field = Scalar::fixed_size_list(
            Arc::new(DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::U8, Nullability::Nullable)),
                2,
                Nullability::Nullable,
            )),
            vec![inner_fixed1, inner_fixed2],
            Nullability::Nullable,
        );

        let struct_scalar = Scalar::struct_(
            struct_dtype,
            vec![fixed_list_field, var_list_field, nested_fixed_field],
        );

        let struct_view = struct_scalar.as_struct();

        // Check fixed_list field: not null, but has null element.
        let fixed_field = struct_view.field("fixed_list").unwrap();
        assert!(!fixed_field.is_null());
        assert!(fixed_field.as_list().element(1).unwrap().is_null());

        // Check var_list field: is null.
        assert!(struct_view.field("var_list").unwrap().is_null());

        // Check nested_fixed field: not null, but second element is null.
        let nested_field = struct_view.field("nested_fixed").unwrap();
        assert!(!nested_field.is_null());
        assert!(nested_field.as_list().element(1).unwrap().is_null());
    }

    #[test]
    fn test_casting_preserves_null_positions() {
        // Ensure that casting preserves the positions of null elements.
        let fixed_list = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(1i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::primitive(3i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::NonNullable,
        );

        // Cast to I64.
        let target = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
            4,
            Nullability::NonNullable,
        );

        let casted = fixed_list.cast(&target).unwrap();
        let casted_list = casted.as_list();

        // Verify null positions are preserved.
        assert!(!casted_list.element(0).unwrap().is_null());
        assert!(casted_list.element(1).unwrap().is_null());
        assert!(!casted_list.element(2).unwrap().is_null());
        assert!(casted_list.element(3).unwrap().is_null());

        // Verify non-null values were cast correctly.
        assert_eq!(
            casted_list
                .element(0)
                .unwrap()
                .as_primitive()
                .typed_value::<i64>(),
            Some(1)
        );
        assert_eq!(
            casted_list
                .element(2)
                .unwrap()
                .as_primitive()
                .typed_value::<i64>(),
            Some(3)
        );
    }
}
