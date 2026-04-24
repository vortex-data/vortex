// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for nested structures including Lists, FixedSizeLists, and Structs.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    #[test]
    fn test_fixed_size_list_of_fixed_size_list() {
        // Create FixedSizeList[2] of FixedSizeList[3] of I32.
        let inner_element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let inner_dtype = Arc::new(DType::FixedSizeList(
            Arc::clone(&inner_element_dtype),
            3,
            Nullability::NonNullable,
        ));

        // Create inner FixedSizeLists.
        let inner_list1 = Scalar::fixed_size_list(
            Arc::clone(&inner_element_dtype),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
                Scalar::primitive(3i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let inner_list2 = Scalar::fixed_size_list(
            inner_element_dtype,
            vec![
                Scalar::primitive(4i32, Nullability::NonNullable),
                Scalar::primitive(5i32, Nullability::NonNullable),
                Scalar::primitive(6i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Create outer FixedSizeList.
        let outer_list = Scalar::fixed_size_list(
            inner_dtype,
            vec![inner_list1, inner_list2],
            Nullability::NonNullable,
        );

        assert!(matches!(outer_list.dtype(), DType::FixedSizeList(_, 2, _)));

        // Access nested elements.
        let outer = outer_list.as_list();
        assert_eq!(outer.len(), 2);

        let first_inner = outer.element(0).unwrap();
        let first_inner_list = first_inner.as_list();
        assert_eq!(first_inner_list.len(), 3);
        assert_eq!(
            first_inner_list
                .element(0)
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(1)
        );

        let second_inner = outer.element(1).unwrap();
        let second_inner_list = second_inner.as_list();
        assert_eq!(
            second_inner_list
                .element(2)
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(6)
        );
    }

    #[test]
    fn test_fixed_size_list_of_list() {
        // Create FixedSizeList[2] of variable List of I32.
        let inner_element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let inner_dtype = Arc::new(DType::List(
            Arc::clone(&inner_element_dtype),
            Nullability::NonNullable,
        ));

        let inner_list1 = Scalar::list(
            Arc::clone(&inner_element_dtype),
            vec![
                Scalar::primitive(10i32, Nullability::NonNullable),
                Scalar::primitive(20i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let inner_list2 = Scalar::list(
            inner_element_dtype,
            vec![
                Scalar::primitive(30i32, Nullability::NonNullable),
                Scalar::primitive(40i32, Nullability::NonNullable),
                Scalar::primitive(50i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let outer_fixed_list = Scalar::fixed_size_list(
            inner_dtype,
            vec![inner_list1, inner_list2],
            Nullability::NonNullable,
        );

        let outer = outer_fixed_list.as_list();
        assert_eq!(outer.len(), 2);

        // First inner list has 2 elements.
        let first_inner = outer.element(0).unwrap();
        assert_eq!(first_inner.as_list().len(), 2);

        // Second inner list has 3 elements.
        let second_inner = outer.element(1).unwrap();
        assert_eq!(second_inner.as_list().len(), 3);
    }

    #[test]
    fn test_list_of_fixed_size_list() {
        // Create variable List of FixedSizeList[3] of I32.
        let inner_element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let inner_dtype = Arc::new(DType::FixedSizeList(
            Arc::clone(&inner_element_dtype),
            3,
            Nullability::NonNullable,
        ));

        let fixed_list1 = Scalar::fixed_size_list(
            Arc::clone(&inner_element_dtype),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
                Scalar::primitive(3i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let fixed_list2 = Scalar::fixed_size_list(
            inner_element_dtype,
            vec![
                Scalar::primitive(4i32, Nullability::NonNullable),
                Scalar::primitive(5i32, Nullability::NonNullable),
                Scalar::primitive(6i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let outer_list = Scalar::list(
            inner_dtype,
            vec![fixed_list1, fixed_list2],
            Nullability::NonNullable,
        );

        assert!(matches!(outer_list.dtype(), DType::List(..)));

        let outer = outer_list.as_list();
        assert_eq!(outer.len(), 2);

        // Each inner list should be FixedSizeList[3].
        let first_inner = outer.element(0).unwrap();
        assert!(matches!(first_inner.dtype(), DType::FixedSizeList(_, 3, _)));
        assert_eq!(first_inner.as_list().len(), 3);
    }

    #[test]
    fn test_fixed_size_list_containing_structs() {
        // Create FixedSizeList[2] of Struct{a: I32, b: Utf8}.
        let struct_dtype = DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Utf8(Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        let struct1 = Scalar::struct_(
            struct_dtype.clone(),
            vec![
                Scalar::primitive(100i32, Nullability::NonNullable),
                Scalar::utf8("first", Nullability::NonNullable),
            ],
        );

        let struct2 = Scalar::struct_(
            struct_dtype.clone(),
            vec![
                Scalar::primitive(200i32, Nullability::NonNullable),
                Scalar::utf8("second", Nullability::NonNullable),
            ],
        );

        let fixed_list_of_structs = Scalar::fixed_size_list(
            Arc::new(struct_dtype),
            vec![struct1, struct2],
            Nullability::NonNullable,
        );

        let list = fixed_list_of_structs.as_list();
        assert_eq!(list.len(), 2);

        // Access struct fields through the list.
        let first_struct = list.element(0).unwrap();
        let first = first_struct.as_struct();
        assert_eq!(
            first
                .field("a")
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(100)
        );
        assert_eq!(
            first
                .field("b")
                .unwrap()
                .as_utf8()
                .value()
                .cloned()
                .unwrap(),
            "first".into()
        );
    }

    #[test]
    fn test_struct_with_fixed_size_list_field() {
        // Create Struct with a FixedSizeList field.
        let fixed_list_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            3,
            Nullability::NonNullable,
        );

        let struct_dtype = DType::struct_(
            [
                ("name", DType::Utf8(Nullability::NonNullable)),
                ("values", fixed_list_dtype),
                (
                    "count",
                    DType::Primitive(PType::U32, Nullability::NonNullable),
                ),
            ],
            Nullability::NonNullable,
        );

        let fixed_list_field = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            vec![
                Scalar::primitive(10i64, Nullability::NonNullable),
                Scalar::primitive(20i64, Nullability::NonNullable),
                Scalar::primitive(30i64, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let struct_scalar = Scalar::struct_(
            struct_dtype,
            vec![
                Scalar::utf8("test_struct", Nullability::NonNullable),
                fixed_list_field,
                Scalar::primitive(3u32, Nullability::NonNullable),
            ],
        );

        let struct_view = struct_scalar.as_struct();
        assert_eq!(
            struct_view
                .field("name")
                .unwrap()
                .as_utf8()
                .value()
                .unwrap()
                .as_str(),
            "test_struct"
        );

        let values_field = struct_view.field("values").unwrap();
        assert!(matches!(
            values_field.dtype(),
            DType::FixedSizeList(_, 3, _)
        ));
        assert_eq!(values_field.as_list().len(), 3);
        assert_eq!(
            values_field
                .as_list()
                .element(1)
                .unwrap()
                .as_primitive()
                .typed_value::<i64>(),
            Some(20)
        );
    }

    #[test]
    fn test_list_cast_element_types() {
        // Test casting list elements between different primitive types.
        let list = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            Some(ScalarValue::List(vec![
                Some(ScalarValue::Primitive(PValue::U16(6))),
                Some(ScalarValue::Primitive(PValue::U16(100))),
            ])),
        );

        // Cast U16 -> U32.
        let target_u32 = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
            Nullability::Nullable,
        );
        let casted = list.cast(&target_u32).unwrap();
        assert_eq!(casted.dtype(), &target_u32);

        // Cast U16 -> U8 (with values that fit).
        let target_u8 = DType::List(
            Arc::from(DType::Primitive(PType::U8, Nullability::Nullable)),
            Nullability::Nullable,
        );
        let casted = list.cast(&target_u8).unwrap();
        assert_eq!(casted.dtype(), &target_u8);

        // Cast U16 -> I32.
        let target_i32 = DType::List(
            Arc::from(DType::Primitive(PType::I32, Nullability::Nullable)),
            Nullability::Nullable,
        );
        let casted = list.cast(&target_i32).unwrap();
        assert_eq!(casted.dtype(), &target_i32);
    }

    #[test]
    fn test_list_cast_element_overflow() {
        // Test that casting U16 values too large for U8 fails.
        let list_with_large_values = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            Some(ScalarValue::List(vec![
                Some(ScalarValue::Primitive(PValue::U16(100))),
                Some(ScalarValue::Primitive(PValue::U16(256))), // Too large for U8
                Some(ScalarValue::Primitive(PValue::U16(1000))), // Too large for U8
            ])),
        );

        let target_u8 = DType::List(
            Arc::from(DType::Primitive(PType::U8, Nullability::Nullable)),
            Nullability::Nullable,
        );

        let result = list_with_large_values.cast(&target_u8);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot cast 256u16 to u8")
        );
    }

    #[test]
    fn test_list_cast_nested_lists() {
        // Create a list of lists.
        let inner_list1 = Scalar::list(
            Arc::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );
        let inner_list2 = Scalar::list(
            Arc::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(3i32, Nullability::NonNullable),
                Scalar::primitive(4i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );
        let nested_list = Scalar::list(
            Arc::from(DType::List(
                Arc::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
                Nullability::NonNullable,
            )),
            vec![inner_list1, inner_list2],
            Nullability::NonNullable,
        );

        // Cast to nested list with I64 elements.
        let target = DType::List(
            Arc::from(DType::List(
                Arc::from(DType::Primitive(PType::I64, Nullability::NonNullable)),
                Nullability::NonNullable,
            )),
            Nullability::NonNullable,
        );
        let casted = nested_list.cast(&target).unwrap();
        assert_eq!(casted.dtype(), &target);
    }

    #[test]
    fn test_list_cast_empty_list() {
        // Test casting empty list.
        let empty_list = Scalar::list(
            Arc::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![],
            Nullability::NonNullable,
        );

        // Cast to different element type.
        let target = DType::List(
            Arc::from(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        let casted = empty_list.cast(&target).unwrap();
        assert_eq!(casted.dtype(), &target);
        assert_eq!(casted.as_list().len(), 0);

        // Cast empty list to FixedSizeList[0].
        let target_fixed = DType::FixedSizeList(
            Arc::from(DType::Primitive(PType::I64, Nullability::NonNullable)),
            0,
            Nullability::NonNullable,
        );
        let casted_fixed = empty_list.cast(&target_fixed).unwrap();
        assert_eq!(casted_fixed.dtype(), &target_fixed);
    }

    #[test]
    fn test_list_cast_string_elements() {
        // Create a list of strings.
        let string_list = Scalar::list(
            Arc::from(DType::Utf8(Nullability::NonNullable)),
            vec![
                Scalar::utf8("hello", Nullability::NonNullable),
                Scalar::utf8("world", Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Cast to nullable strings.
        let target = DType::List(
            Arc::from(DType::Utf8(Nullability::Nullable)),
            Nullability::NonNullable,
        );
        let casted = string_list.cast(&target).unwrap();
        assert_eq!(casted.dtype(), &target);
    }

    #[test]
    fn test_list_cast_struct_elements() {
        // Create a list of structs.
        let struct_dtype = DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );
        let struct1 = Scalar::struct_(
            struct_dtype.clone(),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(10i64, Nullability::NonNullable),
            ],
        );
        let struct2 = Scalar::struct_(
            struct_dtype.clone(),
            vec![
                Scalar::primitive(2i32, Nullability::NonNullable),
                Scalar::primitive(20i64, Nullability::NonNullable),
            ],
        );
        let struct_list = Scalar::list(
            Arc::from(struct_dtype),
            vec![struct1, struct2],
            Nullability::NonNullable,
        );

        // Cast struct fields.
        let target_struct_dtype = DType::struct_(
            [
                ("a", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );
        let target = DType::List(Arc::from(target_struct_dtype), Nullability::NonNullable);
        let casted = struct_list.cast(&target).unwrap();
        assert_eq!(casted.dtype(), &target);
    }

    #[test]
    fn test_list_cast_incompatible_element_types() {
        // Create a list of integers.
        let int_list = Scalar::list(
            Arc::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![Scalar::primitive(1i32, Nullability::NonNullable)],
            Nullability::NonNullable,
        );

        // Try to cast to list of strings - should fail.
        let target = DType::List(
            Arc::from(DType::Utf8(Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        assert!(int_list.cast(&target).is_err());
    }

    #[test]
    fn test_list_to_fixed_size_list_cast() {
        // Create a list with 3 elements.
        let list = Scalar::list(
            Arc::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
                Scalar::primitive(3i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Cast to FixedSizeList with matching size.
        let target = DType::FixedSizeList(
            Arc::from(DType::Primitive(PType::I64, Nullability::NonNullable)),
            3,
            Nullability::NonNullable,
        );
        let casted = list.cast(&target).unwrap();
        assert_eq!(casted.dtype(), &target);
        assert_eq!(casted.as_list().len(), 3);
    }

    #[test]
    fn test_fixed_size_list_to_list_cast() {
        // Create a FixedSizeList with 2 elements.
        let fixed_list = Scalar::fixed_size_list(
            Arc::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(10i32, Nullability::NonNullable),
                Scalar::primitive(20i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Cast to regular List.
        let target = DType::List(
            Arc::from(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        let casted = fixed_list.cast(&target).unwrap();
        assert_eq!(casted.dtype(), &target);
        assert_eq!(casted.as_list().len(), 2);
    }

    #[test]
    fn test_fixed_size_list_to_fixed_size_list_cast() {
        // Create a FixedSizeList[4] with I32 elements.
        let fixed_list = Scalar::fixed_size_list(
            Arc::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
                Scalar::primitive(3i32, Nullability::NonNullable),
                Scalar::primitive(4i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Cast to FixedSizeList[4] with I64 elements.
        let target = DType::FixedSizeList(
            Arc::from(DType::Primitive(PType::I64, Nullability::NonNullable)),
            4,
            Nullability::NonNullable,
        );
        let casted = fixed_list.cast(&target).unwrap();
        assert_eq!(casted.dtype(), &target);
    }

    #[test]
    fn test_fixed_size_list_size_mismatch_error() {
        // Create a list with 2 elements.
        let list = Scalar::list(
            Arc::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Try to cast to FixedSizeList[3] - should fail.
        let target = DType::FixedSizeList(
            Arc::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
            3,
            Nullability::NonNullable,
        );
        let result = list.cast(&target);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("tried to cast to a `FixedSizeList[3]` but had 2 elements")
        );
    }

    #[test]
    fn test_deeply_nested_mixed_structures() {
        // Create a deeply nested structure:
        // List of FixedSizeList[2] of Struct{id: I32, data: FixedSizeList[2] of I64}.

        let inner_fixed_list_dtype = Arc::new(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            2,
            Nullability::NonNullable,
        ));

        let struct_dtype = Arc::new(DType::struct_(
            [
                ("id", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("data", (*inner_fixed_list_dtype).clone()),
            ],
            Nullability::NonNullable,
        ));

        let middle_fixed_list_dtype = Arc::new(DType::FixedSizeList(
            Arc::clone(&struct_dtype),
            2,
            Nullability::NonNullable,
        ));

        // Create inner FixedSizeList for struct field.
        let inner_list1 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            vec![
                Scalar::primitive(100i64, Nullability::NonNullable),
                Scalar::primitive(200i64, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let inner_list2 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            vec![
                Scalar::primitive(300i64, Nullability::NonNullable),
                Scalar::primitive(400i64, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Create structs.
        let struct1 = Scalar::struct_(
            (*struct_dtype).clone(),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                inner_list1,
            ],
        );

        let struct2 = Scalar::struct_(
            (*struct_dtype).clone(),
            vec![
                Scalar::primitive(2i32, Nullability::NonNullable),
                inner_list2,
            ],
        );

        // Create middle FixedSizeList[2] of structs.
        let middle_list = Scalar::fixed_size_list(
            struct_dtype,
            vec![struct1, struct2],
            Nullability::NonNullable,
        );

        // Create outer List.
        let outer_list = Scalar::list(
            middle_fixed_list_dtype,
            vec![middle_list.clone(), middle_list],
            Nullability::NonNullable,
        );

        // Verify structure.
        assert!(matches!(outer_list.dtype(), DType::List(..)));
        let outer = outer_list.as_list();
        assert_eq!(outer.len(), 2);

        let first_middle = outer.element(0).unwrap();
        assert!(matches!(
            first_middle.dtype(),
            DType::FixedSizeList(_, 2, _)
        ));

        let first_struct = first_middle.as_list().element(0).unwrap();
        let struct_view = first_struct.as_struct();
        assert_eq!(
            struct_view
                .field("id")
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(1)
        );

        let data_field = struct_view.field("data").unwrap();
        assert_eq!(
            data_field
                .as_list()
                .element(0)
                .unwrap()
                .as_primitive()
                .typed_value::<i64>(),
            Some(100)
        );
    }

    #[test]
    fn test_nested_structure_casting() {
        // Test casting nested structures with element type changes.
        let inner_dtype = Arc::new(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::NonNullable,
        ));

        let inner_list1 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let inner_list2 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(3i32, Nullability::NonNullable),
                Scalar::primitive(4i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let outer_list = Scalar::list(
            inner_dtype,
            vec![inner_list1, inner_list2],
            Nullability::NonNullable,
        );

        // Cast to List of FixedSizeList[2] of I64.
        let target_dtype = DType::List(
            Arc::new(DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
                2,
                Nullability::NonNullable,
            )),
            Nullability::NonNullable,
        );

        let casted = outer_list.cast(&target_dtype).unwrap();
        assert!(matches!(casted.dtype(), DType::List(..)));

        let casted_list = casted.as_list();
        let first_inner = casted_list.element(0).unwrap();
        let first_inner_list = first_inner.as_list();
        assert_eq!(
            first_inner_list
                .element(0)
                .unwrap()
                .as_primitive()
                .typed_value::<i64>(),
            Some(1)
        );
    }

    #[test]
    fn test_mixed_list_and_fixed_size_list() {
        // Create a structure mixing List and FixedSizeList at different levels.
        // FixedSizeList[2] of List of FixedSizeList[2] of I32.

        let innermost_dtype = Arc::new(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::NonNullable,
        ));

        let middle_dtype = Arc::new(DType::List(
            Arc::clone(&innermost_dtype),
            Nullability::NonNullable,
        ));

        // Create innermost FixedSizeLists.
        let inner_fixed1 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let inner_fixed2 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(3i32, Nullability::NonNullable),
                Scalar::primitive(4i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Create middle Lists.
        let middle_list1 = Scalar::list(
            Arc::clone(&innermost_dtype),
            vec![inner_fixed1.clone()],
            Nullability::NonNullable,
        );

        let middle_list2 = Scalar::list(
            innermost_dtype,
            vec![inner_fixed2, inner_fixed1],
            Nullability::NonNullable,
        );

        // Create outer FixedSizeList.
        let outer = Scalar::fixed_size_list(
            middle_dtype,
            vec![middle_list1, middle_list2],
            Nullability::NonNullable,
        );

        assert!(matches!(outer.dtype(), DType::FixedSizeList(_, 2, _)));

        let outer_list = outer.as_list();
        assert_eq!(outer_list.len(), 2);

        // Second middle list has 2 inner fixed lists.
        let second_middle = outer_list.element(1).unwrap();
        assert_eq!(second_middle.as_list().len(), 2);
    }

    #[test]
    fn test_nested_structure_nbytes() {
        // Test nbytes calculation for nested structures.
        let inner_dtype = Arc::new(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::NonNullable,
        ));

        let inner_list1 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let inner_list2 = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(3i32, Nullability::NonNullable),
                Scalar::primitive(4i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let outer = Scalar::fixed_size_list(
            inner_dtype,
            vec![inner_list1, inner_list2],
            Nullability::NonNullable,
        );

        // 2 outer elements * 2 inner elements * 4 bytes (i32) = 16 bytes.
        assert_eq!(outer.approx_nbytes(), 16);
    }

    // Tests merged from fixed_size_list.rs

    #[test]
    fn test_fixed_size_list_creation_and_access() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(10i32, Nullability::NonNullable),
            Scalar::primitive(20i32, Nullability::NonNullable),
            Scalar::primitive(30i32, Nullability::NonNullable),
        ];
        let fixed_list = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);

        assert!(matches!(fixed_list.dtype(), DType::FixedSizeList(_, 3, _)));

        let list = fixed_list.as_list();
        assert_eq!(list.len(), 3);
        assert!(!list.is_null());

        // Test element access.
        assert_eq!(
            list.element(0).unwrap().as_primitive().typed_value::<i32>(),
            Some(10)
        );
        assert_eq!(
            list.element(1).unwrap().as_primitive().typed_value::<i32>(),
            Some(20)
        );
        assert_eq!(
            list.element(2).unwrap().as_primitive().typed_value::<i32>(),
            Some(30)
        );
    }

    #[test]
    fn test_fixed_size_list_size_zero() {
        // Test FixedSizeList[0] behavior.
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let empty_fixed_list =
            Scalar::fixed_size_list(Arc::clone(&element_dtype), vec![], Nullability::NonNullable);

        assert!(matches!(
            empty_fixed_list.dtype(),
            DType::FixedSizeList(_, 0, _)
        ));

        let list = empty_fixed_list.as_list();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
        assert!(!list.is_null());

        // Cast to regular list should work.
        let target = DType::List(element_dtype, Nullability::NonNullable);
        let casted = empty_fixed_list.cast(&target).unwrap();
        assert_eq!(casted.as_list().len(), 0);
    }

    #[test]
    fn test_fixed_size_list_cast_to_list() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let fixed_list = Scalar::fixed_size_list(
            Arc::clone(&element_dtype),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Cast to regular List with same element type.
        let target = DType::List(element_dtype, Nullability::NonNullable);
        let casted = fixed_list.cast(&target).unwrap();
        assert!(matches!(casted.dtype(), DType::List(..)));
        assert_eq!(casted.as_list().len(), 2);

        // Cast to regular List with different element type.
        let target_i64 = DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        let casted_i64 = fixed_list.cast(&target_i64).unwrap();
        assert_eq!(casted_i64.as_list().len(), 2);
        assert_eq!(
            casted_i64
                .as_list()
                .element(0)
                .unwrap()
                .as_primitive()
                .typed_value::<i64>(),
            Some(1)
        );
    }

    #[test]
    fn test_list_cast_to_fixed_size_list() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let list = Scalar::list(
            Arc::clone(&element_dtype),
            vec![
                Scalar::primitive(10i32, Nullability::NonNullable),
                Scalar::primitive(20i32, Nullability::NonNullable),
                Scalar::primitive(30i32, Nullability::NonNullable),
                Scalar::primitive(40i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Cast to FixedSizeList[4] should succeed.
        let target = DType::FixedSizeList(element_dtype, 4, Nullability::NonNullable);
        let casted = list.cast(&target).unwrap();
        assert!(matches!(casted.dtype(), DType::FixedSizeList(_, 4, _)));

        // Cast with element type change.
        let target_u32 = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable)),
            4,
            Nullability::NonNullable,
        );
        let casted_u32 = list.cast(&target_u32).unwrap();
        assert_eq!(
            casted_u32
                .as_list()
                .element(0)
                .unwrap()
                .as_primitive()
                .typed_value::<u32>(),
            Some(10)
        );
    }

    #[test]
    fn test_fixed_size_list_size_validation() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let list = Scalar::list(
            Arc::clone(&element_dtype),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        // Try to cast to wrong size - should fail.
        let wrong_size_target =
            DType::FixedSizeList(Arc::clone(&element_dtype), 3, Nullability::NonNullable);
        let result = list.cast(&wrong_size_target);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("tried to cast to a `FixedSizeList[3]` but had 2 elements")
        );

        // Try to cast to wrong size (smaller).
        let smaller_target = DType::FixedSizeList(element_dtype, 1, Nullability::NonNullable);
        let result = list.cast(&smaller_target);
        assert!(result.is_err());
    }

    #[test]
    fn test_fixed_size_list_display() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let fixed_list = Scalar::fixed_size_list(
            element_dtype,
            vec![
                Scalar::primitive(100i32, Nullability::NonNullable),
                Scalar::primitive(200i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let display_str = format!("{}", fixed_list.as_list());
        assert!(display_str.contains("fixed_size<2>"));
        assert!(display_str.contains("100"));
        assert!(display_str.contains("200"));
    }

    #[test]
    fn test_fixed_size_list_equality_and_ordering() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

        // Create two equal fixed size lists.
        let list1 = Scalar::fixed_size_list(
            Arc::clone(&element_dtype),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let list2 = Scalar::fixed_size_list(
            Arc::clone(&element_dtype),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        assert_eq!(list1, list2);

        // Create a different list.
        let list3 = Scalar::fixed_size_list(
            element_dtype,
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(3i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        assert_ne!(list1, list3);

        // Test ordering.
        assert!(list1 < list3);
        assert!(list3 > list1);
    }

    #[test]
    fn test_fixed_size_list_single_element() {
        // Test FixedSizeList[1].
        let element_dtype = Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable));
        let single_element = Scalar::fixed_size_list(
            Arc::clone(&element_dtype),
            vec![Scalar::primitive(42i64, Nullability::NonNullable)],
            Nullability::NonNullable,
        );

        assert!(matches!(
            single_element.dtype(),
            DType::FixedSizeList(_, 1, _)
        ));

        let list = single_element.as_list();
        assert_eq!(list.len(), 1);
        assert_eq!(
            list.element(0).unwrap().as_primitive().typed_value::<i64>(),
            Some(42)
        );

        // Cast to variable list.
        let target = DType::List(element_dtype, Nullability::NonNullable);
        let casted = single_element.cast(&target).unwrap();
        assert_eq!(casted.as_list().len(), 1);
    }

    #[test]
    fn test_fixed_size_list_with_strings() {
        let element_dtype = Arc::new(DType::Utf8(Nullability::NonNullable));
        let string_list = Scalar::fixed_size_list(
            element_dtype,
            vec![
                Scalar::utf8("hello", Nullability::NonNullable),
                Scalar::utf8("world", Nullability::NonNullable),
                Scalar::utf8("test", Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        assert!(matches!(string_list.dtype(), DType::FixedSizeList(_, 3, _)));

        let list = string_list.as_list();
        assert_eq!(
            list.element(0).unwrap().as_utf8().value().unwrap().as_str(),
            "hello"
        );
        assert_eq!(
            list.element(1).unwrap().as_utf8().value().unwrap().as_str(),
            "world"
        );
        assert_eq!(
            list.element(2).unwrap().as_utf8().value().unwrap().as_str(),
            "test"
        );
    }

    #[test]
    fn test_fixed_size_list_nbytes() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

        // FixedSizeList[3] of i32 = 3 * 4 bytes = 12 bytes.
        let fixed_list = Scalar::fixed_size_list(
            Arc::clone(&element_dtype),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
                Scalar::primitive(3i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );
        assert_eq!(fixed_list.approx_nbytes(), 12);

        // Empty FixedSizeList[0] = 0 bytes.
        let empty_list = Scalar::fixed_size_list(element_dtype, vec![], Nullability::NonNullable);
        assert_eq!(empty_list.approx_nbytes(), 0);

        // FixedSizeList with strings.
        let string_list = Scalar::fixed_size_list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            vec![
                Scalar::utf8("ab", Nullability::NonNullable),  // 2 bytes
                Scalar::utf8("cde", Nullability::NonNullable), // 3 bytes
            ],
            Nullability::NonNullable,
        );
        assert_eq!(string_list.approx_nbytes(), 5);
    }

    #[test]
    fn test_fixed_size_list_casting_with_nullability() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::Nullable));
        let fixed_list = Scalar::fixed_size_list(
            element_dtype,
            vec![
                Scalar::primitive(1i32, Nullability::Nullable),
                Scalar::primitive(2i32, Nullability::Nullable),
            ],
            Nullability::Nullable,
        );

        // Cast to non-nullable container.
        let target = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            2,
            Nullability::NonNullable,
        );
        let casted = fixed_list.cast(&target).unwrap();
        assert_eq!(casted.dtype().nullability(), Nullability::NonNullable);

        // Cast to non-nullable elements.
        let target_nonnull_elems = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::Nullable,
        );
        let casted_elems = fixed_list.cast(&target_nonnull_elems).unwrap();
        match casted_elems.dtype() {
            DType::FixedSizeList(elem_dtype, ..) => {
                assert_eq!(elem_dtype.nullability(), Nullability::NonNullable);
            }
            _ => panic!("Expected FixedSizeList"),
        }
    }

    // Additional tests for pure List structures (without FixedSizeList)

    #[test]
    fn test_list_basic_operations() {
        // Create a simple list.
        let list = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
                Scalar::primitive(3i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        assert!(matches!(list.dtype(), DType::List(..)));
        assert_eq!(list.as_list().len(), 3);
        assert!(!list.is_null());
    }

    #[test]
    fn test_list_of_lists() {
        // Create nested lists without FixedSizeList.
        let inner_list1 = Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            vec![
                Scalar::utf8("a", Nullability::NonNullable),
                Scalar::utf8("b", Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let inner_list2 = Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            vec![
                Scalar::utf8("c", Nullability::NonNullable),
                Scalar::utf8("d", Nullability::NonNullable),
                Scalar::utf8("e", Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let outer_list = Scalar::list(
            Arc::new(DType::List(
                Arc::new(DType::Utf8(Nullability::NonNullable)),
                Nullability::NonNullable,
            )),
            vec![inner_list1, inner_list2],
            Nullability::NonNullable,
        );

        assert_eq!(outer_list.as_list().len(), 2);
        assert_eq!(outer_list.as_list().element(0).unwrap().as_list().len(), 2);
        assert_eq!(outer_list.as_list().element(1).unwrap().as_list().len(), 3);
    }

    #[test]
    fn test_struct_only_nesting() {
        // Test nested structs without any lists.
        let inner_struct_dtype = DType::struct_(
            [
                ("x", DType::Primitive(PType::F32, Nullability::NonNullable)),
                ("y", DType::Primitive(PType::F32, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        let outer_struct_dtype = DType::struct_(
            [
                ("id", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("point", inner_struct_dtype.clone()),
                ("name", DType::Utf8(Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        let inner_struct = Scalar::struct_(
            inner_struct_dtype,
            vec![
                Scalar::primitive(1.5f32, Nullability::NonNullable),
                Scalar::primitive(2.5f32, Nullability::NonNullable),
            ],
        );

        let outer_struct = Scalar::struct_(
            outer_struct_dtype,
            vec![
                Scalar::primitive(123i64, Nullability::NonNullable),
                inner_struct,
                Scalar::utf8("test_point", Nullability::NonNullable),
            ],
        );

        let struct_view = outer_struct.as_struct();
        assert_eq!(
            struct_view
                .field("id")
                .unwrap()
                .as_primitive()
                .typed_value::<i64>(),
            Some(123)
        );

        let point_field = struct_view.field("point").unwrap();
        let point_struct = point_field.as_struct();
        assert_eq!(
            point_struct
                .field("x")
                .unwrap()
                .as_primitive()
                .typed_value::<f32>(),
            Some(1.5)
        );
    }

    #[test]
    fn test_list_of_structs_without_fixed_size() {
        // Create a list of structs (no FixedSizeList).
        let struct_dtype = DType::struct_(
            [
                ("key", DType::Utf8(Nullability::NonNullable)),
                (
                    "value",
                    DType::Primitive(PType::I64, Nullability::NonNullable),
                ),
            ],
            Nullability::NonNullable,
        );

        let struct1 = Scalar::struct_(
            struct_dtype.clone(),
            vec![
                Scalar::utf8("first", Nullability::NonNullable),
                Scalar::primitive(100i64, Nullability::NonNullable),
            ],
        );

        let struct2 = Scalar::struct_(
            struct_dtype.clone(),
            vec![
                Scalar::utf8("second", Nullability::NonNullable),
                Scalar::primitive(200i64, Nullability::NonNullable),
            ],
        );

        let struct3 = Scalar::struct_(
            struct_dtype.clone(),
            vec![
                Scalar::utf8("third", Nullability::NonNullable),
                Scalar::primitive(300i64, Nullability::NonNullable),
            ],
        );

        let list_of_structs = Scalar::list(
            Arc::new(struct_dtype),
            vec![struct1, struct2, struct3],
            Nullability::NonNullable,
        );

        assert_eq!(list_of_structs.as_list().len(), 3);

        let first_struct = list_of_structs.as_list().element(0).unwrap();
        assert_eq!(
            first_struct
                .as_struct()
                .field("key")
                .unwrap()
                .as_utf8()
                .value()
                .unwrap()
                .as_str(),
            "first"
        );
    }

    #[test]
    fn test_struct_containing_list() {
        // Create a struct with a List field (not FixedSizeList).
        let struct_dtype = DType::struct_(
            [
                ("id", DType::Primitive(PType::I32, Nullability::NonNullable)),
                (
                    "tags",
                    DType::List(
                        Arc::new(DType::Utf8(Nullability::NonNullable)),
                        Nullability::NonNullable,
                    ),
                ),
                ("active", DType::Bool(Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        let tags_list = Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            vec![
                Scalar::utf8("rust", Nullability::NonNullable),
                Scalar::utf8("scala", Nullability::NonNullable),
                Scalar::utf8("python", Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let struct_with_list = Scalar::struct_(
            struct_dtype,
            vec![
                Scalar::primitive(42i32, Nullability::NonNullable),
                tags_list,
                Scalar::bool(true, Nullability::NonNullable),
            ],
        );

        let struct_view = struct_with_list.as_struct();
        assert_eq!(
            struct_view
                .field("id")
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(42)
        );

        let tags_field = struct_view.field("tags").unwrap();
        assert_eq!(tags_field.as_list().len(), 3);
        assert_eq!(
            tags_field
                .as_list()
                .element(0)
                .unwrap()
                .as_utf8()
                .value()
                .unwrap()
                .as_str(),
            "rust"
        );
    }

    #[test]
    fn test_deeply_nested_lists_only() {
        // Create a 3-level nested list structure without any FixedSizeList.
        let level3 = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let level2 = Scalar::list(
            Arc::new(DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                Nullability::NonNullable,
            )),
            vec![level3.clone(), level3],
            Nullability::NonNullable,
        );

        let level1 = Scalar::list(
            Arc::new(DType::List(
                Arc::new(DType::List(
                    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                    Nullability::NonNullable,
                )),
                Nullability::NonNullable,
            )),
            vec![level2.clone(), level2],
            Nullability::NonNullable,
        );

        assert_eq!(level1.as_list().len(), 2);
        assert_eq!(
            level1
                .as_list()
                .element(0)
                .unwrap()
                .as_list()
                .element(0)
                .unwrap()
                .as_list()
                .element(0)
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(1)
        );
    }
}
