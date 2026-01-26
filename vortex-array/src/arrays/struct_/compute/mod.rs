// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod is_constant;
mod mask;
mod min_max;
mod take;
mod zip;

#[cfg(test)]
mod tests {
    use Nullability::NonNullable;
    use Nullability::Nullable;
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::FieldNames;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_dtype::StructFields;
    use vortex_error::VortexExpect;

    use crate::Array;
    use crate::IntoArray as _;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinArray;
    use crate::assert_arrays_eq;
    use crate::compute::cast;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::is_constant;
    use crate::compute::take;
    use crate::validity::Validity;

    #[test]
    fn take_empty_struct() {
        let struct_arr =
            StructArray::try_new(FieldNames::empty(), vec![], 10, Validity::NonNullable).unwrap();
        let indices = PrimitiveArray::from_option_iter([Some(1), None]);
        let taken = take(struct_arr.as_ref(), indices.as_ref()).unwrap();

        assert_arrays_eq!(
            taken,
            StructArray::new(
                FieldNames::empty(),
                vec![],
                2,
                Validity::from_iter([true, false])
            )
        );
    }

    #[test]
    fn take_field_struct() {
        let struct_arr =
            StructArray::from_fields(&[("a", PrimitiveArray::from_iter(0..10).to_array())])
                .unwrap();
        let indices = PrimitiveArray::from_option_iter([Some(1), None]);
        let taken = take(struct_arr.as_ref(), indices.as_ref()).unwrap();
        assert_arrays_eq!(
            taken,
            StructArray::try_from_iter_with_validity(
                [("a", buffer![1, 0])],
                Validity::from_iter([true, false])
            )
            .unwrap()
        );
    }

    #[test]
    fn test_mask_empty_struct() {
        test_mask_conformance(
            StructArray::try_new(FieldNames::empty(), vec![], 5, Validity::NonNullable)
                .unwrap()
                .as_ref(),
        );
    }

    #[test]
    fn test_mask_complex_struct() {
        let xs = buffer![0i64, 1, 2, 3, 4].into_array();
        let ys = VarBinArray::from_iter(
            [Some("a"), Some("b"), None, Some("d"), None],
            DType::Utf8(Nullable),
        )
        .into_array();
        let zs =
            BoolArray::from_iter([Some(true), Some(true), None, None, Some(false)]).into_array();

        test_mask_conformance(
            StructArray::try_new(
                ["xs", "ys", "zs"].into(),
                vec![
                    StructArray::try_new(
                        ["left", "right"].into(),
                        vec![xs.clone(), xs],
                        5,
                        Validity::NonNullable,
                    )
                    .unwrap()
                    .into_array(),
                    ys,
                    zs,
                ],
                5,
                Validity::NonNullable,
            )
            .unwrap()
            .as_ref(),
        );
    }

    #[test]
    fn test_cast_empty_struct() {
        let array = StructArray::try_new(FieldNames::default(), vec![], 5, Validity::NonNullable)
            .unwrap()
            .into_array();
        let non_nullable_dtype = DType::Struct(
            StructFields::new(FieldNames::default(), vec![]),
            NonNullable,
        );
        let casted = cast(&array, &non_nullable_dtype).unwrap();
        assert_eq!(casted.dtype(), &non_nullable_dtype);

        let nullable_dtype =
            DType::Struct(StructFields::new(FieldNames::default(), vec![]), Nullable);
        let casted = cast(&array, &nullable_dtype).unwrap();
        assert_eq!(casted.dtype(), &nullable_dtype);
    }

    #[test]
    fn test_cast_complex_struct() {
        let xs = PrimitiveArray::from_option_iter([Some(0i64), Some(1), Some(2), Some(3), Some(4)]);
        let ys = VarBinArray::from_vec(vec!["a", "b", "c", "d", "e"], DType::Utf8(Nullable));
        let zs = BoolArray::new(
            BitBuffer::from_iter([true, true, false, false, true]),
            Validity::AllValid,
        );
        let fully_nullable_array = StructArray::try_new(
            ["xs", "ys", "zs"].into(),
            vec![
                StructArray::try_new(
                    ["left", "right"].into(),
                    vec![xs.to_array(), xs.to_array()],
                    5,
                    Validity::AllValid,
                )
                .unwrap()
                .into_array(),
                ys.into_array(),
                zs.into_array(),
            ],
            5,
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let top_level_non_nullable = fully_nullable_array.dtype().as_nonnullable();
        let casted = cast(&fully_nullable_array, &top_level_non_nullable).unwrap();
        assert_eq!(casted.dtype(), &top_level_non_nullable);

        let non_null_xs_right = DType::Struct(
            StructFields::new(
                ["xs", "ys", "zs"].into(),
                vec![
                    DType::Struct(
                        StructFields::new(
                            ["left", "right"].into(),
                            vec![
                                DType::Primitive(PType::I64, NonNullable),
                                DType::Primitive(PType::I64, Nullable),
                            ],
                        ),
                        Nullable,
                    ),
                    DType::Utf8(Nullable),
                    DType::Bool(Nullable),
                ],
            ),
            Nullable,
        );
        let casted = cast(&fully_nullable_array, &non_null_xs_right).unwrap();
        assert_eq!(casted.dtype(), &non_null_xs_right);

        let non_null_xs = DType::Struct(
            StructFields::new(
                ["xs", "ys", "zs"].into(),
                vec![
                    DType::Struct(
                        StructFields::new(
                            ["left", "right"].into(),
                            vec![
                                DType::Primitive(PType::I64, Nullable),
                                DType::Primitive(PType::I64, Nullable),
                            ],
                        ),
                        NonNullable,
                    ),
                    DType::Utf8(Nullable),
                    DType::Bool(Nullable),
                ],
            ),
            Nullable,
        );
        let casted = cast(&fully_nullable_array, &non_null_xs).unwrap();
        assert_eq!(casted.dtype(), &non_null_xs);
    }

    #[test]
    fn test_empty_struct_is_constant() {
        let array = StructArray::new_fieldless_with_len(2);
        let is_constant =
            is_constant(array.as_ref()).vortex_expect("operation should succeed in test");
        assert_eq!(is_constant, Some(true));
    }

    #[test]
    fn test_take_empty_struct_conformance() {
        test_take_conformance(
            StructArray::try_new(FieldNames::empty(), vec![], 5, Validity::NonNullable)
                .unwrap()
                .as_ref(),
        );
    }

    #[test]
    fn test_take_simple_struct_conformance() {
        let xs = buffer![1i64, 2, 3, 4, 5].into_array();
        let ys = VarBinArray::from_iter(
            ["a", "b", "c", "d", "e"].map(Some),
            DType::Utf8(NonNullable),
        )
        .into_array();

        test_take_conformance(
            StructArray::try_new(["xs", "ys"].into(), vec![xs, ys], 5, Validity::NonNullable)
                .unwrap()
                .as_ref(),
        );
    }

    #[test]
    fn test_take_nullable_struct_conformance() {
        // Test struct with nullable fields
        let xs = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]);
        let ys = VarBinArray::from_iter(
            [Some("a"), Some("b"), None, Some("d"), None],
            DType::Utf8(Nullable),
        );

        test_take_conformance(
            StructArray::try_new(
                ["xs", "ys"].into(),
                vec![xs.into_array(), ys.into_array()],
                5,
                Validity::NonNullable,
            )
            .unwrap()
            .as_ref(),
        );
    }

    #[test]
    fn test_take_nested_struct_conformance() {
        // Test nested struct
        let inner_xs = buffer![10i32, 20, 30, 40, 50].into_array();
        let inner_ys = buffer![100i32, 200, 300, 400, 500].into_array();
        let inner_struct = StructArray::try_new(
            ["x", "y"].into(),
            vec![inner_xs, inner_ys],
            5,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let outer_zs = BoolArray::from_iter([true, false, true, false, true]).into_array();

        test_take_conformance(
            StructArray::try_new(
                ["inner", "z"].into(),
                vec![inner_struct, outer_zs],
                5,
                Validity::NonNullable,
            )
            .unwrap()
            .as_ref(),
        );
    }

    #[test]
    fn test_take_single_element_struct_conformance() {
        let xs = buffer![42i64].into_array();
        let ys = VarBinArray::from_iter(["hello"].map(Some), DType::Utf8(NonNullable)).into_array();

        test_take_conformance(
            StructArray::try_new(["xs", "ys"].into(), vec![xs, ys], 1, Validity::NonNullable)
                .unwrap()
                .as_ref(),
        );
    }

    #[test]
    fn test_take_large_struct_conformance() {
        // Test with larger array for additional edge cases
        let xs = PrimitiveArray::from_iter(0i64..100).into_array();
        let ys = VarBinArray::from_iter(
            (0..100).map(|i| format!("str_{i}")).map(Some),
            DType::Utf8(NonNullable),
        )
        .into_array();
        let zs = BoolArray::from_iter((0..100).map(|i| i % 2 == 0)).into_array();

        test_take_conformance(
            StructArray::try_new(
                ["xs", "ys", "zs"].into(),
                vec![xs, ys, zs],
                100,
                Validity::NonNullable,
            )
            .unwrap()
            .as_ref(),
        );
    }

    // Consistency tests
    #[rstest]
    // From test_all_consistency
    #[case::struct_simple({
        let xs = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let ys = VarBinArray::from_iter(
            ["a", "b", "c", "d", "e"].map(Some),
            DType::Utf8(NonNullable),
        );
        StructArray::try_new(
            ["xs", "ys"].into(),
            vec![xs.into_array(), ys.into_array()],
            5,
            Validity::NonNullable,
        )
        .unwrap()
    })]
    #[case::struct_nullable({
        let xs = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]);
        let ys = VarBinArray::from_iter(
            [Some("a"), Some("b"), None, Some("d"), None],
            DType::Utf8(Nullable),
        );
        StructArray::try_new(
            ["xs", "ys"].into(),
            vec![xs.into_array(), ys.into_array()],
            5,
            Validity::NonNullable,
        )
        .unwrap()
    })]
    // Additional test cases
    #[case::empty_struct(StructArray::try_new(FieldNames::empty(), vec![], 5, Validity::NonNullable).unwrap())]
    #[case::single_field({
        let xs = buffer![42i64].into_array();
        StructArray::try_new(["xs"].into(), vec![xs], 1, Validity::NonNullable).unwrap()
    })]
    #[case::large_struct({
        let xs = PrimitiveArray::from_iter(0..100i64).into_array();
        let ys = VarBinArray::from_iter(
            (0..100).map(|i| format!("value_{i}")).map(Some),
            DType::Utf8(NonNullable),
        ).into_array();
        StructArray::try_new(["xs", "ys"].into(), vec![xs, ys], 100, Validity::NonNullable).unwrap()
    })]
    fn test_struct_consistency(#[case] array: StructArray) {
        test_array_consistency(array.as_ref());
    }
}
