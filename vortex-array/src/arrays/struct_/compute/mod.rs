// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod filter;
mod is_constant;
mod mask;
mod min_max;
mod take;
mod zip;

#[cfg(test)]
mod tests {
    use Nullability::{NonNullable, Nullable};
    use rstest::rstest;
    use vortex_buffer::{BitBuffer, buffer};
    use vortex_dtype::{DType, FieldNames, Nullability, PType, StructFields};
    use vortex_error::VortexUnwrap;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::arrays::{BoolArray, PrimitiveArray, StructArray, VarBinArray};
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::{cast, filter, is_constant, take};
    use crate::validity::Validity;
    use crate::{Array, IntoArray as _};

    #[test]
    fn filter_empty_struct() {
        let struct_arr =
            StructArray::try_new(FieldNames::empty(), vec![], 10, Validity::NonNullable).unwrap();
        let mask = vec![
            false, true, false, true, false, true, false, true, false, true,
        ];
        let filtered = filter(struct_arr.as_ref(), &Mask::from_iter(mask)).unwrap();
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn take_empty_struct() {
        let struct_arr =
            StructArray::try_new(FieldNames::empty(), vec![], 10, Validity::NonNullable).unwrap();
        let indices = PrimitiveArray::from_option_iter([Some(1), None]);
        let taken = take(struct_arr.as_ref(), indices.as_ref()).unwrap();
        assert_eq!(taken.len(), 2);

        assert_eq!(
            taken.scalar_at(0),
            Scalar::struct_(
                DType::Struct(StructFields::new(FieldNames::default(), vec![]), Nullable),
                vec![]
            )
        );
        assert_eq!(
            taken.scalar_at(1),
            Scalar::null(DType::Struct(
                StructFields::new(FieldNames::default(), vec![]),
                Nullable
            ))
        );
    }

    #[test]
    fn take_field_struct() {
        let struct_arr =
            StructArray::from_fields(&[("a", PrimitiveArray::from_iter(0..10).to_array())])
                .unwrap();
        let indices = PrimitiveArray::from_option_iter([Some(1), None]);
        let taken = take(struct_arr.as_ref(), indices.as_ref()).unwrap();
        assert_eq!(taken.len(), 2);

        assert_eq!(
            taken.scalar_at(0),
            Scalar::struct_(
                struct_arr.dtype().union_nullability(Nullable),
                vec![Scalar::primitive(1, NonNullable)],
            )
        );
        assert_eq!(
            taken.scalar_at(1),
            Scalar::null(struct_arr.dtype().union_nullability(Nullable),)
        );
    }

    #[test]
    fn filter_empty_struct_with_empty_filter() {
        let struct_arr =
            StructArray::try_new(FieldNames::empty(), vec![], 0, Validity::NonNullable).unwrap();
        let filtered = filter(struct_arr.as_ref(), &Mask::from_iter::<[bool; 0]>([])).unwrap();
        assert_eq!(filtered.len(), 0);
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
    fn test_filter_empty_struct() {
        test_filter_conformance(
            StructArray::try_new(FieldNames::empty(), vec![], 5, Validity::NonNullable)
                .unwrap()
                .as_ref(),
        );
    }

    #[test]
    fn test_filter_complex_struct() {
        let xs = buffer![0i64, 1, 2, 3, 4].into_array();
        let ys = VarBinArray::from_iter(
            [Some("a"), Some("b"), None, Some("d"), None],
            DType::Utf8(Nullable),
        )
        .into_array();
        let zs =
            BoolArray::from_iter([Some(true), Some(true), None, None, Some(false)]).into_array();

        test_filter_conformance(
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
    fn test_cast_cannot_change_name_order() {
        let array = StructArray::try_new(
            ["xs", "ys", "zs"].into(),
            vec![
                buffer![1u8].into_array(),
                buffer![1u8].into_array(),
                buffer![1u8].into_array(),
            ],
            1,
            Validity::NonNullable,
        )
        .unwrap();

        let tu8 = DType::Primitive(PType::U8, NonNullable);

        let result = cast(
            array.as_ref(),
            &DType::Struct(
                StructFields::new(
                    FieldNames::from(["ys", "xs", "zs"]),
                    vec![tu8.clone(), tu8.clone(), tu8],
                ),
                NonNullable,
            ),
        );
        assert!(
            result.as_ref().is_err_and(|err| {
                err.to_string()
                    .contains("cannot cast {xs=u8, ys=u8, zs=u8} to {ys=u8, xs=u8, zs=u8}")
            }),
            "{result:?}"
        );
    }

    #[test]
    fn test_cast_complex_struct() {
        let xs = PrimitiveArray::from_option_iter([Some(0i64), Some(1), Some(2), Some(3), Some(4)]);
        let ys = VarBinArray::from_vec(vec!["a", "b", "c", "d", "e"], DType::Utf8(Nullable));
        let zs = BoolArray::from_bit_buffer(
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
        let is_constant = is_constant(array.as_ref()).vortex_unwrap();
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
