// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use rstest::rstest;
use vortex::error::VortexResult;

use super::intersect_sorted;
use super::nonnegative_number_from_value;
use super::row_range_from_comparison;
use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb::Value;

#[test]
fn nonnegative_number_from_value_handles_signed_unsigned_negative_and_invalid() -> VortexResult<()>
{
    assert_eq!(nonnegative_number_from_value(&Value::from(42i64))?, 42);
    assert_eq!(nonnegative_number_from_value(&Value::from(7i32))?, 7);
    assert_eq!(nonnegative_number_from_value(&Value::from(99u64))?, 99);
    assert_eq!(nonnegative_number_from_value(&Value::from(12u32))?, 12);
    assert!(
        nonnegative_number_from_value(&Value::from(-1i64))
            .unwrap_err()
            .to_string()
            .contains("negative value")
    );
    assert!(
        nonnegative_number_from_value(&Value::from(-5i32))
            .unwrap_err()
            .to_string()
            .contains("negative value")
    );
    assert!(
        nonnegative_number_from_value(&Value::from("abc"))
            .unwrap_err()
            .to_string()
            .contains("unexpected value type")
    );
    Ok(())
}

#[rstest]
#[case::eq(DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL, 5, Some(5..6))]
#[case::gte(
    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO,
    5,
    Some(5..u64::MAX)
)]
#[case::gt(
    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN,
    5,
    Some(6..u64::MAX)
)]
#[case::gt_max(
    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN,
    u64::MAX,
    Some(u64::MAX..u64::MAX)
)]
#[case::lte(
    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO,
    5,
    Some(0..6)
)]
#[case::lt(DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN, 5, Some(0..5))]
#[case::neq(DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOTEQUAL, 5, None)]
fn row_range_from_comparison_cases(
    #[case] op: DUCKDB_VX_EXPR_TYPE,
    #[case] n: u64,
    #[case] expected: Option<Range<u64>>,
) {
    assert_eq!(row_range_from_comparison(op, n), expected);
}

#[rstest]
#[case::overlap(&[1, 3, 5, 7], &[2, 3, 5, 8], vec![3, 5])]
#[case::disjoint(&[1, 2], &[3, 4], vec![])]
#[case::empty_left(&[], &[1, 2], vec![])]
#[case::empty_right(&[1, 2], &[], vec![])]
#[case::identical(&[0, 4, 9], &[0, 4, 9], vec![0, 4, 9])]
fn intersect_sorted_cases(#[case] left: &[u64], #[case] right: &[u64], #[case] expected: Vec<u64>) {
    assert_eq!(intersect_sorted(left, right), expected);
}

#[cfg(table_filter_test_support)]
mod filter_conversion_tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::assert_arrays_eq;
    use vortex::buffer::buffer;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::dtype::StructFields;
    use vortex::error::VortexResult;
    use vortex::expr::ArrayExprExt;
    use vortex::expr::ScalarFnExprExt;
    use vortex::expr::and;
    use vortex::expr::col;
    use vortex::expr::get_item;
    use vortex::expr::in_list_of;
    use vortex::expr::is_not_null;
    use vortex::expr::is_null;
    use vortex::expr::lit;
    use vortex::expr::or;
    use vortex::scalar::Scalar;
    use vortex::scalar_fn::builtin::compare::Compare;
    use vortex::scalar_fn::builtin::operators::CompareOperator;
    use vortex::scan::selection::Selection;

    use super::super::try_from_table_filter;
    use super::super::try_from_virtual_column_filter;
    use crate::SESSION;
    use crate::cpp::DUCKDB_VX_EXPR_TYPE;
    use crate::duckdb::LogicalType;
    use crate::duckdb::Value;
    use crate::duckdb::test_support::OwnedTableFilter;
    use crate::projection::is_optional_table_filter;

    fn test_scope_dtype() -> DType {
        let nested = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::I32, Nullability::Nullable)],
            ),
            Nullability::Nullable,
        );
        DType::Struct(
            StructFields::new(
                ["a", "s"].into(),
                vec![DType::Primitive(PType::I32, Nullability::Nullable), nested],
            ),
            Nullability::NonNullable,
        )
    }

    #[test]
    fn try_from_table_filter_constant_null_conjunctions_struct_in_and_bloom() -> VortexResult<()> {
        let scope = test_scope_dtype();
        let a = col("a");

        // ConstantComparison
        let eq = OwnedTableFilter::constant(
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL,
            &Value::from(5i32),
        );
        assert_eq!(
            try_from_table_filter(eq.as_ref(), &a, &scope)?,
            Some(Compare.new_expr(
                CompareOperator::Eq,
                [
                    a.clone(),
                    lit(Scalar::primitive(5i32, Nullability::Nullable))
                ]
            ))
        );

        // IsNull / IsNotNull
        assert_eq!(
            try_from_table_filter(OwnedTableFilter::is_null().as_ref(), &a, &scope)?,
            Some(is_null(a.clone()))
        );
        assert_eq!(
            try_from_table_filter(OwnedTableFilter::is_not_null().as_ref(), &a, &scope)?,
            Some(is_not_null(a.clone()))
        );

        // ConjunctionAnd: populated, empty -> lit(true), child returning None -> None
        let and_filter = OwnedTableFilter::conjunction_and(vec![
            OwnedTableFilter::is_not_null(),
            OwnedTableFilter::constant(
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN,
                &Value::from(1i32),
            ),
        ]);
        assert_eq!(
            try_from_table_filter(and_filter.as_ref(), &a, &scope)?,
            Some(and(
                is_not_null(a.clone()),
                Compare.new_expr(
                    CompareOperator::Gt,
                    [
                        a.clone(),
                        lit(Scalar::primitive(1i32, Nullability::Nullable))
                    ]
                )
            ))
        );
        assert_eq!(
            try_from_table_filter(
                OwnedTableFilter::conjunction_and(vec![]).as_ref(),
                &a,
                &scope
            )?,
            Some(lit(true))
        );
        let and_with_none = OwnedTableFilter::conjunction_and(vec![
            OwnedTableFilter::is_not_null(),
            OwnedTableFilter::expr_unsupported(),
        ]);
        assert_eq!(
            try_from_table_filter(and_with_none.as_ref(), &a, &scope)?,
            None
        );

        // ConjunctionOr: populated, empty -> lit(false), child returning None -> None
        let or_filter = OwnedTableFilter::conjunction_or(vec![
            OwnedTableFilter::is_null(),
            OwnedTableFilter::constant(
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN,
                &Value::from(3i32),
            ),
        ]);
        assert_eq!(
            try_from_table_filter(or_filter.as_ref(), &a, &scope)?,
            Some(or(
                is_null(a.clone()),
                Compare.new_expr(
                    CompareOperator::Lt,
                    [
                        a.clone(),
                        lit(Scalar::primitive(3i32, Nullability::Nullable))
                    ]
                )
            ))
        );
        assert_eq!(
            try_from_table_filter(
                OwnedTableFilter::conjunction_or(vec![]).as_ref(),
                &a,
                &scope
            )?,
            Some(lit(false))
        );
        let or_with_none = OwnedTableFilter::conjunction_or(vec![
            OwnedTableFilter::is_null(),
            OwnedTableFilter::expr_unsupported(),
        ]);
        assert_eq!(
            try_from_table_filter(or_with_none.as_ref(), &a, &scope)?,
            None
        );

        // StructExtract
        let struct_filter = OwnedTableFilter::struct_extract(
            0,
            "x",
            OwnedTableFilter::constant(
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO,
                &Value::from(10i32),
            ),
        );
        assert_eq!(
            try_from_table_filter(struct_filter.as_ref(), &col("s"), &scope)?,
            Some(Compare.new_expr(
                CompareOperator::Gte,
                [
                    get_item("x", col("s")),
                    lit(Scalar::primitive(10i32, Nullability::Nullable))
                ]
            ))
        );

        // InFilter
        let in_filter = OwnedTableFilter::in_filter(&[Value::from(2i32), Value::from(4i32)]);
        let expected_in = in_list_of(
            a.clone(),
            DType::Primitive(PType::I32, Nullability::Nullable),
            vec![
                Scalar::primitive(2i32, Nullability::Nullable),
                Scalar::primitive(4i32, Nullability::Nullable),
            ],
        )?;
        assert_eq!(
            try_from_table_filter(in_filter.as_ref(), &a, &scope)?,
            Some(expected_in)
        );

        // Bloom returns Err, and Optional(Bloom) degrades to Ok(None).
        let bloom = OwnedTableFilter::bloom();
        let bloom_err = try_from_table_filter(bloom.as_ref(), &a, &scope).unwrap_err();
        assert!(
            bloom_err
                .to_string()
                .contains("bloom filter table filter is not supported"),
            "unexpected error: {bloom_err}"
        );
        let opt_bloom = OwnedTableFilter::optional(OwnedTableFilter::bloom());
        assert!(is_optional_table_filter(opt_bloom.as_ref()));
        assert_eq!(try_from_table_filter(opt_bloom.as_ref(), &a, &scope)?, None);

        // Optional wrapping a valid filter succeeds.
        let opt_valid = OwnedTableFilter::optional(OwnedTableFilter::is_not_null());
        assert_eq!(
            try_from_table_filter(opt_valid.as_ref(), &a, &scope)?,
            Some(is_not_null(a.clone()))
        );

        // ExpressionRef returning Some and None.
        let expr_cmp = OwnedTableFilter::expr_comparison(
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOTEQUAL,
            &Value::from(9i32),
            false,
        );
        assert_eq!(
            try_from_table_filter(expr_cmp.as_ref(), &a, &scope)?,
            Some(Compare.new_expr(
                CompareOperator::NotEq,
                [
                    a.clone(),
                    lit(Scalar::primitive(9i32, Nullability::Nullable))
                ]
            ))
        );
        let expr_unsupported = OwnedTableFilter::expr_unsupported();
        assert_eq!(
            try_from_table_filter(expr_unsupported.as_ref(), &a, &scope)?,
            None
        );

        Ok(())
    }

    #[rstest]
    #[case::eq(
    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL,
    [false, true, false, false]
)]
    #[case::neq(
    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOTEQUAL,
    [true, false, true, true]
)]
    #[case::lt(
    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN,
    [true, false, false, false]
)]
    #[case::lte(
    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO,
    [true, true, false, false]
)]
    #[case::gt(
    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN,
    [false, false, true, true]
)]
    #[case::gte(
    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO,
    [false, true, true, true]
)]
    fn dynamic_table_filter_evaluates_uninitialized_and_updated_values(
        #[case] op: DUCKDB_VX_EXPR_TYPE,
        #[case] expected_after_set: [bool; 4],
    ) -> VortexResult<()> {
        let scope = test_scope_dtype();
        let (filter, data) = OwnedTableFilter::dynamic(op, &Value::from(0i32));
        let expr = try_from_table_filter(filter.as_ref(), &col("a"), &scope)?
            .expect("dynamic filter converts");

        let batch =
            StructArray::try_from_iter([("a", buffer![1i32, 2, 3, 4].into_array())])?.into_array();
        let mut ctx = SESSION.create_execution_ctx();

        // Uninitialized dynamic filter lets every row pass.
        let before = batch.clone().apply(&expr)?;
        assert_arrays_eq!(
            before,
            BoolArray::from_iter([Some(true), Some(true), Some(true), Some(true)]),
            &mut ctx
        );

        // Setting the shared dynamic filter value takes effect on subsequent evaluations.
        data.set_value(&Value::from(2i32));
        let after = batch.apply(&expr)?;
        assert_arrays_eq!(
            after,
            BoolArray::from_iter(expected_after_set.map(Some)),
            &mut ctx
        );

        Ok(())
    }

    #[test]
    fn dynamic_filter_rejects_unsupported_operator_and_supports_v2_optional_wrapper()
    -> VortexResult<()> {
        let scope = test_scope_dtype();
        let (bad, _) = OwnedTableFilter::dynamic(
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
            &Value::from(0i32),
        );
        let bad_err = try_from_table_filter(bad.as_ref(), &col("a"), &scope).unwrap_err();
        assert!(
            bad_err
                .to_string()
                .contains("unsupported dynamic filter operator"),
            "unexpected error: {bad_err}"
        );

        let (expr_dyn, data) = OwnedTableFilter::expr_dynamic(
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO,
            &Value::from(0i32),
            true,
        );
        assert!(is_optional_table_filter(expr_dyn.as_ref()));
        let expr = try_from_table_filter(expr_dyn.as_ref(), &col("a"), &scope)?
            .expect("v2 optional dynamic filter converts");
        data.set_value(&Value::from(3i32));
        let batch =
            StructArray::try_from_iter([("a", buffer![1i32, 2, 3, 4].into_array())])?.into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            batch.apply(&expr)?,
            BoolArray::from_iter([Some(false), Some(false), Some(true), Some(true)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn try_from_virtual_column_filter_legacy_variants() -> VortexResult<()> {
        let in_f = OwnedTableFilter::in_filter(&[
            Value::from(9u64),
            Value::from(1u64),
            Value::from(4u64),
            Value::from(1u64),
        ]);
        let (sel, range) = try_from_virtual_column_filter(in_f.as_ref())?;
        assert!(range.is_none());
        match sel {
            Selection::IncludeByIndex(buf) => assert_eq!(buf.as_slice(), &[1, 4, 9]),
            other => panic!("expected IncludeByIndex, got {other:?}"),
        }

        let cmp = OwnedTableFilter::constant(
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO,
            &Value::from(10u64),
        );
        let (sel, range) = try_from_virtual_column_filter(cmp.as_ref())?;
        assert!(matches!(sel, Selection::All));
        assert_eq!(range, Some(10..u64::MAX));

        // ConjunctionAnd tightens range and intersects two InFilters.
        let conj = OwnedTableFilter::conjunction_and(vec![
            OwnedTableFilter::constant(
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO,
                &Value::from(2u64),
            ),
            OwnedTableFilter::constant(
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN,
                &Value::from(10u64),
            ),
            OwnedTableFilter::in_filter(&[Value::from(1u64), Value::from(3u64), Value::from(7u64)]),
            OwnedTableFilter::in_filter(&[Value::from(3u64), Value::from(5u64), Value::from(7u64)]),
        ]);
        let (sel, range) = try_from_virtual_column_filter(conj.as_ref())?;
        assert_eq!(range, Some(2..10));
        match sel {
            Selection::IncludeByIndex(buf) => assert_eq!(buf.as_slice(), &[3, 7]),
            other => panic!("expected IncludeByIndex, got {other:?}"),
        }

        // Contradictory bounds yield an empty range (0..0) rather than None (all rows).
        let empty_conj = OwnedTableFilter::conjunction_and(vec![
            OwnedTableFilter::constant(
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO,
                &Value::from(20u64),
            ),
            OwnedTableFilter::constant(
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN,
                &Value::from(10u64),
            ),
        ]);
        let (sel, range) = try_from_virtual_column_filter(empty_conj.as_ref())?;
        assert!(matches!(sel, Selection::All));
        assert_eq!(range, Some(0..0));

        // Optional wrapping valid vs erroring filter (negative integer).
        let opt_ok = OwnedTableFilter::optional(OwnedTableFilter::constant(
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL,
            &Value::from(3u64),
        ));
        assert_eq!(
            try_from_virtual_column_filter(opt_ok.as_ref())?.1,
            Some(3..4)
        );
        let opt_err = OwnedTableFilter::optional(OwnedTableFilter::constant(
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL,
            &Value::from(-1i64),
        ));
        assert_eq!(try_from_virtual_column_filter(opt_err.as_ref())?.1, None);

        // Unsupported filter variant falls back to (Selection::All, None).
        let (sel, range) = try_from_virtual_column_filter(OwnedTableFilter::is_null().as_ref())?;
        assert!(matches!(sel, Selection::All));
        assert_eq!(range, None);

        Ok(())
    }

    #[test]
    fn try_from_virtual_column_filter_v2_expression_ref_variants() -> VortexResult<()> {
        let u64_ty = LogicalType::uint64();

        // Optional(COMPARE_IN) with unsorted duplicates -> sorted, deduped IncludeByIndex.
        let expr_in = OwnedTableFilter::expr_in(
            &u64_ty,
            &[
                Value::from(8u64),
                Value::from(2u64),
                Value::from(8u64),
                Value::from(5u64),
            ],
            true,
        );
        let (sel, range) = try_from_virtual_column_filter(expr_in.as_ref())?;
        assert!(range.is_none());
        match sel {
            Selection::IncludeByIndex(buf) => assert_eq!(buf.as_slice(), &[2, 5, 8]),
            other => panic!("expected IncludeByIndex, got {other:?}"),
        }

        // Optional(COMPARE_IN) with empty list or negative value -> All/None.
        for bad_in in [
            OwnedTableFilter::expr_in(&u64_ty, &[], true),
            OwnedTableFilter::expr_in(&LogicalType::int64(), &[Value::from(-3i64)], true),
        ] {
            let (sel, range) = try_from_virtual_column_filter(bad_in.as_ref())?;
            assert!(matches!(sel, Selection::All));
            assert_eq!(range, None);
        }

        // Optional(BoundComparison) with constant -> row range; negative -> None.
        let expr_cmp = OwnedTableFilter::expr_comparison(
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO,
            &Value::from(15u64),
            true,
        );
        let (sel, range) = try_from_virtual_column_filter(expr_cmp.as_ref())?;
        assert!(matches!(sel, Selection::All));
        assert_eq!(range, Some(0..16));

        let expr_cmp_neg = OwnedTableFilter::expr_comparison(
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL,
            &Value::from(-1i64),
            true,
        );
        assert_eq!(
            try_from_virtual_column_filter(expr_cmp_neg.as_ref())?.1,
            None
        );

        // Non-optional dynamic ExpressionRef and unsupported operator -> All/None.
        let (dyn_no_opt, _) = OwnedTableFilter::expr_dynamic(
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL,
            &Value::from(1u64),
            false,
        );
        for ignored in [dyn_no_opt, OwnedTableFilter::expr_unsupported()] {
            let (sel, range) = try_from_virtual_column_filter(ignored.as_ref())?;
            assert!(matches!(sel, Selection::All));
            assert_eq!(range, None);
        }

        Ok(())
    }
}
