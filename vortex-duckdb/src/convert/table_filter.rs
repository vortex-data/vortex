// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use itertools::Itertools;
use tracing::debug;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::ScalarFnExprExt;
use vortex::expr::and_collect;
use vortex::expr::get_item;
use vortex::expr::in_list_of;
use vortex::expr::is_not_null;
use vortex::expr::is_null;
use vortex::expr::lit;
use vortex::expr::or_collect;
use vortex::scalar::Scalar;
use vortex::scalar_fn::builtin::compare::Compare;
use vortex::scalar_fn::builtin::operators::CompareOperator;
use vortex::scan::selection::Selection;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;

use super::expr::try_from_bound_expression_with_col_sub;
use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb::DynamicFilter;
use crate::duckdb::ExpressionClass;
use crate::duckdb::ExpressionRef;
use crate::duckdb::ExtractedValue;
use crate::duckdb::TableFilterClass;
use crate::duckdb::TableFilterRef;
use crate::duckdb::ValueRef;

pub fn try_from_table_filter(
    value: &TableFilterRef,
    col: &Expression,
    scope_dtype: &DType,
) -> VortexResult<Option<Expression>> {
    Ok(Some(match value.as_class() {
        TableFilterClass::ConstantComparison(const_) => {
            let scalar: Scalar = const_.value.try_into()?;

            Compare.new_expr(const_.operator.try_into()?, [col.clone(), lit(scalar)])
        }
        TableFilterClass::ConjunctionAnd(conj_and) => {
            let Some(children) = conj_and
                .children()
                .map(|child| try_from_table_filter(child, col, scope_dtype))
                .try_collect::<_, Option<Vec<_>>, _>()?
            else {
                return Ok(None);
            };

            and_collect(children).unwrap_or_else(|| lit(true))
        }
        // This is a disjunction.
        TableFilterClass::ConjunctionOr(disjuction_or) => {
            let Some(children) = disjuction_or
                .children()
                .map(|child| try_from_table_filter(child, col, scope_dtype))
                .try_collect::<_, Option<Vec<_>>, _>()?
            else {
                return Ok(None);
            };

            or_collect(children).unwrap_or_else(|| lit(false))
        }
        TableFilterClass::IsNull => is_null(col.clone()),
        TableFilterClass::IsNotNull => is_not_null(col.clone()),
        TableFilterClass::StructExtract(name, child_filter) => {
            return try_from_table_filter(child_filter, &get_item(name, col.clone()), scope_dtype);
        }
        TableFilterClass::Optional(child) => {
            // Optional expressions are optional not yet supported.
            return try_from_table_filter(child, col, scope_dtype).or_else(|_err| {
                // Failed to convert the optional expression, but it's optional, so who cares?
                Ok(None)
            });
        }
        TableFilterClass::InFilter(values) => {
            let scalars: Vec<_> = values.iter().map(Scalar::try_from).try_collect()?;
            assert!(
                !scalars.is_empty(),
                "IN filter must have at least one value"
            );
            let dtype = scalars[0].dtype().clone();
            in_list_of(col.clone(), dtype, scalars)?
        }
        TableFilterClass::Dynamic(dynamic) => {
            dynamic_filter_expr(dynamic, col.return_dtype(scope_dtype)?, col.clone())?
        }
        TableFilterClass::ExpressionRef(expr) => {
            match try_from_bound_expression_with_col_sub(expr, col)? {
                Some(expression) => expression,
                None => return Ok(None),
            }
        }
        TableFilterClass::Bloom => {
            vortex_bail!("bloom filter table filter is not supported")
        }
    }))
}

/// A filter expression that rereads shared dynamic filter value on every
/// evaluation.
pub(super) fn dynamic_filter_expr(
    dynamic: DynamicFilter,
    dtype: DType,
    child: Expression,
) -> VortexResult<Expression> {
    let op = match dynamic.operator {
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL => CompareOperator::Eq,
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOTEQUAL => CompareOperator::NotEq,
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN => CompareOperator::Lt,
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN => CompareOperator::Gt,
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO => CompareOperator::Lte,
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO => {
            CompareOperator::Gte
        }
        _ => {
            vortex_bail!("unsupported dynamic filter operator {:?}", dynamic.operator)
        }
    };
    let data = dynamic.data;

    Ok(vortex::expr::dynamic(
        op,
        move || {
            let value = data.latest()?;
            let scalar = Scalar::try_from(&*value)
                .vortex_expect("failed to convert dynamic filter value to scalar");
            scalar.into_value()
        },
        dtype,
        true, // If there is no value, we say that all rows pass the dynamic filter.
        child,
    ))
}

fn nonnegative_number_from_value(value: &ValueRef) -> VortexResult<u64> {
    match value.extract() {
        ExtractedValue::BigInt(i) => {
            u64::try_from(i).map_err(|_| vortex_err!("negative value: {i}"))
        }
        ExtractedValue::Integer(i) => {
            u64::try_from(i).map_err(|_| vortex_err!("negative value: {i}"))
        }
        ExtractedValue::UBigInt(u) => Ok(u),
        ExtractedValue::UInteger(u) => Ok(u64::from(u)),
        _ => vortex_bail!("unexpected value type"),
    }
}

fn row_range_from_comparison(operator: DUCKDB_VX_EXPR_TYPE, n: u64) -> Option<Range<u64>> {
    match operator {
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL => Some(n..n + 1),
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO => Some(n..u64::MAX),
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN => {
            Some(n.saturating_add(1)..u64::MAX)
        }
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO => {
            Some(0..n.saturating_add(1))
        }
        DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN => Some(0..n),
        _ => None,
    }
}

fn intersect_sorted(left: &[u64], right: &[u64]) -> Vec<u64> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Equal => {
                result.push(left[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    result
}

/// Since 2.0 some filters are ExpressionRef instead of TableFilterRef
fn try_from_v2_virtual_column_filter(
    expr: &ExpressionRef,
) -> VortexResult<(Selection, Option<Range<u64>>)> {
    let Some(ExpressionClass::BoundFunction(bfn)) = expr.as_class() else {
        return Ok((Selection::All, None));
    };
    let Some(inner) = bfn.optional() else {
        return Ok((Selection::All, None));
    };
    match inner.as_class() {
        Some(ExpressionClass::BoundOperator(op))
            if op.op == DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_IN =>
        {
            let indices = op
                .children()
                .skip(1) // first child is the column which we know already
                .map(|child| match child.as_class() {
                    Some(ExpressionClass::BoundConstant(constant)) => {
                        nonnegative_number_from_value(constant.value).ok()
                    }
                    _ => None,
                })
                .collect::<Option<Vec<u64>>>();
            let Some(mut indices) = indices.filter(|indices| !indices.is_empty()) else {
                return Ok((Selection::All, None));
            };
            indices.sort_unstable();
            indices.dedup();
            Ok((
                Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from_iter(indices))?),
                None,
            ))
        }
        Some(ExpressionClass::BoundComparison(cmp)) => {
            if let Some(ExpressionClass::BoundConstant(constant)) = cmp.right.as_class()
                && let Ok(n) = nonnegative_number_from_value(constant.value)
            {
                return Ok((Selection::All, row_range_from_comparison(cmp.op, n)));
            }
            Ok((Selection::All, None))
        }
        _ => Ok((Selection::All, None)),
    }
}

/// For constant comparison on IN filters over file_index or file_row_number
/// virtual column, create a selection and a range covering the same range as
/// expressions do.
pub fn try_from_virtual_column_filter(
    filter: &TableFilterRef,
) -> VortexResult<(Selection, Option<Range<u64>>)> {
    match filter.as_class() {
        TableFilterClass::ExpressionRef(expr) => try_from_v2_virtual_column_filter(expr),
        TableFilterClass::InFilter(values) => {
            let indices = values
                .iter()
                .map(nonnegative_number_from_value)
                .collect::<VortexResult<Vec<u64>>>()?;
            Ok((
                Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from_iter(indices))?),
                None,
            ))
        }
        TableFilterClass::ConstantComparison(const_) => {
            let n = nonnegative_number_from_value(const_.value)?;
            Ok((
                Selection::All,
                row_range_from_comparison(const_.operator, n),
            ))
        }
        TableFilterClass::ConjunctionAnd(conj) => {
            let mut start = 0u64;
            let mut end = u64::MAX;
            let mut indices: Option<Vec<u64>> = None;
            for child in conj.children() {
                let (sel, range) = try_from_virtual_column_filter(child)?;
                if let Selection::IncludeByIndex(buf) = sel {
                    indices = Some(match indices {
                        None => buf.as_slice().to_vec(),
                        Some(existing) => intersect_sorted(&existing, buf.as_slice()),
                    });
                }
                if let Some(r) = range {
                    start = start.max(r.start);
                    end = end.min(r.end);
                }
            }
            let range = (start < end).then_some(start..end);
            let sel = indices
                .map(|v| {
                    StrictSortedBuffer::try_new(Buffer::from_iter(v)).map(Selection::IncludeByIndex)
                })
                .transpose()?
                .unwrap_or(Selection::All);
            Ok((sel, range))
        }
        TableFilterClass::Optional(child) => try_from_virtual_column_filter(child).or_else(|_| {
            debug!("rejecting unknown optional filter {child:?}");
            Ok((Selection::All, None))
        }),
        _ => {
            debug!("rejecting unknown filter {filter:?}");
            Ok((Selection::All, None))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex::error::VortexResult;

    use super::intersect_sorted;
    use super::nonnegative_number_from_value;
    use super::row_range_from_comparison;
    use crate::cpp::DUCKDB_VX_EXPR_TYPE;
    use crate::duckdb::Value;

    #[test]
    fn nonnegative_number_from_value_handles_signed_unsigned_negative_and_invalid()
    -> VortexResult<()> {
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
    fn intersect_sorted_cases(
        #[case] left: &[u64],
        #[case] right: &[u64],
        #[case] expected: Vec<u64>,
    ) {
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
        fn try_from_table_filter_constant_null_conjunctions_struct_in_and_bloom() -> VortexResult<()>
        {
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

            let batch = StructArray::try_from_iter([("a", buffer![1i32, 2, 3, 4].into_array())])?
                .into_array();
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
            let batch = StructArray::try_from_iter([("a", buffer![1i32, 2, 3, 4].into_array())])?
                .into_array();
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
                Value::from(1u64),
                Value::from(4u64),
                Value::from(9u64),
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
                OwnedTableFilter::in_filter(&[
                    Value::from(1u64),
                    Value::from(3u64),
                    Value::from(7u64),
                ]),
                OwnedTableFilter::in_filter(&[
                    Value::from(3u64),
                    Value::from(5u64),
                    Value::from(7u64),
                ]),
            ]);
            let (sel, range) = try_from_virtual_column_filter(conj.as_ref())?;
            assert_eq!(range, Some(2..10));
            match sel {
                Selection::IncludeByIndex(buf) => assert_eq!(buf.as_slice(), &[3, 7]),
                other => panic!("expected IncludeByIndex, got {other:?}"),
            }

            // Contradictory bounds yield range = None.
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
            assert_eq!(range, None);

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
            let (sel, range) =
                try_from_virtual_column_filter(OwnedTableFilter::is_null().as_ref())?;
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
}
