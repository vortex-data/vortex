// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use itertools::Itertools;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::get_item;
use vortex::expr::is_not_null;
use vortex::expr::is_null;
use vortex::expr::list_contains;
use vortex::expr::lit;
use vortex::expr::or_collect;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::operators::CompareOperator;
use vortex::scan::selection::Selection;

use super::expr::try_from_bound_expression_with_col_sub;
use crate::cpp::DUCKDB_VX_EXPR_TYPE;
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

            Binary.new_expr(const_.operator.try_into()?, [col.clone(), lit(scalar)])
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
            // TODO(ngates): I'm pretty sure we actually need this as ScalarValue with the
            //  scope dtype
            let scalars: Vec<_> = values.iter().map(Scalar::try_from).try_collect()?;
            assert!(
                !scalars.is_empty(),
                "IN filter must have at least one value"
            );
            let dtype = scalars[0].dtype().clone();
            let list_scalar = Scalar::list(Arc::new(dtype), scalars, Nullability::Nullable);
            list_contains(lit(list_scalar), col.clone())
        }
        TableFilterClass::Dynamic(dynamic) => {
            let op = match dynamic.operator {
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL => CompareOperator::Eq,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOTEQUAL => CompareOperator::NotEq,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN => CompareOperator::Lt,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN => CompareOperator::Gt,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO => {
                    CompareOperator::Lte
                }
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO => {
                    CompareOperator::Gte
                }
                _ => vortex_bail!(
                    "unsupported dynamic filter operator: {:?}",
                    dynamic.operator
                ),
            };
            let data = dynamic.data;

            vortex::expr::dynamic(
                op,
                move || {
                    let value = data.latest()?;
                    let scalar = Scalar::try_from(&*value)
                        .vortex_expect("failed to convert dynamic filter value to scalar");
                    scalar.into_value()
                },
                col.return_dtype(scope_dtype)?,
                true, // If there is no value, we say that all rows pass the dynamic filter.
                col.clone(),
            )
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

/// For constant comparison on IN filters over file_index or file_row_number
/// virtual column, create a selection and a range covering the same range as
/// expressions do.
pub fn try_from_virtual_column_filter(
    filter: &TableFilterRef,
) -> VortexResult<(Selection, Option<Range<u64>>)> {
    match filter.as_class() {
        TableFilterClass::InFilter(values) => {
            let indices = values
                .iter()
                .map(nonnegative_number_from_value)
                .collect::<VortexResult<Vec<u64>>>()?;
            Ok((Selection::IncludeByIndex(Buffer::from_iter(indices)), None))
        }
        TableFilterClass::ConstantComparison(const_) => {
            let n = nonnegative_number_from_value(const_.value)?;
            let range = match const_.operator {
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL => Some(n..n + 1),
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO => {
                    Some(n..u64::MAX)
                }
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN => {
                    Some(n.saturating_add(1)..u64::MAX)
                }
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO => {
                    Some(0..n.saturating_add(1))
                }
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN => Some(0..n),
                _ => None,
            };
            Ok((Selection::All, range))
        }
        TableFilterClass::ConjunctionAnd(conj) => {
            let mut start = 0u64;
            let mut end = u64::MAX;
            let mut indices: Option<Vec<u64>> = None;
            for child in conj.children() {
                let (sel, range) = try_from_virtual_column_filter(child)?;
                if let Selection::IncludeByIndex(buf) = sel {
                    indices = Some(match indices {
                        None => buf.iter().copied().collect(),
                        Some(existing) => intersect_sorted(&existing, buf.as_ref()),
                    });
                }
                if let Some(r) = range {
                    start = start.max(r.start);
                    end = end.min(r.end);
                }
            }
            let range = (start < end).then_some(start..end);
            let sel = indices
                .map(|v| Selection::IncludeByIndex(Buffer::from_iter(v)))
                .unwrap_or(Selection::All);
            Ok((sel, range))
        }
        TableFilterClass::Optional(child) => {
            try_from_virtual_column_filter(child).or_else(|_| Ok((Selection::All, None)))
        }
        _ => Ok((Selection::All, None)),
    }
}
