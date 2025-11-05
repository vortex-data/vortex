// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex::compute::Operator;
use vortex::dtype::{DType, Nullability};
use vortex::error::{VortexExpect, VortexResult, vortex_bail};
use vortex::expr::{
    Binary, Expression, VTableExt, and_collect, get_item, is_null, list_contains, lit, not,
    or_collect,
};
use vortex::scalar::Scalar;

use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb::{TableFilter, TableFilterClass};

pub fn try_from_table_filter(
    value: &TableFilter,
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
                .map(|child| try_from_table_filter(&child, col, scope_dtype))
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
                .map(|child| try_from_table_filter(&child, col, scope_dtype))
                .try_collect::<_, Option<Vec<_>>, _>()?
            else {
                return Ok(None);
            };

            or_collect(children).unwrap_or_else(|| lit(false))
        }
        TableFilterClass::IsNull => is_null(col.clone()),
        TableFilterClass::IsNotNull => not(is_null(col.clone())),
        TableFilterClass::StructExtract(name, child_filter) => {
            return try_from_table_filter(&child_filter, &get_item(name, col.clone()), scope_dtype);
        }
        TableFilterClass::Optional(child) => {
            // Optional expressions are optional not yet supported.
            return try_from_table_filter(&child, col, scope_dtype).or_else(|_err| {
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
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL => Operator::Eq,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOTEQUAL => Operator::NotEq,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN => Operator::Lt,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN => Operator::Gt,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO => Operator::Lte,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO => {
                    Operator::Gte
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
                    let scalar = Scalar::try_from(value.as_ref())
                        .vortex_expect("failed to convert dynamic filter value to scalar");
                    Some(scalar.into_value())
                },
                col.return_dtype(scope_dtype)?,
                true, // If there is no value, we say that all rows pass the dynamic filter.
                col.clone(),
            )
        }
        TableFilterClass::Expression(expr) => {
            // TODO(ngates): figure out which column ID DuckDB is using for the expression.
            vortex_bail!("expression table filter is not supported: {}", expr);
        }
    }))
}
