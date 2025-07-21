// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::duckdb::{TableFilter, TableFilterClass};
use itertools::Itertools;
use std::sync::Arc;
use vortex::dtype::{DType, Nullability};
use vortex::error::{VortexResult, vortex_bail};
use vortex::expr::{
    BinaryExpr, ExprRef, and_collect, get_item, is_null, list_contains, lit, not, or_collect,
};
use vortex::scalar::Scalar;

pub fn try_from_table_filter(
    value: &TableFilter,
    col: &ExprRef,
    scope_dtype: &DType,
) -> VortexResult<Option<ExprRef>> {
    Ok(Some(match value.as_class() {
        TableFilterClass::ConstantComparison(const_) => {
            let scalar: Scalar = (&const_.value).try_into()?;
            BinaryExpr::new_expr(col.clone(), const_.operator.try_into()?, lit(scalar))
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
                println!("Failed to convert optional table filter: {:?}", child);
                Ok(None)
            });
        }
        TableFilterClass::InFilter(values) => {
            let scalars: Vec<_> = values.iter().map(|v| Scalar::try_from(v)).try_collect()?;
            let dtype = col.return_dtype(scope_dtype)?;
            let list_scalar = Scalar::list(Arc::new(dtype), scalars, Nullability::Nullable);
            list_contains(lit(list_scalar), col.clone())
        }
        TableFilterClass::Dynamic(_) => {
            // Dynamic expressions are optional and not yet supported.
            return Ok(None);
        }
        TableFilterClass::Expression(expr) => {
            // TODO(ngates): figure out which column ID DuckDB is using for the expression.
            vortex_bail!("expression table filter is not supported: {}", expr);
        }
    }))
}
