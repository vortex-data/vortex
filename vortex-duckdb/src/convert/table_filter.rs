// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex::dtype::Nullability;
use vortex::error::{VortexResult, vortex_bail};
use vortex::expr::{
    BinaryExpr, ExprRef, and_collect, get_item, is_null, list_contains, lit, not, or_collect,
};
use vortex::scalar::Scalar;

use crate::duckdb::{TableFilter, TableFilterClass};

pub fn try_from_table_filter(value: &TableFilter, col: &ExprRef) -> VortexResult<Option<ExprRef>> {
    Ok(Some(match value.as_class() {
        TableFilterClass::ConstantComparison(const_) => {
            let scalar: Scalar = (&const_.value).try_into()?;
            BinaryExpr::new_expr(col.clone(), const_.operator.try_into()?, lit(scalar))
        }
        TableFilterClass::ConjunctionAnd(conj_and) => {
            let Some(children) = conj_and
                .children()
                .map(|child| try_from_table_filter(&child, col))
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
                .map(|child| try_from_table_filter(&child, col))
                .try_collect::<_, Option<Vec<_>>, _>()?
            else {
                return Ok(None);
            };

            or_collect(children).unwrap_or_else(|| lit(false))
        }
        TableFilterClass::IsNull => is_null(col.clone()),
        TableFilterClass::IsNotNull => not(is_null(col.clone())),
        TableFilterClass::StructExtract(name, child_filter) => {
            return try_from_table_filter(&child_filter, &get_item(name, col.clone()));
        }
        TableFilterClass::Optional(child) => {
            // Optional expressions are optional not yet supported.
            return try_from_table_filter(&child, col).or_else(|_err| {
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
            let list_scalar = Scalar::list(Arc::new(dtype), scalars, Nullability::Nullable);
            list_contains(lit(list_scalar), col.clone())
        }
        TableFilterClass::Dynamic(_dynamic) => {
            // Dynamic expressions are optional and not yet supported.
            vortex_bail!("dynamic table filter is not supported");
        }
        TableFilterClass::Expression(expr) => {
            // TODO(ngates): figure out which column ID DuckDB is using for the expression.
            vortex_bail!("expression table filter is not supported: {}", expr);
        }
    }))
}
