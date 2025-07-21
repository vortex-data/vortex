// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::duckdb::{TableFilter, TableFilterClass};
use itertools::Itertools;
use vortex::error::{VortexResult, vortex_bail};
use vortex::expr::{BinaryExpr, ExprRef, and_collect, is_null, lit, not, or_collect};
use vortex::scalar::Scalar;

pub fn try_from_table_filter(value: &TableFilter, col: &ExprRef) -> VortexResult<Option<ExprRef>> {
    Ok(Some(match value.as_class() {
        TableFilterClass::ConstantComparison(const_) => {
            let scalar: Scalar = const_.value.try_into()?;
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
        TableFilterClass::StructExtract => {
            // Struct extract expressions are optional and not yet supported.
            vortex_bail!("struct extract table filter is not supported");
        }
        TableFilterClass::Optional(child) => {
            // Optional expressions are optional not yet supported.
            return try_from_table_filter(&child, col).or_else(|_err| {
                // Failed to convert the optional expression, but it's optional, so who cares?
                println!("Failed to convert optional table filter: {:?}", child);
                Ok(None)
            });
        }
        TableFilterClass::InFilter => {
            vortex_bail!("in filter table filter is not supported");
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
