// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::OptimizerRule;
use crate::expr::BoundExpression;
use crate::expr::ExpressionId;
use crate::expr::bound;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::literal::Literal;

/// Removes identity casts and evaluates casts of literal values during optimization.
///
/// # Example
///
/// ```text
/// original: cast(lit(1_i32), i64)
/// rewritten: lit(1_i64)
/// ```
#[derive(Debug)]
pub(crate) struct CastLiteralOrIdentity;

impl OptimizerRule for CastLiteralOrIdentity {
    fn expression_id(&self) -> ExpressionId {
        Cast.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        let target = expr.as_::<Cast>();
        let child = expr.child(0);
        if child.dtype() == target {
            return Ok(Some(child.clone()));
        }
        let Some(scalar) = child.as_opt::<Literal>() else {
            return Ok(None);
        };
        Ok(scalar.cast(target).ok().map(bound::lit))
    }
}
