// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::expr::ExprArray;
use crate::arrays::expr::ExprVTable;
use crate::expr::root;
use crate::transform::ArrayReduceRule;
use crate::transform::ArrayRuleContext;
use crate::vtable::OperatorVTable;

impl OperatorVTable<ExprVTable> for ExprVTable {}

/// Rule to optimize expressions within ExprArrays.
#[derive(Default, Debug)]
pub struct ExprOptimizationRule;

impl ArrayReduceRule<ExprVTable> for ExprOptimizationRule {
    fn reduce(&self, array: &ExprArray, ctx: &ArrayRuleContext) -> VortexResult<Option<ArrayRef>> {
        // Try to optimize the expression with type information
        let optimized_expr = ctx
            .expr_optimizer()
            .optimize_typed(array.expr().clone(), array.child().dtype())?;

        if optimized_expr != *array.expr() {
            // If the expression simplified to just root(), return the child directly
            if optimized_expr == root() {
                return Ok(Some(array.child().clone()));
            }

            let new_dtype = optimized_expr.return_dtype(array.child().dtype())?;
            Ok(Some(
                ExprArray::try_new(array.child().clone(), optimized_expr, new_dtype)?.into(),
            ))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability;

    use super::*;
    use crate::ArraySession;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::PrimitiveVTable;
    use crate::expr::get_item;
    use crate::expr::pack;
    use crate::expr::root;
    use crate::expr::session::ExprSession;
    use crate::expr::transform::ExprOptimizer;

    #[test]
    fn test_expr_array_reduce_pack_unpack() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);

        let expr = get_item("a", pack([("a", root())], Nullability::NonNullable));

        let expr_array = ExprArray::new_infer_dtype(array.into_array(), expr)?;

        // Use the optimizer to optimize the expression array
        let array_session = ArraySession::default();
        let expr_session = ExprSession::default();
        let expr_optimizer = ExprOptimizer::new(&expr_session);
        let optimizer = array_session.optimizer(expr_optimizer);

        let reduced = optimizer.optimize_array(expr_array.into_array())?;

        assert!(reduced.is::<PrimitiveVTable>());

        Ok(())
    }
}
