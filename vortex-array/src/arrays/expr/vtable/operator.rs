// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::expr::{ExprArray, ExprVTable};
use crate::expr::root;
use crate::expr::session::ExprSession;
use crate::expr::transform::ExprOptimizer;
use crate::vtable::OperatorVTable;

impl OperatorVTable<ExprVTable> for ExprVTable {
    fn reduce(array: &ExprArray) -> VortexResult<Option<ArrayRef>> {
        // Get the default expression session
        let session = ExprSession::default();
        let optimizer = ExprOptimizer::new(&session);

        // Try to optimize the expression with type information
        let optimized_expr =
            optimizer.optimize_typed(array.expr().clone(), array.child().dtype())?;

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
    use vortex_error::VortexExpect;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::{PrimitiveArray, PrimitiveVTable};
    use crate::expr::{get_item, pack, root};

    #[test]
    fn test_expr_array_reduce_pack_unpack() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);

        let expr = get_item("a", pack([("a", root())], Nullability::NonNullable));

        let expr_array = ExprArray::new_infer_dtype(array.into_array(), expr)?;

        // Call reduce - it should optimize pack(a: $).a to just $
        let reduced = expr_array.reduce()?.vortex_expect("reduce failed");

        assert!(reduced.is::<PrimitiveVTable>());

        Ok(())
    }
}
