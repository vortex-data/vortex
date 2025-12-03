// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_compute::logical::LogicalNot;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Vector;

use crate::ArrayRef;
use crate::compute::invert;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::ExpressionView;
use crate::expr::ScalarFnExprExt;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::functions::EmptyOptions;
use crate::scalar_fns::not;

/// Expression that logically inverts boolean values.
pub struct Not;

impl VTable for Not {
    type Instance = ();

    fn id(&self) -> ExprId {
        ExprId::new_ref("vortex.not")
    }

    fn serialize(&self, _instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        Ok(Some(()))
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if expr.children().len() != 1 {
            vortex_bail!(
                "Not expression expects exactly one child, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Not expression", child_idx),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "not(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let child_dtype = expr.child(0).return_dtype(scope)?;
        if !matches!(child_dtype, DType::Bool(_)) {
            vortex_bail!(
                "Not expression expects a boolean child, got: {}",
                child_dtype
            );
        }
        Ok(child_dtype)
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let child_result = expr.child(0).evaluate(scope)?;
        invert(&child_result)
    }

    fn execute(&self, _data: &Self::Instance, mut args: ExecutionArgs) -> VortexResult<Vector> {
        let child = args.vectors.pop().vortex_expect("Missing input child");
        Ok(child.into_bool().not().into())
    }

    fn is_null_sensitive(&self, _instance: &Self::Instance) -> bool {
        false
    }

    fn is_fallible(&self, _instance: &Self::Instance) -> bool {
        false
    }

    fn expr_v2(&self, view: &ExpressionView<Self>) -> VortexResult<Expression> {
        ScalarFnExprExt::try_new_expr(&not::NotFn, EmptyOptions, view.children().clone())
    }
}

/// Creates an expression that logically inverts boolean values.
///
/// Returns the logical negation of the input boolean expression.
///
/// ```rust
/// # use vortex_array::expr::{not, root};
/// let expr = not(root());
/// ```
pub fn not(operand: Expression) -> Expression {
    Not.new_expr((), vec![operand])
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;

    use super::not;
    use crate::ToCanonical;
    use crate::arrays::BoolArray;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::root::root;
    use crate::expr::test_harness;

    #[test]
    fn invert_booleans() {
        let not_expr = not(root());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        assert_eq!(
            not_expr
                .evaluate(&bools.to_array())
                .unwrap()
                .to_bool()
                .bit_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, false, true, true, false, false]
        );
    }

    #[test]
    fn test_display_order_of_operations() {
        let a = not(get_item("a", root()));
        let b = get_item("a", not(root()));
        assert_ne!(a.to_string(), b.to_string());
        assert_eq!(a.to_string(), "not($.a)");
        assert_eq!(b.to_string(), "not($).a");
    }

    #[test]
    fn dtype() {
        let not_expr = not(root());
        let dtype = DType::Bool(Nullability::NonNullable);
        assert_eq!(
            not_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let dtype = test_harness::struct_dtype();
        assert_eq!(
            not(col("bool1")).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }
}
