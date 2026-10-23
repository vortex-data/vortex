// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::expr::BoundExpression;
use crate::scalar_fn::ExpressionReduceNode;

impl BoundExpression {
    /// Optimize the root expression node only, iterating to convergence.
    ///
    /// This applies optimization rules repeatedly until no more changes occur:
    /// 1. `simplify` - scalar-function simplifications over the bound node
    /// 2. `reduce` - abstract reduction rules via `ReduceNode`
    pub fn optimize(&self) -> VortexResult<BoundExpression> {
        Ok(self.try_optimize()?.unwrap_or_else(|| self.clone()))
    }

    /// Apply this node's own simplification rule, if it has one.
    ///
    /// Non-scalar nodes carry no rules, so they never simplify.
    fn simplify_node(&self) -> VortexResult<Option<BoundExpression>> {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => scalar_fn.simplify(self),
            BoundExpression::Root { .. } => Ok(None),
        }
    }

    /// Apply this node's own abstract reduction rule, if it has one.
    fn reduce_node<'a>(
        &self,
        node: &ExpressionReduceNode<'a>,
    ) -> VortexResult<Option<ExpressionReduceNode<'a>>> {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => scalar_fn.reduce_expression(node),
            BoundExpression::Root { .. } => Ok(None),
        }
    }

    /// Try to optimize the root expression node only, returning None if no optimizations applied.
    fn try_optimize(&self) -> VortexResult<Option<BoundExpression>> {
        // Copy-on-write: `current` stays None until a rule fires, so unchanged nodes (the common
        // case) are never cloned.
        let mut current: Option<BoundExpression> = None;
        let mut loop_counter = 0;

        loop {
            if loop_counter > 100 {
                vortex_error::vortex_bail!(
                    "Exceeded maximum optimization iterations (possible infinite loop)"
                );
            }
            loop_counter += 1;

            let expr = current.as_ref().unwrap_or(self);
            let mut changed = false;

            if let Some(simplified) = expr.simplify_node()? {
                current = Some(simplified);
                changed = true;
            }

            // Try reduce via ReduceNode. The node borrows the expression, so constructing it is
            // free; the block scopes the borrow so `current` can be updated.
            let reduced = {
                let expr = current.as_ref().unwrap_or(self);
                expr.reduce_node(&ExpressionReduceNode::new(expr))?
                    .map(ExpressionReduceNode::into_expression)
            };
            if let Some(reduced_expr) = reduced {
                current = Some(reduced_expr);
                changed = true;
            }

            if !changed {
                break;
            }
        }

        Ok(current)
    }

    /// Optimize the entire expression tree recursively.
    ///
    /// Optimizes children first (bottom-up), then optimizes the root.
    pub fn optimize_recursive(&self) -> VortexResult<BoundExpression> {
        Ok(self
            .try_optimize_recursive()?
            .unwrap_or_else(|| self.clone()))
    }

    /// Try to optimize the entire expression tree recursively.
    pub fn try_optimize_recursive(&self) -> VortexResult<Option<BoundExpression>> {
        // First optimize the root
        let mut current = self.try_optimize()?;

        // Then recursively optimize children. The new children vector is only allocated once a
        // child actually changes, so fully-optimized subtrees cost no allocations.
        let expr = current.as_ref().unwrap_or(self);
        let children = expr.children();
        let mut new_children: Option<Vec<BoundExpression>> = None;
        for (idx, child) in children.iter().enumerate() {
            if let Some(optimized) = child.try_optimize_recursive()? {
                new_children
                    .get_or_insert_with(|| children[..idx].to_vec())
                    .push(optimized);
            } else if let Some(new_children) = new_children.as_mut() {
                new_children.push(child.clone());
            }
        }

        if let Some(new_children) = new_children {
            let updated = expr.clone().with_children(new_children)?;

            // After updating children, try to optimize root again
            current = Some(updated.try_optimize()?.unwrap_or(updated));
        }

        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::cast;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::lt_eq;
    use crate::expr::or;
    use crate::expr::root;
    use crate::scalar::Scalar;
    use crate::scalar_fn::builtin::literal::Literal;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn optimize_or_chain_correctness() -> VortexResult<()> {
        let expr = or(
            eq(get_item("x", root()), lit(1i32)),
            eq(get_item("x", root()), lit(2i32)),
        );
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let optimized = expr.bind(&scope)?.optimize_recursive()?;

        let data =
            StructArray::from_fields(&[("x", buffer![0i32, 1, 2, 3].into_array())])?.into_array();
        let actual = data.apply_bound(&optimized)?;
        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([false, true, true, false]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn optimize_folds_cast_of_literal_in_comparison() -> VortexResult<()> {
        let expr = lt_eq(
            get_item("x", root()),
            cast(
                lit(3i32),
                DType::Primitive(PType::F64, Nullability::NonNullable),
            ),
        );
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::F64, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let optimized = expr.bind(&scope)?.optimize_recursive()?;

        // Prune rules pattern-match a bare Literal on the comparison RHS; a cast wrapper
        // silently disables pruning.
        let rhs = optimized
            .child(1)
            .as_opt::<Literal>()
            .ok_or_else(|| vortex_err!("expected a bare literal RHS, got {optimized}"))?;
        assert_eq!(rhs, &Scalar::primitive(3.0f64, Nullability::NonNullable));
        Ok(())
    }

    #[rstest]
    #[case::and_annihilator(and(root(), lit(false)), DType::Bool(Nullability::Nullable))]
    #[case::or_annihilator(or(root(), lit(true)), DType::Bool(Nullability::Nullable))]
    #[case::and_nullable_identity(
        and(root(), lit(Scalar::from(Some(true)))),
        DType::Bool(Nullability::NonNullable)
    )]
    #[case::or_nullable_identity(
        or(root(), lit(Scalar::from(Some(false)))),
        DType::Bool(Nullability::NonNullable)
    )]
    fn literal_simplification_preserves_result_dtype(
        #[case] expr: Expression,
        #[case] scope: DType,
    ) -> VortexResult<()> {
        let original = expr.bind(&scope)?;
        let optimized = expr.bind(&scope)?.optimize_recursive()?;
        assert_eq!(optimized.dtype(), original.dtype());

        Ok(())
    }
}
