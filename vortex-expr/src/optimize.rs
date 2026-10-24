// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::DynReduceNode;
use vortex_array::scalar_fn::ErasedReduceNode;
use vortex_array::scalar_fn::ReduceNodeValidity;
use vortex_array::scalar_fn::ScalarFnRef;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::BoundExpression;
use crate::bound;

/// The [`ErasedReduceNode`] over a bound expression tree.
///
/// Reduction rules in `vortex-array` are written once against `ReduceNode`; this node lets them
/// run over expressions through [`DynReduceNode`].
#[derive(Clone)]
pub struct ExpressionReduceNode(BoundExpression);

impl ExpressionReduceNode {
    /// Creates a node over the given bound expression.
    pub fn node(expression: BoundExpression) -> DynReduceNode {
        DynReduceNode::new(Self(expression))
    }

    /// Returns the bound expression backing a reduced node.
    pub fn expression(node: &DynReduceNode) -> &BoundExpression {
        &node
            .downcast_ref::<Self>()
            .vortex_expect("expression trees reduce to expression nodes")
            .0
    }
}

impl ErasedReduceNode for ExpressionReduceNode {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn node_dtype(&self) -> VortexResult<DType> {
        Ok(self.0.dtype().clone())
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        self.0.as_scalar()
    }

    fn child(&self, idx: usize) -> DynReduceNode {
        Self::node(self.0.child(idx).clone())
    }

    fn child_count(&self) -> usize {
        self.0.children().len()
    }

    fn new_node(
        &self,
        scalar_fn: ScalarFnRef,
        children: &[DynReduceNode],
    ) -> VortexResult<DynReduceNode> {
        let expression = BoundExpression::try_new(
            scalar_fn,
            children.iter().map(|c| Self::expression(c).clone()),
        )?;
        Ok(Self::node(expression))
    }

    fn as_constant(&self) -> Option<Scalar> {
        self.0.as_opt::<Literal>().cloned()
    }

    fn new_constant(&self, value: Scalar) -> DynReduceNode {
        Self::node(bound::lit(value))
    }

    fn validity(&self, node: &DynReduceNode) -> VortexResult<ReduceNodeValidity<DynReduceNode>> {
        match self.0.as_scalar() {
            Some(scalar_fn) => scalar_fn.validity_dyn(node),
            None => Ok(ReduceNodeValidity::Irreducible),
        }
    }

    fn is_same(&self, other: &DynReduceNode) -> bool {
        self.0 == *Self::expression(other)
    }
}

impl BoundExpression {
    /// Optimize the root expression node only, iterating to convergence.
    ///
    /// This applies optimization rules repeatedly until no more changes occur:
    /// 1. `simplify` - scalar-function simplifications over the node
    /// 2. `reduce` - abstract reduction rules via `ReduceNode`
    pub fn optimize(&self) -> VortexResult<BoundExpression> {
        Ok(self.try_optimize()?.unwrap_or_else(|| self.clone()))
    }

    /// Apply this node's own simplification rule, if it has one.
    ///
    /// Non-scalar nodes carry no rules, so they never simplify.
    fn simplify_node(&self) -> VortexResult<Option<BoundExpression>> {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => Ok(scalar_fn
                .simplify_dyn(&ExpressionReduceNode::node(self.clone()))?
                .map(|simplified| ExpressionReduceNode::expression(&simplified).clone())),
            BoundExpression::Root { .. } => Ok(None),
        }
    }

    /// Apply this node's own abstract reduction rule, if it has one.
    fn reduce_node(&self) -> VortexResult<Option<BoundExpression>> {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => Ok(scalar_fn
                .reduce_dyn(&ExpressionReduceNode::node(self.clone()))?
                .map(|reduced| ExpressionReduceNode::expression(&reduced).clone())),
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

            let expr = current.as_ref().unwrap_or(self);
            if let Some(reduced) = expr.reduce_node()? {
                current = Some(reduced);
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
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::builtin::literal::Literal;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::ArrayExprExt;
    use crate::Expression;
    use crate::and;
    use crate::cast;
    use crate::eq;
    use crate::get_item;
    use crate::lit;
    use crate::lt_eq;
    use crate::or;
    use crate::root;

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
