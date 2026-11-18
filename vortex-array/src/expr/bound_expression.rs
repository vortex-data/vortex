// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::expression::unbound_variable;
use crate::expr::scope::Scope;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExpressionReduceNode;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::is_not_null::IsNotNull;

/// An [`Expression`] that has been type-checked against a [`Scope`].
///
/// Scalar nodes obtain their dtype by checking their arguments with the scalar function's
/// `return_dtype`. Every node stores its dtype so callers can read it without walking the tree.
///
/// Binding is purely logical: it deals only in [`DType`]s and never sees an array, a length, or an
/// encoding.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BoundExpression {
    node: BoundExpressionNode,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum BoundExpressionNode {
    Scalar {
        dtype: DType,
        scalar_fn: ScalarFnRef,
        // Sharing keeps clones cheap and lets iterative drop take uniquely owned children.
        children: Arc<Vec<BoundExpression>>,
    },
    Root {
        dtype: DType,
    },
}

/// A bound-expression wrapper that compares shared tree identity instead of structure.
#[derive(Clone, Debug)]
pub struct ExactBoundExpr(pub BoundExpression);

impl PartialEq for ExactBoundExpr {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0.node, &other.0.node) {
            (
                BoundExpressionNode::Root { dtype: lhs_dtype },
                BoundExpressionNode::Root { dtype: rhs_dtype },
            ) => lhs_dtype == rhs_dtype,
            (
                BoundExpressionNode::Scalar {
                    dtype: lhs_dtype,
                    scalar_fn: lhs_fn,
                    children: lhs_children,
                },
                BoundExpressionNode::Scalar {
                    dtype: rhs_dtype,
                    scalar_fn: rhs_fn,
                    children: rhs_children,
                },
            ) => {
                lhs_fn == rhs_fn
                    && Arc::ptr_eq(lhs_children, rhs_children)
                    && lhs_dtype == rhs_dtype
            }
            _ => false,
        }
    }
}

impl Eq for ExactBoundExpr {}

impl Hash for ExactBoundExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // DType differences are resolved by equality. Omitting the potentially lazy dtype keeps
        // identity-keyed cache lookups from deserializing an entire schema just to compute a hash.
        match &self.0.node {
            BoundExpressionNode::Root { .. } => state.write_u8(0),
            BoundExpressionNode::Scalar {
                scalar_fn,
                children,
                ..
            } => {
                state.write_u8(1);
                scalar_fn.hash(state);
                Arc::as_ptr(children).hash(state);
            }
        }
    }
}

impl BoundExpression {
    /// Create a bound root expression with the given dtype.
    pub fn new_root(dtype: DType) -> Self {
        Self {
            node: BoundExpressionNode::Root { dtype },
        }
    }

    /// Create a bound scalar node from a scalar function and already-bound children.
    ///
    /// Checks arity, binds the scalar function's options to the argument dtypes, and computes the
    /// result dtype with the scalar function's `return_dtype`. Returns any error from these checks.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        Self::try_new_vec(scalar_fn, children.into_iter().collect())
    }

    fn try_new_vec(scalar_fn: ScalarFnRef, children: Vec<BoundExpression>) -> VortexResult<Self> {
        vortex_ensure!(
            scalar_fn.signature().arity().matches(children.len()),
            "Expression arity mismatch: expected {} children but got {}",
            scalar_fn.signature().arity(),
            children.len()
        );

        let arg_dtypes = children
            .iter()
            .map(|child| child.dtype().clone())
            .collect_vec();
        let scalar_fn = scalar_fn.bind_options(&arg_dtypes)?;
        let dtype = scalar_fn.return_dtype(&arg_dtypes)?;

        Ok(Self {
            node: BoundExpressionNode::Scalar {
                dtype,
                scalar_fn,
                children: children.into(),
            },
        })
    }

    /// Rebuild this node with new bound children, recomputing its dtype.
    pub fn with_children(
        self,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        let Some(scalar_fn) = self.as_scalar() else {
            vortex_ensure!(
                children.is_empty(),
                "Root expression cannot have {} children",
                children.len()
            );
            return Ok(self);
        };

        Self::try_new_vec(scalar_fn.clone(), children)
    }

    /// The dtype this expression evaluates to.
    pub fn dtype(&self) -> &DType {
        match &self.node {
            BoundExpressionNode::Scalar { dtype, .. } | BoundExpressionNode::Root { dtype } => {
                dtype
            }
        }
    }

    /// The bound children of this node, in argument order. Empty for a scope root.
    pub fn children(&self) -> &[BoundExpression] {
        match &self.node {
            BoundExpressionNode::Scalar { children, .. } => children.as_slice(),
            BoundExpressionNode::Root { .. } => &[],
        }
    }

    /// Return the child at `index`.
    pub fn child(&self, index: usize) -> &BoundExpression {
        &self.children()[index]
    }

    /// The scalar function for this node, or `None` if it is the scope root.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match &self.node {
            BoundExpressionNode::Scalar { scalar_fn, .. } => Some(scalar_fn),
            BoundExpressionNode::Root { .. } => None,
        }
    }

    /// Return whether this node uses the given scalar-function vtable.
    pub fn is<V: ScalarFnVTable>(&self) -> bool {
        self.as_scalar().is_some_and(ScalarFnRef::is::<V>)
    }

    /// Return whether this expression tree contains a node using the given scalar-function vtable.
    pub fn contains<V: ScalarFnVTable>(&self) -> VortexResult<bool> {
        let mut contains = false;
        pre_order_visit_down(self, |node| {
            if node.is::<V>() {
                contains = true;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })?;
        Ok(contains)
    }

    /// Return the typed scalar-function options when this node uses the given vtable.
    pub fn as_opt<V: ScalarFnVTable>(&self) -> Option<&V::Options> {
        self.as_scalar().and_then(ScalarFnRef::as_opt::<V>)
    }

    /// Return the typed scalar-function options for this node.
    ///
    /// # Panics
    ///
    /// Panics when this node is the scope root or uses a different scalar-function vtable.
    pub fn as_<V: ScalarFnVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("Bound expression options type mismatch")
    }

    /// Whether this node is the scope root.
    pub fn is_root(&self) -> bool {
        matches!(self.node, BoundExpressionNode::Root { .. })
    }

    /// Return whether every scope root in this expression has `dtype`.
    ///
    /// Expressions without a scope root, such as literals, match every dtype.
    pub fn is_root_bound_to(&self, dtype: &DType) -> bool {
        let mut is_bound_to = true;
        pre_order_visit_down(self, |node| {
            if node.is_root() && node.dtype() != dtype {
                is_bound_to = false;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })
        .vortex_expect("bound expression traversal cannot not fail");
        is_bound_to
    }

    /// Returns a new expression representing the validity mask output of this expression.
    ///
    /// Returned expression evaluates to a non-nullable boolean array.
    /// When scalar function's validity is irreducible, returns
    /// "is_not_null(self)", which requires evaluating "self".
    pub fn validity(&self) -> VortexResult<BoundExpression> {
        match &self.node {
            // The scope is exactly as valid as itself.
            BoundExpressionNode::Root { .. } => Ok(self.clone()),
            BoundExpressionNode::Scalar { .. } => {
                let node = ExpressionReduceNode::new(self);
                Ok(match node.validity()? {
                    ReduceNodeValidity::Reduced(reduced) => reduced.into_expression(),
                    // IsNotNull(x) -> x.validity() symbolic reduction rule works only
                    // when node's.validity is Reduced to avoid infinite recursion.
                    ReduceNodeValidity::Irreducible => {
                        IsNotNull.try_new_bound_expr(EmptyOptions, [self.clone()])?
                    }
                })
            }
        }
    }

    /// Display the bound expression as a formatted tree structure.
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }
}

impl Debug for BoundExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.node, f)
    }
}

impl Display for BoundExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.as_scalar() {
            Some(scalar_fn) => scalar_fn.fmt_sql(self, f),
            None => f.write_str("$"),
        }
    }
}

impl Expression {
    /// Bind this expression against a root dtype, type-checking every node in a single walk.
    ///
    /// The returned tree carries a dtype on each node, so callers needing types at more than one
    /// node should bind once and read fields rather than calling
    /// [`return_dtype`](Expression::return_dtype) repeatedly.
    pub fn bind(&self, dtype: &DType) -> VortexResult<BoundExpression> {
        self.bind_scope(&Scope::new(dtype.clone()))
    }

    /// Bind this expression against an explicit [`Scope`].
    pub fn bind_scope(&self, scope: &Scope) -> VortexResult<BoundExpression> {
        if self.is_root() {
            return Ok(BoundExpression::new_root(scope.root().clone()));
        }
        if let Expression::Variable(name) = self {
            return Err(unbound_variable(name));
        }

        let children: Vec<_> = self
            .children()
            .iter()
            .map(|child| child.bind_scope(scope))
            .try_collect()?;
        let scalar_fn = self
            .as_scalar()
            .vortex_expect("leaves were handled above, so this is a scalar node");
        BoundExpression::try_new(scalar_fn.clone(), children)
    }
}

/// Iterative drop to avoid stack overflows on deep trees.
impl Drop for BoundExpression {
    fn drop(&mut self) {
        let BoundExpressionNode::Scalar { children, .. } = &mut self.node else {
            return;
        };
        let Some(children) = Arc::get_mut(children) else {
            return;
        };

        let mut to_drop = std::mem::take(children);
        while let Some(mut child) = to_drop.pop() {
            if let BoundExpressionNode::Scalar { children, .. } = &mut child.node
                && let Some(grandchildren) = Arc::get_mut(children)
            {
                to_drop.append(grandchildren);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::test_harness::struct_dtype;
    use crate::scalar_fn::fns::literal::Literal;

    fn scope() -> Scope {
        Scope::new(struct_dtype())
    }

    #[test]
    fn root_binds_to_the_scope() -> VortexResult<()> {
        let bound = root().bind_scope(&scope())?;
        assert!(bound.is_root());
        assert_eq!(bound.dtype(), &struct_dtype());
        assert_eq!(bound, BoundExpression::new_root(struct_dtype()));
        Ok(())
    }

    #[test]
    fn every_node_carries_its_dtype() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind_scope(&scope())?;

        assert_eq!(bound.dtype(), &DType::Bool(Nullability::NonNullable));

        let lhs = &bound.children()[0];
        assert_eq!(
            lhs.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(lhs.children()[0].dtype(), &struct_dtype());
        Ok(())
    }

    #[test]
    fn bind_agrees_with_return_dtype() -> VortexResult<()> {
        for (expr, expected) in [
            (root(), struct_dtype()),
            (
                col("a"),
                DType::Primitive(PType::I32, Nullability::NonNullable),
            ),
            (
                eq(col("a"), lit(1_i32)),
                DType::Bool(Nullability::NonNullable),
            ),
            (lit(true), DType::Bool(Nullability::NonNullable)),
        ] {
            assert_eq!(
                expr.bind(&struct_dtype())?.dtype(),
                &expected,
                "binding {expr}"
            );
            assert_eq!(
                expr.return_dtype(&struct_dtype())?,
                expected,
                "return dtype of {expr}"
            );
        }
        Ok(())
    }

    #[test]
    fn contains_scalar_function() -> VortexResult<()> {
        let bound = eq(col("a"), lit(1_i32)).bind_scope(&scope())?;
        assert!(bound.contains::<Literal>()?);
        assert!(!root().bind_scope(&scope())?.contains::<Literal>()?);
        Ok(())
    }

    #[test]
    fn bound_to_checks_every_root() -> VortexResult<()> {
        let dtype = struct_dtype();
        let bound = eq(col("a"), col("a")).bind(&dtype)?;
        assert!(bound.is_root_bound_to(&dtype));
        assert!(!bound.is_root_bound_to(&DType::Bool(Nullability::NonNullable)));
        assert!(
            lit(true)
                .bind(&dtype)?
                .is_root_bound_to(&DType::Bool(Nullability::NonNullable))
        );
        Ok(())
    }

    #[test]
    fn bound_display_matches_unbound() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            let bound = expr.bind_scope(&scope())?;
            assert_eq!(bound.to_string(), expr.to_string());
            assert_eq!(
                bound.display_tree().to_string(),
                expr.display_tree().to_string()
            );
        }
        Ok(())
    }

    #[test]
    fn clone_shares_children() -> VortexResult<()> {
        let bound = eq(col("a"), lit(1_i32)).bind_scope(&scope())?;
        let cloned = bound.clone();

        let (
            BoundExpressionNode::Scalar { children: a, .. },
            BoundExpressionNode::Scalar { children: b, .. },
        ) = (&bound.node, &cloned.node)
        else {
            unreachable!("eq is a scalar node")
        };
        assert!(Arc::ptr_eq(a, b));
        Ok(())
    }

    #[test]
    fn repeated_subtree_is_bound_per_occurrence() -> VortexResult<()> {
        let shared = col("a");
        let bound = eq(shared.clone(), shared).bind_scope(&scope())?;
        let children = bound.children();
        assert_eq!(children.len(), 2);
        for child in children {
            assert_eq!(
                child.dtype(),
                &DType::Primitive(PType::I32, Nullability::NonNullable)
            );
        }
        Ok(())
    }

    #[test]
    fn structural_and_exact_equality_are_distinct() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind_scope(&scope())?;
        let independently_bound = expr.bind_scope(&scope())?;

        assert_eq!(bound, independently_bound);
        assert_eq!(ExactBoundExpr(bound.clone()), ExactBoundExpr(bound.clone()));
        assert_ne!(ExactBoundExpr(bound), ExactBoundExpr(independently_bound));
        Ok(())
    }

    #[test]
    fn binding_reports_a_type_error() {
        let expr = eq(col("a"), lit("nope"));
        assert!(expr.bind_scope(&scope()).is_err());
    }
}
