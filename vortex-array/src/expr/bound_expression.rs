// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::scope::Scope;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::stats::rewrite::StatsRewriteCtx;

/// A shared handle to a [`BoundExpression`].
///
/// Bound trees are immutable and shared by handle: every node is reference counted, so cloning a
/// subtree, storing it in a cache, or handing it to another thread is a refcount bump rather than
/// a copy of the tree.
pub type BoundExpressionRef = Arc<BoundExpression>;

/// An [`Expression`] that has been type-checked against a [`Scope`].
///
/// Every node carries its own dtype, so reading one is a field access rather than a walk of the
/// subtree. Holding a `BoundExpression` is proof that the whole tree type-checked.
///
/// Nodes are handed around as [`BoundExpressionRef`] rather than by value.
///
/// Binding is purely logical: it deals only in [`DType`]s and never sees an array, a length, or an
/// encoding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundExpression {
    /// A scalar function applied to bound children.
    Scalar {
        /// The dtype this node evaluates to.
        dtype: DType,
        /// The scalar function for this node.
        scalar_fn: ScalarFnRef,
        /// The bound children, in argument order.
        ///
        /// Each child is shared, so rebuilding a node keeps the untouched subtrees in place.
        children: Box<[BoundExpressionRef]>,
    },
    /// The scope itself. Its dtype is the scope's root dtype.
    Root {
        /// The dtype this node evaluates to.
        dtype: DType,
    },
}

/// A bound-expression wrapper that compares shared tree identity instead of structure.
///
/// Two wrappers are equal when they hold the same node, or when they hold nodes built from the
/// same scalar function over the very same child handles. Structurally equal trees built
/// independently are not equal, which is what keeps identity-keyed caches from walking a tree on
/// every lookup.
#[derive(Clone, Debug)]
pub struct ExactBoundExpr(pub BoundExpressionRef);

impl PartialEq for ExactBoundExpr {
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        match (&*self.0, &*other.0) {
            (
                BoundExpression::Root { dtype: lhs_dtype },
                BoundExpression::Root { dtype: rhs_dtype },
            ) => lhs_dtype == rhs_dtype,
            (
                BoundExpression::Scalar {
                    dtype: lhs_dtype,
                    scalar_fn: lhs_fn,
                    children: lhs_children,
                },
                BoundExpression::Scalar {
                    dtype: rhs_dtype,
                    scalar_fn: rhs_fn,
                    children: rhs_children,
                },
            ) => {
                lhs_fn == rhs_fn
                    && lhs_children.len() == rhs_children.len()
                    && lhs_children
                        .iter()
                        .zip(rhs_children.iter())
                        .all(|(lhs, rhs)| Arc::ptr_eq(lhs, rhs))
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
        match &*self.0 {
            BoundExpression::Root { .. } => state.write_u8(0),
            BoundExpression::Scalar {
                scalar_fn,
                children,
                ..
            } => {
                state.write_u8(1);
                scalar_fn.hash(state);
                for child in children.iter() {
                    Arc::as_ptr(child).hash(state);
                }
            }
        }
    }
}

impl BoundExpression {
    /// Create a bound root expression with the given dtype.
    pub fn new_root(dtype: DType) -> BoundExpressionRef {
        Arc::new(Self::Root { dtype })
    }

    /// Create a bound scalar node from a scalar function and already-bound children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: impl IntoIterator<Item = BoundExpressionRef>,
    ) -> VortexResult<BoundExpressionRef> {
        Self::try_new_boxed(scalar_fn, children.into_iter().collect())
    }

    fn try_new_boxed(
        scalar_fn: ScalarFnRef,
        children: Box<[BoundExpressionRef]>,
    ) -> VortexResult<BoundExpressionRef> {
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
        let dtype = scalar_fn.return_dtype(&arg_dtypes)?;

        Ok(Arc::new(Self::Scalar {
            dtype,
            scalar_fn,
            children,
        }))
    }

    /// Rebuild this node with new bound children, recomputing its dtype.
    pub fn with_children(
        self: BoundExpressionRef,
        children: impl IntoIterator<Item = BoundExpressionRef>,
    ) -> VortexResult<BoundExpressionRef> {
        let children: Box<[_]> = children.into_iter().collect();
        let BoundExpression::Scalar { scalar_fn, .. } = self.as_ref() else {
            vortex_ensure!(
                children.is_empty(),
                "Root expression cannot have {} children",
                children.len()
            );
            return Ok(self);
        };

        Self::try_new_boxed(scalar_fn.clone(), children)
    }

    /// The dtype this expression evaluates to.
    pub fn dtype(&self) -> &DType {
        match self {
            Self::Scalar { dtype, .. } | Self::Root { dtype } => dtype,
        }
    }

    /// The bound children of this node, in argument order. Empty for [`BoundExpression::Root`].
    pub fn children(&self) -> &[BoundExpressionRef] {
        match self {
            Self::Scalar { children, .. } => children,
            Self::Root { .. } => &[],
        }
    }

    /// Return the child at `index`.
    pub fn child(&self, index: usize) -> &BoundExpressionRef {
        &self.children()[index]
    }

    /// The scalar function for this node, or `None` if it is the scope root.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match self {
            Self::Scalar { scalar_fn, .. } => Some(scalar_fn),
            Self::Root { .. } => None,
        }
    }

    /// Return whether this node uses the given scalar-function vtable.
    pub fn is<V: ScalarFnVTable>(&self) -> bool {
        self.as_scalar().is_some_and(ScalarFnRef::is::<V>)
    }

    /// Return whether this expression tree contains a node using the given scalar-function vtable.
    pub fn contains<V: ScalarFnVTable>(&self) -> VortexResult<bool> {
        Ok(self.any_node(|node| node.is::<V>()))
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
        matches!(self, Self::Root { .. })
    }

    /// Return whether every scope root in this expression has `dtype`.
    ///
    /// Expressions without a scope root, such as literals, match every dtype.
    pub fn is_root_bound_to(&self, dtype: &DType) -> bool {
        !self.any_node(|node| node.is_root() && node.dtype() != dtype)
    }

    /// Return an expression that proves this predicate is definitely false from statistics.
    pub fn falsify(
        self: BoundExpressionRef,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpressionRef>> {
        StatsRewriteCtx::new(session).falsify(&self)
    }

    /// Return an expression that proves this predicate is definitely true from statistics.
    pub fn satisfy(
        self: BoundExpressionRef,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpressionRef>> {
        StatsRewriteCtx::new(session).satisfy(&self)
    }

    /// Display the bound expression as a formatted tree structure.
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }

    /// Return whether any node of this tree satisfies `predicate`, walking iteratively.
    fn any_node(&self, mut predicate: impl FnMut(&BoundExpression) -> bool) -> bool {
        let mut stack = vec![self];
        while let Some(node) = stack.pop() {
            if predicate(node) {
                return true;
            }
            stack.extend(node.children().iter().map(Arc::as_ref));
        }
        false
    }
}

impl Display for BoundExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar { scalar_fn, .. } => scalar_fn.fmt_sql(self, f),
            Self::Root { .. } => f.write_str("$"),
        }
    }
}

impl Expression {
    /// Bind this expression against a root dtype, type-checking every node in a single walk.
    ///
    /// The returned tree carries a dtype on each node, so callers needing types at more than one
    /// node should bind once and read fields rather than calling
    /// [`return_dtype`](Expression::return_dtype) repeatedly.
    pub fn bind(&self, dtype: &DType) -> VortexResult<BoundExpressionRef> {
        self.bind_scope(&Scope::new(dtype.clone()))
    }

    /// Bind this expression against an explicit [`Scope`].
    pub fn bind_scope(&self, scope: &Scope) -> VortexResult<BoundExpressionRef> {
        if self.is_root() {
            return Ok(BoundExpression::new_root(scope.root().clone()));
        }

        let children: Vec<_> = self
            .children()
            .iter()
            .map(|child| child.bind_scope(scope))
            .try_collect()?;
        let scalar_fn = self
            .as_scalar()
            .vortex_expect("root was handled above, so this is a scalar node");
        BoundExpression::try_new(scalar_fn.clone(), children)
    }
}

/// Iterative drop to avoid stack overflows on deep trees.
impl Drop for BoundExpression {
    fn drop(&mut self) {
        let Self::Scalar { children, .. } = self else {
            return;
        };

        let mut to_drop = std::mem::take(children).into_vec();
        while let Some(child) = to_drop.pop() {
            // Descending is only useful for the last owner of a subtree; releasing a shared
            // handle is O(1) and leaves the children alone.
            if let Some(mut node) = Arc::into_inner(child)
                && let Self::Scalar { children, .. } = &mut node
            {
                to_drop.append(&mut std::mem::take(children).into_vec());
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
    use crate::expr::bound;
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
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            assert_eq!(
                expr.bind(&struct_dtype())?.dtype(),
                &expr.return_dtype(&struct_dtype())?,
                "disagreement for {expr}"
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
    fn clone_shares_the_tree() -> VortexResult<()> {
        let bound = eq(col("a"), lit(1_i32)).bind_scope(&scope())?;
        let cloned = Arc::clone(&bound);

        assert!(Arc::ptr_eq(&bound, &cloned));
        Ok(())
    }

    #[test]
    fn rebuilding_a_node_shares_untouched_children() -> VortexResult<()> {
        let bound = eq(col("a"), lit(1_i32)).bind_scope(&scope())?;
        let children = bound.children().to_vec();
        let rebuilt = Arc::clone(&bound).with_children(children)?;

        assert!(!Arc::ptr_eq(&bound, &rebuilt));
        for (old, new) in bound.children().iter().zip(rebuilt.children()) {
            assert!(Arc::ptr_eq(old, new));
        }
        Ok(())
    }

    #[test]
    fn repeated_subtree_is_bound_per_occurrence() -> VortexResult<()> {
        let shared = col("a");
        let bound = eq(shared.clone(), shared).bind_scope(&scope())?;
        let children = bound.children();
        assert_eq!(children[0].dtype(), children[1].dtype());
        Ok(())
    }

    #[test]
    fn structural_and_exact_equality_are_distinct() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind_scope(&scope())?;
        let independently_bound = expr.bind_scope(&scope())?;

        assert_eq!(bound, independently_bound);
        assert_eq!(
            ExactBoundExpr(Arc::clone(&bound)),
            ExactBoundExpr(Arc::clone(&bound))
        );
        assert_ne!(ExactBoundExpr(bound), ExactBoundExpr(independently_bound));
        Ok(())
    }

    #[test]
    fn deep_trees_drop_without_overflowing_the_stack() -> VortexResult<()> {
        let mut expr = lit(true).bind(&struct_dtype())?;
        for _ in 0..100_000 {
            expr = bound::not(expr);
        }
        drop(expr);
        Ok(())
    }

    #[test]
    fn binding_reports_a_type_error() {
        let expr = eq(col("a"), lit("nope"));
        assert!(expr.bind_scope(&scope()).is_err());
    }
}
