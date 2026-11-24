// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Borrow;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::Weak;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::expr::display::DisplayTreeExpr;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExpressionReduceNode;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::builtin::is_not_null::IsNotNull;
use crate::scalar_fn::builtin::literal::Literal;
use crate::scalar_fn::display::ExprDisplay;

/// An expression that has been type-checked against a scope's root dtype.
///
/// Every node carries its own dtype, so reading one is a field access rather than a walk of the
/// subtree. Holding a `BoundExpression` is proof that the whole tree type-checked.
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
        /// Sharing keeps clones cheap even though the iterative [`Drop`] implementation prevents
        /// consumers from destructuring a `BoundExpression` by value.
        children: Arc<Vec<BoundExpression>>,
    },
    /// The scope itself. Its dtype is the scope's root dtype.
    Root {
        /// The dtype this node evaluates to.
        dtype: DType,
    },
}

/// A bound-expression wrapper that compares shared tree identity instead of structure.
#[derive(Clone, Debug)]
pub struct ExactBoundExpr(pub BoundExpression);

/// A bound expression compared by shared tree identity, as [`ExactBoundExpr`] compares it.
///
/// Maps keyed by [`ExactBoundExpr`] or [`WeakBoundExprKey`] can be probed with
/// `expr as &dyn ExactKey` for a borrowed [`BoundExpression`], which spares cloning the
/// expression for every lookup. Clones touch reference counts that every scan thread shares.
pub trait ExactKey {
    /// The identity of the expression that this key stands for.
    fn identity(&self) -> ExactIdentity<'_>;
}

/// What identifies a bound expression to an [`ExactKey`]: the dtype of a root, or the function,
/// dtype and shared children of a scalar node.
pub enum ExactIdentity<'a> {
    /// A root of the given dtype.
    Root(&'a DType),
    /// A scalar node, identified by the address of its shared children.
    Scalar {
        /// The address of the shared children of the node.
        children: *const Vec<BoundExpression>,
        /// The scalar function of the node.
        scalar_fn: &'a ScalarFnRef,
        /// The dtype of the node.
        dtype: &'a DType,
    },
}

impl BoundExpression {
    fn exact_identity(&self) -> ExactIdentity<'_> {
        match self {
            BoundExpression::Root { dtype } => ExactIdentity::Root(dtype),
            BoundExpression::Scalar {
                dtype,
                scalar_fn,
                children,
            } => ExactIdentity::Scalar {
                children: Arc::as_ptr(children),
                scalar_fn,
                dtype,
            },
        }
    }
}

impl ExactKey for ExactBoundExpr {
    fn identity(&self) -> ExactIdentity<'_> {
        self.0.exact_identity()
    }
}

impl ExactKey for BoundExpression {
    fn identity(&self) -> ExactIdentity<'_> {
        self.exact_identity()
    }
}

impl<'a> Borrow<dyn ExactKey + 'a> for ExactBoundExpr {
    fn borrow(&self) -> &(dyn ExactKey + 'a) {
        self
    }
}

impl PartialEq for dyn ExactKey + '_ {
    fn eq(&self, other: &Self) -> bool {
        match (self.identity(), other.identity()) {
            (ExactIdentity::Root(lhs_dtype), ExactIdentity::Root(rhs_dtype)) => {
                lhs_dtype == rhs_dtype
            }
            (
                ExactIdentity::Scalar {
                    children: lhs_children,
                    scalar_fn: lhs_fn,
                    dtype: lhs_dtype,
                },
                ExactIdentity::Scalar {
                    children: rhs_children,
                    scalar_fn: rhs_fn,
                    dtype: rhs_dtype,
                },
            ) => {
                std::ptr::eq(lhs_children, rhs_children)
                    && lhs_fn == rhs_fn
                    && lhs_dtype == rhs_dtype
            }
            _ => false,
        }
    }
}

impl Eq for dyn ExactKey + '_ {}

impl Hash for dyn ExactKey + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Equal keys share their children, so the identity of the children is enough to hash.
        // Omitting the function's options and the potentially lazy dtype keeps identity-keyed
        // cache lookups from hashing literal lists or deserializing an entire schema.
        match self.identity() {
            ExactIdentity::Root(_) => state.write_u8(0),
            ExactIdentity::Scalar { children, .. } => {
                state.write_u8(1);
                children.hash(state);
            }
        }
    }
}

/// A key that identifies a bound expression as [`ExactBoundExpr`] does, without keeping the
/// expression alive.
///
/// Caches that outlive the scans they serve, such as those of layout readers reused across
/// queries, key their entries by it. A key can only be looked up while something else holds its
/// expression, so [`Self::is_live`] tells which entries can still be hit, and the caches drop the
/// others. The weak reference keeps the allocation of the children, so no later expression can
/// reuse their address and match a stale key.
#[derive(Clone, Debug)]
pub enum WeakBoundExprKey {
    /// A root of the given dtype.
    Root(DType),
    /// A scalar node, identified by its shared children.
    Scalar {
        /// The shared children of the node.
        children: Weak<Vec<BoundExpression>>,
        /// The scalar function of the node.
        scalar_fn: ScalarFnRef,
        /// The dtype of the node.
        dtype: DType,
    },
}

impl WeakBoundExprKey {
    /// The key of `expr`.
    pub fn new(expr: &BoundExpression) -> Self {
        match expr {
            BoundExpression::Root { dtype } => Self::Root(dtype.clone()),
            BoundExpression::Scalar {
                dtype,
                scalar_fn,
                children,
            } => Self::Scalar {
                children: Arc::downgrade(children),
                scalar_fn: scalar_fn.clone(),
                dtype: dtype.clone(),
            },
        }
    }

    /// Whether the expression of this key is still alive, so that a lookup can still hit it.
    pub fn is_live(&self) -> bool {
        match self {
            Self::Root(_) => true,
            Self::Scalar { children, .. } => children.strong_count() > 0,
        }
    }
}

impl ExactKey for WeakBoundExprKey {
    fn identity(&self) -> ExactIdentity<'_> {
        match self {
            Self::Root(dtype) => ExactIdentity::Root(dtype),
            Self::Scalar {
                children,
                scalar_fn,
                dtype,
            } => ExactIdentity::Scalar {
                children: children.as_ptr(),
                scalar_fn,
                dtype,
            },
        }
    }
}

impl<'a> Borrow<dyn ExactKey + 'a> for WeakBoundExprKey {
    fn borrow(&self) -> &(dyn ExactKey + 'a) {
        self
    }
}

impl PartialEq for WeakBoundExprKey {
    fn eq(&self, other: &Self) -> bool {
        (self as &dyn ExactKey) == (other as &dyn ExactKey)
    }
}

impl Eq for WeakBoundExprKey {}

impl Hash for WeakBoundExprKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self as &dyn ExactKey).hash(state);
    }
}

impl PartialEq for ExactBoundExpr {
    fn eq(&self, other: &Self) -> bool {
        (self as &dyn ExactKey) == (other as &dyn ExactKey)
    }
}

impl Eq for ExactBoundExpr {}

impl Hash for ExactBoundExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self as &dyn ExactKey).hash(state);
    }
}

impl BoundExpression {
    /// Create a bound root expression with the given dtype.
    pub fn new_root(dtype: DType) -> Self {
        Self::Root { dtype }
    }

    /// Create a bound scalar node from a scalar function and already-bound children.
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

        Ok(Self::Scalar {
            dtype,
            scalar_fn,
            children: children.into(),
        })
    }

    /// Rebuild this node with new bound children, recomputing its dtype.
    pub fn with_children(
        self,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        let BoundExpression::Scalar { scalar_fn, .. } = &self else {
            vortex_ensure!(
                children.is_empty(),
                "Root expression cannot have {} children",
                children.len()
            );
            return Ok(self);
        };

        Self::try_new_vec(scalar_fn.clone(), children)
    }

    /// Returns a copy of this node with a fresh top-level children allocation, preserving its
    /// scalar function, dtype, and child subtrees.
    ///
    /// Caches keyed by [`WeakBoundExprKey`] use this when a cached output expression would
    /// otherwise reuse the key's own children allocation and keep its weak key alive forever.
    pub fn detach_identity(&self) -> Self {
        match self {
            Self::Root { dtype } => Self::Root {
                dtype: dtype.clone(),
            },
            Self::Scalar {
                dtype,
                scalar_fn,
                children,
            } => Self::Scalar {
                dtype: dtype.clone(),
                scalar_fn: scalar_fn.clone(),
                children: Arc::new((**children).clone()),
            },
        }
    }

    /// The dtype this expression evaluates to.
    pub fn dtype(&self) -> &DType {
        match self {
            Self::Scalar { dtype, .. } | Self::Root { dtype } => dtype,
        }
    }

    /// The bound children of this node, in argument order. Empty for [`BoundExpression::Root`].
    pub fn children(&self) -> &[BoundExpression] {
        match self {
            Self::Scalar { children, .. } => children.as_slice(),
            Self::Root { .. } => &[],
        }
    }

    /// Return the child at `index`.
    pub fn child(&self, index: usize) -> &BoundExpression {
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

    /// Whether any node of this tree, visited depth-first, satisfies `predicate`.
    fn any_node(&self, mut predicate: impl FnMut(&BoundExpression) -> bool) -> bool {
        let mut stack = vec![self];
        while let Some(node) = stack.pop() {
            if predicate(node) {
                return true;
            }
            stack.extend(node.children().iter().rev());
        }
        false
    }

    /// Create a bound literal node.
    pub fn literal(value: Scalar) -> Self {
        Self::try_new(Literal.bind(value), [])
            .vortex_expect("literal expressions are always well-typed")
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

    /// Returns a new expression representing the validity mask output of this expression.
    ///
    /// Returned expression evaluates to a non-nullable boolean array.
    /// When scalar function's validity is irreducible, returns
    /// "is_not_null(self)", which requires evaluating "self".
    pub fn validity(&self) -> VortexResult<BoundExpression> {
        match self {
            // The scope is exactly as valid as itself.
            Self::Root { .. } => Ok(self.clone()),
            Self::Scalar { .. } => {
                let node = ExpressionReduceNode::new(self);
                Ok(match node.validity()? {
                    ReduceNodeValidity::Reduced(reduced) => reduced.into_expression(),
                    // IsNotNull(x) -> x.validity() symbolic reduction rule works only
                    // when node's.validity is Reduced to avoid infinite recursion.
                    ReduceNodeValidity::Irreducible => {
                        BoundExpression::try_new(IsNotNull.bind(EmptyOptions), [self.clone()])?
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

impl Display for BoundExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar { scalar_fn, .. } => scalar_fn.fmt_sql(self, f),
            Self::Root { .. } => f.write_str("$"),
        }
    }
}

/// Iterative drop to avoid stack overflows on deep trees.
impl Drop for BoundExpression {
    fn drop(&mut self) {
        let Self::Scalar { children, .. } = self else {
            return;
        };
        let Some(children) = Arc::get_mut(children) else {
            return;
        };

        let mut to_drop = std::mem::take(children);
        while let Some(mut child) = to_drop.pop() {
            if let BoundExpression::Scalar { children, .. } = &mut child
                && let Some(grandchildren) = Arc::get_mut(children)
            {
                to_drop.append(grandchildren);
            }
        }
    }
}

impl ExprDisplay for BoundExpression {
    fn display_child(&self, index: usize) -> &dyn ExprDisplay {
        &self.children()[index]
    }

    fn display_children_count(&self) -> usize {
        self.children().len()
    }
}
