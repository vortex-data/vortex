// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::lambda::Lambda;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::expr::variable::Variable;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::stats::rewrite::StatsRewriteCtx;

/// An empty child slice, returned by [`Expression::children`] for childless variants.
const NO_CHILDREN: &[Expression] = &[];

/// A node in a Vortex expression tree.
///
/// Most nodes are a scalar function applied to child expressions. [`Expression::Root`] is the scope
/// itself: a language primitive rather than a registered function, because its dtype comes from the
/// scope rather than from children and it is not executable. A [`ScalarFnVTable`] can answer neither
/// of those, so `Root` is a variant instead.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expression {
    /// A scalar function applied to child expressions.
    Scalar {
        /// The scalar fn for this node.
        scalar_fn: ScalarFnRef,
        /// Any children of this expression.
        children: Arc<Vec<Expression>>,
    },
    /// The full scope of the expression evaluation.
    Root,
    /// A reference to a name bound by an enclosing [`Expression::Lambda`].
    ///
    /// Like [`Expression::Root`], its dtype comes from the scope rather than from children, so it
    /// is resolved during binding.
    Variable(Variable),
    /// A body evaluated under a frame binding `params`.
    ///
    /// A lambda is **not a value**: its parameter dtypes are determined by whatever applies it, so
    /// it has no dtype of its own and cannot be bound by
    /// [`bind_scope`](Expression::bind_scope). Use
    /// [`Lambda::bind`], which takes the parameter types.
    Lambda(Lambda),
}

impl Expression {
    /// Create a new expression node from a scalar_fn expression and its children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);

        vortex_ensure!(
            scalar_fn.signature().arity().matches(children.len()),
            "Expression arity mismatch: expected {} children but got {}",
            scalar_fn.signature().arity(),
            children.len()
        );

        Ok(Self::Scalar {
            scalar_fn,
            children: children.into(),
        })
    }

    /// Whether this expression is the scope root.
    pub fn is_root(&self) -> bool {
        matches!(self, Self::Root)
    }

    /// The variable this expression references, if it is a variable.
    pub fn as_variable(&self) -> Option<&Variable> {
        match self {
            Self::Variable(variable) => Some(variable),
            _ => None,
        }
    }

    /// The lambda this expression holds, if it is one.
    pub fn as_lambda(&self) -> Option<&Lambda> {
        match self {
            Self::Lambda(lambda) => Some(lambda),
            _ => None,
        }
    }

    /// Returns the scalar fn for this expression, or `None` if it is not a scalar node.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match self {
            Self::Scalar { scalar_fn, .. } => Some(scalar_fn),
            Self::Root | Self::Variable(_) | Self::Lambda(_) => None,
        }
    }

    /// Whether this expression's scalar fn is of the given vtable type.
    pub fn is<V: ScalarFnVTable>(&self) -> bool {
        self.as_scalar().is_some_and(|sf| sf.is::<V>())
    }

    /// The typed options for this expression if its scalar fn matches the given vtable type.
    pub fn as_opt<V: ScalarFnVTable>(&self) -> Option<&V::Options> {
        self.as_scalar().and_then(|sf| sf.as_opt::<V>())
    }

    /// The typed options for this expression.
    ///
    /// # Panics
    ///
    /// Panics if the vtable type does not match.
    pub fn as_<V: ScalarFnVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("Expression options type mismatch")
    }

    /// Returns the children of this expression.
    /// Returns the sub-expressions of this node.
    ///
    /// A [`Expression::Lambda`] yields its body, so generic traversal reaches it — but only by
    /// passing through the `Lambda` node, which is a scope boundary. A pass that is not
    /// scope-aware must handle that variant rather than descending blindly.
    pub fn children(&self) -> &[Expression] {
        match self {
            Self::Scalar { children, .. } => children.as_slice(),
            Self::Lambda(lambda) => std::slice::from_ref(lambda.body_arc()),
            Self::Root | Self::Variable(_) => NO_CHILDREN,
        }
    }

    /// Returns the n'th child of this expression.
    pub fn child(&self, n: usize) -> &Expression {
        &self.children()[n]
    }

    /// Replace the children of this expression with the provided new children.
    pub fn with_children(
        self,
        children: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        match &self {
            Self::Root | Self::Variable(_) => {
                vortex_ensure!(
                    children.is_empty(),
                    "Expression arity mismatch: {self} expects 0 children but got {}",
                    children.len()
                );
                Ok(self.clone())
            }
            Self::Lambda(lambda) => {
                vortex_ensure!(
                    children.len() == 1,
                    "Expression arity mismatch: a lambda expects 1 child but got {}",
                    children.len()
                );
                Ok(Self::Lambda(Lambda::new(
                    lambda.params().iter().cloned(),
                    children.into_iter().next().vortex_expect("checked above"),
                )))
            }
            Self::Scalar { scalar_fn, .. } => {
                vortex_ensure!(
                    scalar_fn.signature().arity().matches(children.len()),
                    "Expression arity mismatch: expected {} children but got {}",
                    scalar_fn.signature().arity(),
                    children.len()
                );
                Ok(Self::Scalar {
                    scalar_fn: scalar_fn.clone(),
                    children: children.into(),
                })
            }
        }
    }

    /// Computes the return dtype of this expression given the input dtype.
    pub fn return_dtype(&self, scope: &DType) -> VortexResult<DType> {
        match self {
            Self::Root => Ok(scope.clone()),
            // A variable resolves against a frame, which this entry point does not carry. Erroring
            // keeps callers that only have a root dtype from silently mistyping a lambda body.
            Self::Variable(variable) => vortex_bail!(
                "variable '{variable}' can only be typed by binding against a scope with frames"
            ),
            Self::Lambda(_) => {
                vortex_bail!("a lambda has no data type; use Lambda::bind to type its body")
            }
            Self::Scalar {
                scalar_fn,
                children,
            } => {
                let dtypes: Vec<_> = children
                    .iter()
                    .map(|c| c.return_dtype(scope))
                    .try_collect()?;
                scalar_fn.return_dtype(&dtypes)
            }
        }
    }

    /// Returns a new expression representing the validity mask output of this expression.
    ///
    /// The returned expression evaluates to a non-nullable boolean array.
    pub fn validity(&self) -> VortexResult<Expression> {
        match self {
            // The scope is exactly as valid as itself.
            Self::Root => Ok(Self::Root),
            // A variable is exactly as valid as whatever it is bound to.
            Self::Variable(_) => Ok(self.clone()),
            Self::Lambda(_) => {
                vortex_bail!("a lambda has no validity; it is not a value")
            }
            Self::Scalar { scalar_fn, .. } => scalar_fn.validity(self),
        }
    }

    /// Returns an expression that proves this predicate is definitely false from stats.
    ///
    /// `scope` is the dtype of the row this expression evaluates over.
    ///
    /// If the returned expression evaluates to `true` for a stats scope, this expression is
    /// guaranteed to be false for every row in that scope. `false` and `null` are unknown.
    pub fn falsify(
        &self,
        scope: &DType,
        session: &VortexSession,
    ) -> VortexResult<Option<Expression>> {
        StatsRewriteCtx::new(session, scope).falsify(self)
    }

    /// Returns an expression that proves this predicate is definitely true from stats.
    ///
    /// `scope` is the dtype of the row this expression evaluates over.
    ///
    /// If the returned expression evaluates to `true` for a stats scope, this expression is
    /// guaranteed to be true for every row in that scope. `false` and `null` are unknown.
    pub fn satisfy(
        &self,
        scope: &DType,
        session: &VortexSession,
    ) -> VortexResult<Option<Expression>> {
        StatsRewriteCtx::new(session, scope).satisfy(self)
    }

    /// Format the expression as a compact string.
    ///
    /// Since this is a recursive formatter, it is exposed on the public Expression type.
    /// See fmt_data that is only implemented on the vtable trait.
    pub fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => write!(f, "$"),
            Self::Variable(variable) => write!(f, "${variable}"),
            Self::Lambda(lambda) => Display::fmt(lambda, f),
            Self::Scalar { scalar_fn, .. } => scalar_fn.fmt_sql(self, f),
        }
    }

    /// Display the expression as a formatted tree structure.
    ///
    /// This provides a hierarchical view of the expression that shows the relationships
    /// between parent and child expressions, making complex nested expressions easier
    /// to understand and debug.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use vortex_array::dtype::{DType, Nullability, PType};
    /// # use vortex_array::scalar_fn::fns::like::{Like, LikeOptions};
    /// # use vortex_array::scalar_fn::ScalarFnVTableExt;
    /// # use vortex_array::expr::{and, cast, eq, get_item, gt, lit, not, root, select};
    /// // Build a complex nested expression
    /// let complex_expr = select(
    ///     ["result"],
    ///     and(
    ///         not(eq(get_item("status", root()), lit("inactive"))),
    ///         and(
    ///             Like.new_expr(LikeOptions::default(), [get_item("name", root()), lit("%admin%")]),
    ///             gt(
    ///                 cast(get_item("score", root()), DType::Primitive(PType::F64, Nullability::NonNullable)),
    ///                 lit(75.0)
    ///             )
    ///         )
    ///     )
    /// );
    ///
    /// println!("{}", complex_expr.display_tree());
    /// ```
    ///
    /// This produces output like:
    ///
    /// ```text
    /// Select(include): {result}
    /// └── Binary(and)
    ///     ├── lhs: Not
    ///     │   └── Binary(=)
    ///     │       ├── lhs: GetItem(status)
    ///     │       │   └── Root
    ///     │       └── rhs: Literal(value: "inactive", dtype: utf8)
    ///     └── rhs: Binary(and)
    ///         ├── lhs: Like
    ///         │   ├── child: GetItem(name)
    ///         │   │   └── Root
    ///         │   └── pattern: Literal(value: "%admin%", dtype: utf8)
    ///         └── rhs: Binary(>)
    ///             ├── lhs: Cast(target: f64)
    ///             │   └── GetItem(score)
    ///             │       └── Root
    ///             └── rhs: Literal(value: 75f64, dtype: f64)
    /// ```
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }

    /// Returns true if this expression contains expression E inside.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use vortex_array::scalar_fn::fns::literal::Literal;
    /// # use vortex_array::expr::{eq, lit, root};
    /// let expression = &eq(root(), lit(3u64));
    /// assert!(expression.contains::<Literal>().unwrap());
    /// let expression = root();
    /// assert!(!expression.contains::<Literal>().unwrap());
    /// ```
    pub fn contains<E: ScalarFnVTable>(&self) -> VortexResult<bool> {
        let mut contains = false;
        pre_order_visit_down(self, |node| {
            if node.is::<E>() {
                contains = true;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })?;
        Ok(contains)
    }
}

/// `Root` stands in as the default so that a body can be moved out during the iterative [`Drop`]
/// below. It is never observed by callers.
impl Default for Expression {
    fn default() -> Self {
        Self::Root
    }
}

/// The default display implementation for expressions uses the 'SQL'-style format.
impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_sql(f)
    }
}

/// Iterative drop for expression to avoid stack overflows.
impl Drop for Expression {
    fn drop(&mut self) {
        let mut children_to_drop = Vec::new();
        match self {
            Self::Scalar { children, .. } => {
                if let Some(children) = Arc::get_mut(children) {
                    children_to_drop.append(children);
                }
            }
            Self::Lambda(lambda) => {
                if let Some(body) = lambda.take_unique_body() {
                    children_to_drop.push(body);
                }
            }
            Self::Root | Self::Variable(_) => return,
        }

        while let Some(mut child) = children_to_drop.pop() {
            match &mut child {
                Self::Scalar { children, .. } => {
                    if let Some(expr_children) = Arc::get_mut(children) {
                        children_to_drop.append(expr_children);
                    }
                }
                Self::Lambda(lambda) => {
                    if let Some(body) = lambda.take_unique_body() {
                        children_to_drop.push(body);
                    }
                }
                Self::Root | Self::Variable(_) => {}
            }
        }
    }
}
