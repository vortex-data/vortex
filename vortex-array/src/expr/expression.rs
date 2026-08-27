// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;

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

    /// Returns the scalar fn for this expression, or `None` if it is not a scalar node.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match self {
            Self::Scalar { scalar_fn, .. } => Some(scalar_fn),
            Self::Root => None,
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
    pub fn children(&self) -> &[Expression] {
        match self {
            Self::Scalar { children, .. } => children.as_slice(),
            Self::Root => NO_CHILDREN,
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
            Self::Root => {
                vortex_ensure!(
                    children.is_empty(),
                    "Expression arity mismatch: root expects 0 children but got {}",
                    children.len()
                );
                Ok(Self::Root)
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
            Self::Scalar { scalar_fn, .. } => scalar_fn.validity(self),
        }
    }

    /// Format the expression as a compact string.
    ///
    /// Since this is a recursive formatter, it is exposed on the public Expression type.
    /// See fmt_data that is only implemented on the vtable trait.
    pub fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => write!(f, "$"),
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

/// The default display implementation for expressions uses the 'SQL'-style format.
impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_sql(f)
    }
}

/// The number of [`Expression`] drops that may nest before a drop becomes iterative.
///
/// Expression trees are almost always shallow, so recursion, which the compiler-generated drop
/// glue does anyway, is the fast path. Each level uses a few hundred bytes of stack, so this
/// limit stays far below the stack of even a small worker thread.
const MAX_DROP_DEPTH: u32 = 32;

thread_local! {
    /// The number of [`Expression`] drops on the stack of this thread.
    static DROP_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Counts one level of recursive [`Expression`] drop, and releases it when the level completes.
struct DropDepthGuard;

impl DropDepthGuard {
    /// Returns a guard if the recursion is still inside the depth limit, else `None`.
    fn enter() -> Option<Self> {
        DROP_DEPTH.with(|depth| {
            let current = depth.get();
            (current < MAX_DROP_DEPTH).then(|| {
                depth.set(current + 1);
                Self
            })
        })
    }
}

impl Drop for DropDepthGuard {
    fn drop(&mut self) {
        DROP_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// Recursive drop for expressions, which becomes iterative on deep trees to protect the stack.
impl Drop for Expression {
    fn drop(&mut self) {
        let Self::Scalar { children, .. } = self else {
            return;
        };
        // Shared children outlive this node, so no descendant drops here.
        let Some(children) = Arc::get_mut(children) else {
            return;
        };
        if children.is_empty() {
            return;
        }

        let mut children_to_drop = std::mem::take(children);

        match DropDepthGuard::enter() {
            // The children drop here, and not through the drop glue of this node, because the
            // recursion below must occur while the guard holds the deeper level.
            Some(_guard) => drop(children_to_drop),
            // Below the deepest recursive level, unwind the remainder of the tree iteratively.
            None => {
                while let Some(mut child) = children_to_drop.pop() {
                    if let Self::Scalar { children, .. } = &mut child
                        && let Some(expr_children) = Arc::get_mut(children)
                    {
                        children_to_drop.append(expr_children);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::expr::lit;
    use crate::expr::not;

    /// A chain of `not` nodes of the given depth.
    fn deep_expression(depth: usize) -> Expression {
        let mut expr = lit(true);
        for _ in 0..depth {
            expr = not(expr);
        }
        expr
    }

    #[test]
    fn deep_expression_drops_within_a_small_stack() -> VortexResult<()> {
        const DEPTH: usize = 100_000;
        const STACK_SIZE: usize = 256 * 1024;

        let dropper = thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| drop(deep_expression(DEPTH)))?;

        assert!(
            dropper.join().is_ok(),
            "dropping a tree of depth {DEPTH} exhausted a {STACK_SIZE} byte stack"
        );

        Ok(())
    }

    #[test]
    fn shallow_expression_keeps_shared_children() {
        let expr = not(lit(true));
        let shared = expr.clone();

        drop(expr);

        assert_eq!(shared.children().len(), 1);
    }
}
