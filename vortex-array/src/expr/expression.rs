// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::expr::StatsCatalog;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::stats::Stat;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::fns::root::Root;

/// A node in a Vortex expression tree.
///
/// Expressions represent scalar computations that can be performed on data. Each
/// expression consists of an encoding (vtable), heap-allocated metadata, and child expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Expression {
    /// The scalar fn for this node.
    scalar_fn: ScalarFnRef,
    /// Any children of this expression.
    children: Arc<Vec<Expression>>,
}

impl Deref for Expression {
    type Target = ScalarFnRef;

    fn deref(&self) -> &Self::Target {
        &self.scalar_fn
    }
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

        Ok(Self {
            scalar_fn,
            children: children.into(),
        })
    }

    /// Returns the scalar fn vtable for this expression.
    pub fn scalar_fn(&self) -> &ScalarFnRef {
        &self.scalar_fn
    }

    /// Returns the children of this expression.
    pub fn children(&self) -> &Arc<Vec<Expression>> {
        &self.children
    }

    /// Returns the n'th child of this expression.
    pub fn child(&self, n: usize) -> &Expression {
        &self.children[n]
    }

    /// Replace the children of this expression with the provided new children.
    pub fn with_children(
        mut self,
        children: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        vortex_ensure!(
            self.signature().arity().matches(children.len()),
            "Expression arity mismatch: expected {} children but got {}",
            self.signature().arity(),
            children.len()
        );
        self.children = Arc::new(children);
        Ok(self)
    }

    /// Computes the return dtype of this expression given the input dtype.
    pub fn return_dtype(&self, scope: &DType) -> VortexResult<DType> {
        if self.is::<Root>() {
            return Ok(scope.clone());
        }

        let dtypes: Vec<_> = self
            .children
            .iter()
            .map(|c| c.return_dtype(scope))
            .try_collect()?;
        self.scalar_fn.return_dtype(&dtypes)
    }

    /// Returns a new expression representing the validity mask output of this expression.
    ///
    /// The returned expression evaluates to a non-nullable boolean array.
    pub fn validity(&self) -> VortexResult<Expression> {
        self.scalar_fn.validity(self)
    }

    /// An expression over zone-statistics which implies all records in the zone evaluate to false.
    ///
    /// Given an expression, `e`, if `e.stat_falsification(..)` evaluates to true, it is guaranteed
    /// that `e` evaluates to false on all records in the zone. However, the inverse is not
    /// necessarily true: even if the falsification evaluates to false, `e` need not evaluate to
    /// true on all records.
    ///
    /// The [`StatsCatalog`] can be used to constrain or rename stats used in the final expr.
    ///
    /// # Examples
    ///
    /// - An expression over one variable: `x > 0` is false for all records in a zone if the maximum
    ///   value of the column `x` in that zone is less than or equal to zero: `max(x) <= 0`.
    /// - An expression over two variables: `x > y` becomes `max(x) <= min(y)`.
    /// - A conjunctive expression: `x > y AND z < x` becomes `max(x) <= min(y) OR min(z) >= max(x).
    ///
    /// Some expressions, in theory, have falsifications but this function does not support them
    /// such as `x < (y < z)` or `x LIKE "needle%"`.
    pub fn stat_falsification(&self, catalog: &dyn StatsCatalog) -> Option<Expression> {
        self.scalar_fn().stat_falsification(self, catalog)
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
        crate::stats::rewrite::StatsRewriteCtx::new(session, scope).falsify(self)
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
        crate::stats::rewrite::StatsRewriteCtx::new(session, scope).satisfy(self)
    }

    /// Returns an expression representing the zoned statistic for the given stat, if available.
    ///
    /// The [`StatsCatalog`] returns expressions that can be evaluated using the zone map as a
    /// scope. Expressions can implement this function to propagate such statistics through the
    /// expression tree. For example, the `a + 10` expression could propagate `min: min(a) + 10`.
    ///
    /// NOTE(gatesn): we currently cannot represent statistics over nested fields. Please file an
    /// issue to discuss a solution to this.
    pub fn stat_expression(&self, stat: Stat, catalog: &dyn StatsCatalog) -> Option<Expression> {
        self.scalar_fn().stat_expression(self, stat, catalog)
    }

    /// Returns an expression representing the zoned maximum statistic, if available.
    pub fn stat_min(&self, catalog: &dyn StatsCatalog) -> Option<Expression> {
        self.stat_expression(Stat::Min, catalog)
    }

    /// Returns an expression representing the zoned maximum statistic, if available.
    pub fn stat_max(&self, catalog: &dyn StatsCatalog) -> Option<Expression> {
        self.stat_expression(Stat::Max, catalog)
    }

    /// Format the expression as a compact string.
    ///
    /// Since this is a recursive formatter, it is exposed on the public Expression type.
    /// See fmt_data that is only implemented on the vtable trait.
    pub fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.scalar_fn().fmt_sql(self, f)
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
        if let Some(children) = Arc::get_mut(&mut self.children) {
            let mut children_to_drop = std::mem::take(children);

            while let Some(mut child) = children_to_drop.pop() {
                if let Some(expr_children) = Arc::get_mut(&mut child.children) {
                    children_to_drop.append(expr_children);
                }
            }
        }
    }
}
