// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{display, AnalysisExpr, ExprInstance, ExprVTable, ScopeVar, StatsCatalog, VTable};
use std::any::Any;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use vortex_array::ArrayRef;
use vortex_dtype::{DType, FieldPath};
use vortex_error::{VortexExpect, VortexResult};

/// A node in a Vortex expression tree.
///
/// Expressions represent scalar computations that can be performed on data. Each
/// expression consists of an encoding (vtable), heap-allocated metadata, and child expressions.
#[derive(Clone, Debug)]
pub struct Expression {
    /// The vtable for this expression.
    vtable: ExprVTable,
    /// The instance data for this expression.
    instance: Arc<dyn Any>,
    /// Any children of this expression.
    children: Arc<[Expression]>,
}

impl Expression {
    /// Creates a new expression with the given encoding, metadata, and children.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided `encoding` is not compatible with the
    /// `metadata` and `children` or the encoding's own validation logic fails.
    pub fn try_new(
        vtable: ExprVTable,
        instance: Arc<dyn Any>,
        children: Arc<[Expression]>,
    ) -> VortexResult<Self> {
        // Validate that the encoding is compatible with the metadata and children.
        vtable
            .as_dyn()
            .validate(instance.as_ref(), children.as_ref())?;
        Ok(Self {
            vtable,
            instance,
            children,
        })
    }

    /// Creates a new expression with the given encoding, metadata, and children.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the provided `encoding` is compatible with the
    /// `metadata` and `children`. Failure to do so may lead to undefined behavior
    ///  when the expression is used.
    pub unsafe fn new_unchecked(
        vtable: ExprVTable,
        instance: Arc<dyn Any>,
        children: Arc<[Expression]>,
    ) -> Self {
        Self {
            vtable,
            instance,
            children,
        }
    }

    /// Returns if the expression is an instance of the given vtable.
    pub fn is<V: VTable>(&self) -> bool {
        self.vtable.as_dyn().as_any().is::<V>()
    }

    /// Returns a typed view of this expression for the given vtable.
    ///
    /// # Panics
    ///
    /// Panics if the expression's encoding or metadata cannot be cast to the specified vtable.
    pub fn as_view<V: VTable>(&self) -> ExprInstance<'_, V> {
        ExprInstance::new(
            self.instance
                .as_any()
                .downcast_ref::<V::Instance>()
                .vortex_expect("Failed to downcast expression instance to expected type"),
            &self.children,
        )
    }

    /// Returns a typed view of this expression for the given vtable, if the types match.
    pub fn as_view_opt<V: VTable>(&self) -> Option<ExprInstance<'_, V>> {
        self.vtable.as_dyn().as_any().downcast_ref::<V>().map(|_v| {
            ExprInstance::new(
                self.instance
                    .as_any()
                    .downcast_ref::<V::Instance>()
                    .vortex_expect("Failed to downcast expression instance to expected type"),
                &self.children,
            )
        })
    }

    /// Returns the children of this expression.
    pub fn children(&self) -> &Arc<[Expression]> {
        &self.children
    }

    /// Replace the children of this expression with the provided new children.
    pub fn with_children(mut self, children: Arc<[Expression]>) -> VortexResult<Self> {
        self.vtable
            .as_dyn()
            .validate(self.instance.as_ref(), &children)?;
        self.children = children;
        Ok(self)
    }

    /// Computes the return dtype of this expression given the input dtype.
    pub fn return_dtype(&self, scope: &DType) -> VortexResult<DType> {
        self.vtable
            .as_dyn()
            .return_dtype(self.instance.as_ref(), self.children.as_ref(), scope)
    }

    /// Evaluates the expression in the given scope.
    pub fn evaluate(&self, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        self.vtable
            .as_dyn()
            .evaluate(self.instance.as_ref(), self.children.as_ref(), scope)
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
    pub fn stat_falsification(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        self.vtable.as_dyn().stat_falsification(
            self.instance.as_ref(),
            self.children().as_ref(),
            catalog,
        )
    }

    /// An expression for the upper non-null bound of this expression, if available.
    ///
    /// This function returns None if there is no upper bound or it is difficult to compute.
    ///
    /// The returned expression evaluates to null if the maximum value is unknown. In that case, you
    /// _must not_ assume the array is empty _nor_ may you assume the array only contains non-null
    /// values.
    pub fn max(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        self.vtable
            .as_dyn()
            .max(self.instance.as_ref(), self.children.as_ref(), catalog)
    }

    /// An expression for the lower non-null bound of this expression, if available.
    ///
    /// See [AnalysisExpr::max] for important details.
    pub fn min(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        self.vtable
            .as_dyn()
            .min(self.instance.as_ref(), self.children.as_ref(), catalog)
    }

    /// An expression for the NaN count for a column, if available.
    ///
    /// This method returns `None` if the NaNCount stat is unknown.
    pub fn nan_count(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        self.vtable
            .as_dyn()
            .nan_count(self.instance.as_ref(), self.children.as_ref(), catalog)
    }

    pub fn field_path(&self) -> Option<FieldPath> {
        self.vtable
            .as_dyn()
            .field_path(self.instance.as_ref(), self.children.as_ref())
    }

    /// Format the expression as a compact string.
    pub fn fmt_compact(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.vtable
            .as_dyn()
            .fmt_compact(self.instance.as_ref(), self.children.as_ref(), f)
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
    /// # use vortex_dtype::{DType, Nullability, PType};
    /// # use vortex_expr::{and, cast, eq, get_item, gt, lit, not, root, select, IntoExpr, LikeExpr};
    /// // Build a complex nested expression
    /// let complex_expr = select(
    ///     ["result"],
    ///     and(
    ///         not(eq(get_item("status", root()), lit("inactive"))),
    ///         and(
    ///             LikeExpr::new(get_item("name", root()), lit("%admin%"), false, false).into_expr(),
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
        display::DisplayTreeExpr(self)
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        self.vtable.as_dyn().id() == other.vtable.as_dyn().id()
            && self
                .vtable
                .as_dyn()
                .dyn_eq(self.instance.as_ref(), other.instance.as_ref())
            && self.children.eq(&other.children)
    }
}
impl Eq for Expression {}

impl Hash for Expression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vtable.as_dyn().id().hash(state);
        self.vtable.as_dyn().dyn_hash(self.instance.as_ref(), state);
        self.children.hash(state);
    }
}
