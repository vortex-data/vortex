// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use vortex_dtype::{DType, FieldPath};
use vortex_error::{VortexExpect, VortexResult};

use crate::ArrayRef;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::{ChildName, ExprId, ExprVTable, ExpressionView, StatsCatalog, VTable};

/// A node in a Vortex expression tree.
///
/// Expressions represent scalar computations that can be performed on data. Each
/// expression consists of an encoding (vtable), heap-allocated metadata, and child expressions.
#[derive(Clone)]
pub struct Expression {
    /// The vtable for this expression.
    vtable: ExprVTable,
    /// The instance data for this expression.
    data: Arc<dyn Any + Send + Sync>,
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
        data: Arc<dyn Any + Send + Sync>,
        children: Arc<[Expression]>,
    ) -> VortexResult<Self> {
        let this = Self {
            vtable,
            data,
            children,
        };
        // Validate that the encoding is compatible with the metadata and children.
        this.vtable.as_dyn().validate(&this)?;
        Ok(this)
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
        data: Arc<dyn Any + Send + Sync>,
        children: Arc<[Expression]>,
    ) -> Self {
        Self {
            vtable,
            data,
            children,
        }
    }

    /// Returns if the expression is an instance of the given vtable.
    pub fn is<V: VTable>(&self) -> bool {
        self.vtable.is::<V>()
    }

    /// Returns a typed view of this expression for the given vtable.
    ///
    /// # Panics
    ///
    /// Panics if the expression's encoding or metadata cannot be cast to the specified vtable.
    pub fn as_<V: VTable>(&self) -> ExpressionView<'_, V> {
        ExpressionView::maybe_new(self).vortex_expect("Failed to downcast expression {} to {}")
    }

    /// Returns a typed view of this expression for the given vtable, if the types match.
    pub fn as_opt<V: VTable>(&self) -> Option<ExpressionView<'_, V>> {
        ExpressionView::maybe_new(self)
    }

    /// Returns the expression ID.
    pub fn id(&self) -> ExprId {
        self.vtable.as_dyn().id()
    }

    /// Returns the expression's vtable.
    pub fn vtable(&self) -> &ExprVTable {
        &self.vtable
    }

    /// Returns the opaque data of the expression.
    pub fn data(&self) -> &Arc<dyn Any + Send + Sync> {
        &self.data
    }

    /// Returns the children of this expression.
    pub fn children(&self) -> &Arc<[Expression]> {
        &self.children
    }

    /// Returns the n'th child of this expression.
    pub fn child(&self, n: usize) -> &Expression {
        &self.children[n]
    }

    /// Returns the name of the n'th child of this expression.
    pub fn child_name(&self, n: usize) -> ChildName {
        self.vtable.as_dyn().child_name(self.data().as_ref(), n)
    }

    /// Replace the children of this expression with the provided new children.
    pub fn with_children(mut self, children: impl Into<Arc<[Expression]>>) -> VortexResult<Self> {
        self.children = children.into();
        self.vtable.as_dyn().validate(&self)?;
        Ok(self)
    }

    /// Returns the serialized metadata for this expression.
    pub fn serialize_metadata(&self) -> VortexResult<Option<Vec<u8>>> {
        self.vtable.as_dyn().serialize(self.data.as_ref())
    }

    /// Computes the return dtype of this expression given the input dtype.
    pub fn return_dtype(&self, scope: &DType) -> VortexResult<DType> {
        self.vtable.as_dyn().return_dtype(self, scope)
    }

    /// Evaluates the expression in the given scope.
    pub fn evaluate(&self, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        self.vtable.as_dyn().evaluate(self, scope)
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
        self.vtable.as_dyn().stat_falsification(self, catalog)
    }

    /// An expression for the upper non-null bound of this expression, if available.
    ///
    /// This function returns None if there is no upper bound, or it is difficult to compute.
    ///
    /// The returned expression evaluates to null if the maximum value is unknown. In that case, you
    /// _must not_ assume the array is empty _nor_ may you assume the array only contains non-null
    /// values.
    pub fn stat_max(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        self.vtable.as_dyn().stat_max(self, catalog)
    }

    /// An expression for the lower non-null bound of this expression, if available.
    ///
    /// See [`Expression::stat_max`] for important details.
    pub fn stat_min(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        self.vtable.as_dyn().stat_min(self, catalog)
    }

    /// An expression for the NaN count for a column, if available.
    ///
    /// This method returns `None` if the NaNCount stat is unknown.
    pub fn stat_nan_count(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        self.vtable.as_dyn().stat_nan_count(self, catalog)
    }

    // TODO(ngates): I'm not sure what this is really for? We need to clean up stats compute for
    //  expressions.
    pub fn stat_field_path(&self) -> Option<FieldPath> {
        self.vtable.as_dyn().stat_field_path(self)
    }

    /// Format the expression as a compact string.
    ///
    /// Since this is a recursive formatter, it is exposed on the public Expression type.
    /// See fmt_data that is only implemented on the vtable trait.
    pub fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.vtable.as_dyn().fmt_sql(self, f)
    }

    /// Format the instance data of the expression as a string for rendering..
    pub fn fmt_data(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.vtable.as_dyn().fmt_data(self.data().as_ref(), f)
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
    /// # use vortex_array::compute::LikeOptions;
    /// # use vortex_array::expr::VTableExt;
    /// # use vortex_dtype::{DType, Nullability, PType};
    /// # use vortex_array::expr::{and, cast, eq, get_item, gt, lit, not, root, select, Like};
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

struct FormatExpressionData<'a> {
    vtable: &'a ExprVTable,
    data: &'a Arc<dyn Any + Send + Sync>,
}

impl<'a> Debug for FormatExpressionData<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.vtable.as_dyn().fmt_data(self.data.as_ref(), f)
    }
}

impl Debug for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Expression")
            .field("vtable", &self.vtable)
            .field(
                "data",
                &FormatExpressionData {
                    vtable: &self.vtable,
                    data: &self.data,
                },
            )
            .field("children", &self.children)
            .finish()
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        self.vtable.as_dyn().id() == other.vtable.as_dyn().id()
            && self
                .vtable
                .as_dyn()
                .dyn_eq(self.data.as_ref(), other.data.as_ref())
            && self.children.eq(&other.children)
    }
}
impl Eq for Expression {}

impl Hash for Expression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vtable.as_dyn().id().hash(state);
        self.vtable.as_dyn().dyn_hash(self.data.as_ref(), state);
        self.children.hash(state);
    }
}
