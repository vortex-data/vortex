// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::Expression;
use crate::{AnalysisExpr, ExprId, StatsCatalog};
use arcref::ArcRef;
use std::any::Any;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;
use vortex_array::ArrayRef;
use vortex_dtype::{DType, FieldPath};
use vortex_error::{VortexExpect, VortexResult};
use vortex_session::SessionVar;

/// The vtable trait for a Vortex expression.
///
/// This trait defines the interface for expression vtables, including methods for
/// serialization, deserialization, validation, child naming, return type computation,
/// and evaluation.
///
/// This trait is non-object safe and allows the implementer to make use of associated types
/// for improved type safety, while allowing Vortex to enforce runtime checks on the inputs and
/// outputs of each function.
///
/// The [`VTable`] trait should be implemented for a struct that holds global data across
/// all instances of the expression. In almost all cases, this struct will be an empty unit
/// struct, since most expressions do not require any global state.
pub trait VTable: 'static + Sized + Send + Sync {
    /// Instance data for this expression.
    type Instance: 'static + Send + Sync + Debug + PartialEq + Eq + Hash;

    /// Returns the ID of the expr vtable.
    fn id(&self) -> ExprId;

    /// Serialize the metadata for the expression.
    ///
    /// Should return `Ok(None)` if the expression is not serializable, and `Ok(vec![])` if it is
    /// serializable but has no metadata.
    fn serialize(&self, _instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Deserialize an instance of this expression.
    ///
    /// Returns `Ok(None)` if the expression is not serializable.
    fn deserialize(&self, _metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        Ok(None)
    }

    /// Validate the metadata and children for the expression.
    fn validate(&self, expr: &ExprInstance<Self>) -> VortexResult<()>;

    /// Returns the name of the nth child of the expr.
    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName;

    /// Format this expression in nice human-readable format
    ///
    /// Specifically, this format is designed to be readable by humans, at the
    /// expense of details. Use `Display` or `Debug` for more detailed
    /// representation.
    fn fmt_compact(&self, expr: &ExprInstance<Self>, f: &mut Formatter<'_>) -> fmt::Result;

    /// Compute the return [`DType`] of the expression if evaluated in the given scope.
    fn return_dtype(&self, expr: &ExprInstance<Self>, scope: &DType) -> VortexResult<DType>;

    /// Evaluate the expression in the given scope.
    fn evaluate(&self, expr: &ExprInstance<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef>;

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
    fn stat_falsification(
        &self,
        expr: &ExprInstance<Self>,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        None
    }

    /// An expression for the upper non-null bound of this expression, if available.
    ///
    /// This function returns None if there is no upper bound or it is difficult to compute.
    ///
    /// The returned expression evaluates to null if the maximum value is unknown. In that case, you
    /// _must not_ assume the array is empty _nor_ may you assume the array only contains non-null
    /// values.
    fn max(
        &self,
        expr: &ExprInstance<Self>,
        _catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        None
    }

    /// An expression for the lower non-null bound of this expression, if available.
    ///
    /// See [AnalysisExpr::max] for important details.
    fn min(
        &self,
        expr: &ExprInstance<Self>,
        _catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        None
    }

    /// An expression for the NaN count for a column, if available.
    ///
    /// This method returns `None` if the NaNCount stat is unknown.
    fn nan_count(
        &self,
        expr: &ExprInstance<Self>,
        _catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        None
    }

    fn field_path(&self, expr: &ExprInstance<Self>) -> Option<FieldPath> {
        None
    }
}

/// Factory functions for static vtables.
pub trait VTableExt: VTable {
    fn new(
        self: &'static Self,
        instance: Self::Instance,
        children: impl Into<Arc<[Expression]>>,
    ) -> Expression {
        Self::try_new(self, instance, children)
            .vortex_expect("Failed to create expression instance")
    }

    fn try_new(
        self: &'static Self,
        instance: Self::Instance,
        children: impl Into<Arc<[Expression]>>,
    ) -> VortexResult<Expression> {
        Expression::try_new(ExprVTable::from_static(self), Arc::new(instance), children)
    }
}
impl<V: VTable> VTableExt for V {}

/// A reference to the name of a child expression.
pub type ChildName = ArcRef<str>;

/// A placeholder vtable implementation for unsupported optional functionality of an expression.
pub struct NotSupported;

/// A typed expr over an instance of a Vortex expression for a specific vtable.
pub struct ExprInstance<'a, V: VTable> {
    instance: &'a V::Instance,
    // FIXME(ngates): this is tough, because in theory we shouldn't known about ExprNode in general?
    //  e.g. if we hold the expression in an Array, we don't want to have ExprNode dependencies.
    children: &'a [Expression],
}

impl<'a, V: VTable> ExprInstance<'a, V> {
    pub fn from_dyn(instance: &dyn Any, children: &'a [Expression]) -> Self {
        let instance = instance
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        Self { instance, children }
    }

    pub fn new(instance: &'a V::Instance, children: &'a [Expression]) -> Self {
        Self { instance, children }
    }

    pub fn children(&self) -> &'a [Expression] {
        self.children
    }

    pub fn child(&self, idx: usize) -> &Expression {
        &self.children[idx]
    }
}

impl<'a, V: VTable> Deref for ExprInstance<'a, V> {
    type Target = V::Instance;

    fn deref(&self) -> &Self::Target {
        self.instance
    }
}

/// An object-safe trait for dynamic dispatch of Vortex expression vtables.
///
/// This trait is automatically implemented via the [`VTableAdapter`] for any type that
/// implements [`VTable`], and lifts the associated types into dynamic trait objects.
pub trait DynExprVTable: 'static + Send + Sync + private::Sealed {
    fn id(&self) -> ExprId;
    fn validate(&self, instance: &dyn Any, children: &[Expression]) -> VortexResult<()>;
    fn fmt_compact(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        f: &mut Formatter<'_>,
    ) -> fmt::Result;
    fn return_dtype(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        scope: &DType,
    ) -> VortexResult<DType>;
    fn evaluate(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef>;

    fn stat_falsification(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression>;
    fn max(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression>;
    fn min(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression>;
    fn nan_count(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression>;
    fn field_path(&self, instance: &dyn Any, children: &[Expression]) -> Option<FieldPath>;

    fn dyn_eq(&self, instance: &dyn Any, other: &dyn Any) -> bool;
    fn dyn_hash(&self, instance: &dyn Any, state: &mut dyn Hasher);
}

#[repr(transparent)]
pub struct VTableAdapter<V>(V);

impl<V: VTable> DynExprVTable for VTableAdapter<V> {
    fn id(&self) -> ExprId {
        V::id(&self.0)
    }

    fn validate(&self, instance: &dyn Any, children: &[Expression]) -> VortexResult<()> {
        let expr = ExprInstance::from_dyn(instance, children);
        V::validate(&self.0, &expr)
    }

    fn fmt_compact(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        let expr = ExprInstance::from_dyn(instance, children);
        V::fmt_compact(&self.0, &expr, f)
    }

    fn return_dtype(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        scope: &DType,
    ) -> VortexResult<DType> {
        let expr = ExprInstance::from_dyn(instance, children);
        V::return_dtype(&self.0, &expr, scope)
    }

    fn evaluate(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let expr = ExprInstance::from_dyn(instance, children);
        V::evaluate(&self.0, &expr, scope)
    }

    fn stat_falsification(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        let expr = ExprInstance::from_dyn(instance, children);
        V::stat_falsification(&self.0, &expr, catalog)
    }

    fn max(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        let expr = ExprInstance::from_dyn(instance, children);
        V::max(&self.0, &expr, catalog)
    }

    fn min(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        let expr = ExprInstance::from_dyn(instance, children);
        V::min(&self.0, &expr, catalog)
    }

    fn nan_count(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        let expr = ExprInstance::from_dyn(instance, children);
        V::nan_count(&self.0, &expr, catalog)
    }

    fn field_path(&self, instance: &dyn Any, children: &[Expression]) -> Option<FieldPath> {
        let expr = ExprInstance::from_dyn(instance, children);
        V::field_path(&self.0, &expr)
    }

    fn dyn_eq(&self, instance: &dyn Any, other: &dyn Any) -> bool {
        let this_instance = instance
            .as_any()
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        let other_instance = other
            .as_any()
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        this_instance == other_instance
    }

    fn dyn_hash(&self, instance: &dyn Any, state: &mut dyn Hasher) {
        let this_instance = instance
            .as_any()
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        this_instance.hash(state);
    }
}

mod private {
    use crate::{VTable, VTableAdapter};

    pub trait Sealed {}
    impl<V: VTable> Sealed for VTableAdapter<V> {}
}

/// A Vortex expression vtable, used to deserialize or instantiate expressions dynamically.
#[derive(Clone)]
pub struct ExprVTable(ArcRef<dyn DynExprVTable>);

impl ExprVTable {
    /// Only the vortex-expr crate can actually invoke the vtable methods.
    /// All other users must go via session extensions.
    pub(crate) fn as_dyn(&self) -> &dyn DynExprVTable {
        self.0.as_ref()
    }

    /// Creates a new [`ExprVTable`] from a static reference to a vtable.
    pub const fn from_static<V: VTable>(vtable: &V) -> Self {
        // SAFETY: We can safely cast the vtable to a VTableAdapter since it has the same layout.
        let adapted = unsafe { &*(vtable as *const V as *const VTableAdapter<V>) };
        Self(ArcRef::from(adapted))
    }
}

impl Debug for ExprVTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_dyn().id())
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;
    use crate::proto::{deserialize_expr_proto, ExprSerializeProtoExt};
    use crate::session::{ExprRegistry, ExprSession};
    use crate::*;

    #[fixture]
    #[once]
    fn registry() -> ExprRegistry {
        ExprSession::default().registry().clone()
    }

    #[rstest]
    // Root and selection expressions
    #[case(root())]
    #[case(select(["hello", "world"], root()))]
    #[case(select_exclude(["world", "hello"], root()))]
    // Literal expressions
    #[case(lit(42i32))]
    #[case(lit(std::f64::consts::PI))]
    #[case(lit(true))]
    #[case(lit("hello"))]
    // Column access expressions
    #[case(col("column_name"))]
    #[case(get_item("field", root()))]
    // Binary comparison expressions
    #[case(eq(col("a"), lit(10)))]
    #[case(not_eq(col("a"), lit(10)))]
    #[case(gt(col("a"), lit(10)))]
    #[case(gt_eq(col("a"), lit(10)))]
    #[case(lt(col("a"), lit(10)))]
    #[case(lt_eq(col("a"), lit(10)))]
    // Logical expressions
    #[case(and(col("a"), col("b")))]
    #[case(or(col("a"), col("b")))]
    #[case(not(col("a")))]
    // Arithmetic expressions
    #[case(checked_add(col("a"), lit(5)))]
    // Null check expressions
    #[case(is_null(col("nullable_col")))]
    // Type casting expressions
    #[case(cast(
        col("a"),
        DType::Primitive(vortex_dtype::PType::I64, vortex_dtype::Nullability::NonNullable)
    ))]
    // Between expressions
    #[case(between(col("a"), lit(10), lit(20), vortex_array::compute::BetweenOptions { lower_strict: vortex_array::compute::StrictComparison::NonStrict, upper_strict: vortex_array::compute::StrictComparison::NonStrict }))]
    // List contains expressions
    #[case(list_contains(col("list_col"), lit("item")))]
    // Pack expressions - creating struct from fields
    #[case(pack([("field1", col("a")), ("field2", col("b"))], vortex_dtype::Nullability::NonNullable))]
    // Merge expressions - merging struct expressions
    #[case(merge([col("struct1"), col("struct2")]))]
    // Complex nested expressions
    #[case(and(gt(col("a"), lit(0)), lt(col("a"), lit(100))))]
    #[case(or(is_null(col("a")), eq(col("a"), lit(0))))]
    #[case(not(and(eq(col("status"), lit("active")), gt(col("age"), lit(18)))))]
    fn text_expr_serde_round_trip(
        registry: &ExprRegistry,
        #[case] expr: Expression,
    ) -> anyhow::Result<()> {
        let serialized_pb = expr.serialize_proto()?;
        let deserialized_expr = deserialize_expr_proto(&serialized_pb, registry)?;

        assert_eq!(&expr, &deserialized_expr);

        Ok(())
    }
}
