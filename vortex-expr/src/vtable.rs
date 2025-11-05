// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arcref::ArcRef;
use vortex_array::ArrayRef;
use vortex_dtype::{DType, FieldPath};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::expression::Expression;
use crate::{ExprId, ExpressionView, StatsCatalog};

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
    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()>;

    /// Returns the name of the nth child of the expr.
    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName;

    /// Format this expression in nice human-readable SQL-style format
    ///
    /// The implementation should recursively format child expressions by calling
    /// `expr.child(i).fmt_sql(f)`.
    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> fmt::Result;

    /// Format only the instance data for this expression.
    ///
    /// Defaults to a debug representation of the instance data.
    #[allow(clippy::use_debug)]
    fn fmt_data(&self, instance: &Self::Instance, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", instance)
    }

    /// Compute the return [`DType`] of the expression if evaluated in the given scope.
    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType>;

    /// Evaluate the expression in the given scope.
    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef>;

    /// See [`crate::Expression::stat_falsification`].
    fn stat_falsification(
        &self,
        _expr: &ExpressionView<Self>,
        _catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        None
    }

    /// See [`crate::Expression::stat_max`].
    fn stat_max(
        &self,
        _expr: &ExpressionView<Self>,
        _catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        None
    }

    /// See [`crate::Expression::stat_min`].
    fn stat_min(
        &self,
        _expr: &ExpressionView<Self>,
        _catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        None
    }

    /// See [`crate::Expression::stat_nan_count`].
    fn stat_nan_count(
        &self,
        _expr: &ExpressionView<Self>,
        _catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        None
    }

    /// See [`crate::Expression::stat_field_path`].
    fn stat_field_path(&self, _expr: &ExpressionView<Self>) -> Option<FieldPath> {
        None
    }
}

/// Factory functions for static vtables.
pub trait VTableExt: VTable {
    fn new_expr(
        &'static self,
        instance: Self::Instance,
        children: impl Into<Arc<[Expression]>>,
    ) -> Expression {
        Self::try_new_expr(self, instance, children)
            .vortex_expect("Failed to create expression instance")
    }

    fn try_new_expr(
        &'static self,
        instance: Self::Instance,
        children: impl Into<Arc<[Expression]>>,
    ) -> VortexResult<Expression> {
        Expression::try_new(
            ExprVTable::from_static(self),
            Arc::new(instance),
            children.into(),
        )
    }
}
impl<V: VTable> VTableExt for V {}

/// A reference to the name of a child expression.
pub type ChildName = ArcRef<str>;

/// A placeholder vtable implementation for unsupported optional functionality of an expression.
pub struct NotSupported;

/// An object-safe trait for dynamic dispatch of Vortex expression vtables.
///
/// This trait is automatically implemented via the [`VTableAdapter`] for any type that
/// implements [`VTable`], and lifts the associated types into dynamic trait objects.
pub trait DynExprVTable: 'static + Send + Sync + private::Sealed {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> ExprId;
    fn serialize(&self, instance: &dyn Any) -> VortexResult<Option<Vec<u8>>>;
    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Arc<dyn Any + Send + Sync>>>;
    fn child_name(&self, instance: &dyn Any, child_idx: usize) -> ChildName;
    fn validate(&self, expression: &Expression) -> VortexResult<()>;
    fn fmt_sql(&self, expression: &Expression, f: &mut Formatter<'_>) -> fmt::Result;
    fn fmt_data(&self, instance: &dyn Any, f: &mut Formatter<'_>) -> fmt::Result;
    fn return_dtype(&self, expression: &Expression, scope: &DType) -> VortexResult<DType>;
    fn evaluate(&self, expression: &Expression, scope: &ArrayRef) -> VortexResult<ArrayRef>;

    fn stat_falsification(
        &self,
        expression: &Expression,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression>;
    fn stat_max(
        &self,
        expression: &Expression,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression>;
    fn stat_min(
        &self,
        expression: &Expression,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression>;
    fn stat_nan_count(
        &self,
        expression: &Expression,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression>;
    fn stat_field_path(&self, expression: &Expression) -> Option<FieldPath>;

    fn dyn_eq(&self, instance: &dyn Any, other: &dyn Any) -> bool;
    fn dyn_hash(&self, instance: &dyn Any, state: &mut dyn Hasher);
}

#[repr(transparent)]
pub struct VTableAdapter<V>(V);

impl<V: VTable> DynExprVTable for VTableAdapter<V> {
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline(always)]
    fn id(&self) -> ExprId {
        V::id(&self.0)
    }

    fn serialize(&self, instance: &dyn Any) -> VortexResult<Option<Vec<u8>>> {
        let instance = instance
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        V::serialize(&self.0, instance)
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Arc<dyn Any + Send + Sync>>> {
        Ok(V::deserialize(&self.0, metadata)?
            .map(|data| Arc::new(data) as Arc<dyn Any + Send + Sync>))
    }

    fn child_name(&self, instance: &dyn Any, child_idx: usize) -> ChildName {
        let instance = instance
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        V::child_name(&self.0, instance, child_idx)
    }

    fn validate(&self, expression: &Expression) -> VortexResult<()> {
        let expr = ExpressionView::new(expression);
        V::validate(&self.0, &expr)
    }

    fn fmt_sql(&self, expression: &Expression, f: &mut Formatter<'_>) -> fmt::Result {
        let expr = ExpressionView::new(expression);
        V::fmt_sql(&self.0, &expr, f)
    }

    fn fmt_data(&self, instance: &dyn Any, f: &mut Formatter<'_>) -> fmt::Result {
        let instance = instance
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        V::fmt_data(&self.0, instance, f)
    }

    fn return_dtype(&self, expression: &Expression, scope: &DType) -> VortexResult<DType> {
        let expr = ExpressionView::new(expression);
        V::return_dtype(&self.0, &expr, scope)
    }

    fn evaluate(&self, expression: &Expression, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let expr = ExpressionView::new(expression);
        V::evaluate(&self.0, &expr, scope)
    }

    fn stat_falsification(
        &self,
        expression: &Expression,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        let expr = ExpressionView::new(expression);
        V::stat_falsification(&self.0, &expr, catalog)
    }

    fn stat_max(
        &self,
        expression: &Expression,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        let expr = ExpressionView::new(expression);
        V::stat_max(&self.0, &expr, catalog)
    }

    fn stat_min(
        &self,
        expression: &Expression,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        let expr = ExpressionView::new(expression);
        V::stat_min(&self.0, &expr, catalog)
    }

    fn stat_nan_count(
        &self,
        expression: &Expression,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        let expr = ExpressionView::new(expression);
        V::stat_nan_count(&self.0, &expr, catalog)
    }

    fn stat_field_path(&self, expression: &Expression) -> Option<FieldPath> {
        let expr = ExpressionView::new(expression);
        V::stat_field_path(&self.0, &expr)
    }

    fn dyn_eq(&self, instance: &dyn Any, other: &dyn Any) -> bool {
        let this_instance = instance
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        let other_instance = other
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        this_instance == other_instance
    }

    fn dyn_hash(&self, instance: &dyn Any, mut state: &mut dyn Hasher) {
        let this_instance = instance
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        this_instance.hash(&mut state);
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
    pub const fn from_static<V: VTable>(vtable: &'static V) -> Self {
        // SAFETY: We can safely cast the vtable to a VTableAdapter since it has the same layout.
        let adapted: &'static VTableAdapter<V> =
            unsafe { &*(vtable as *const V as *const VTableAdapter<V>) };
        Self(ArcRef::new_ref(adapted as &'static dyn DynExprVTable))
    }

    /// Returns the ID of this vtable.
    pub fn id(&self) -> ExprId {
        self.0.id()
    }

    /// Returns whether this vtable is of a given type.
    pub fn is<V: VTable>(&self) -> bool {
        self.0.as_any().is::<VTableAdapter<V>>()
    }

    /// Returns the typed VTable for this expression.
    pub fn as_opt<V: VTable>(&self) -> Option<&V> {
        self.0
            .as_any()
            .downcast_ref::<VTableAdapter<V>>()
            .map(|adapter| &adapter.0)
    }

    /// Deserialize an instance of this expression vtable from metadata.
    pub fn deserialize(
        &self,
        metadata: &[u8],
        children: Arc<[Expression]>,
    ) -> VortexResult<Expression> {
        let instance_data = self.as_dyn().deserialize(metadata)?.ok_or_else(|| {
            vortex_err!(
                "Expression vtable {} is not deserializable",
                self.as_dyn().id()
            )
        })?;
        Expression::try_new(self.clone(), instance_data, children)
    }
}

impl PartialEq for ExprVTable {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}
impl Eq for ExprVTable {}

impl Display for ExprVTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_dyn().id())
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
    use crate::exprs::between::between;
    use crate::exprs::binary::{and, checked_add, eq, gt, gt_eq, lt, lt_eq, not_eq, or};
    use crate::exprs::cast::cast;
    use crate::exprs::get_item::{col, get_item};
    use crate::exprs::is_null::is_null;
    use crate::exprs::list_contains::list_contains;
    use crate::exprs::literal::lit;
    use crate::exprs::merge::merge;
    use crate::exprs::not::not;
    use crate::exprs::pack::pack;
    use crate::exprs::root::root;
    use crate::exprs::select::{select, select_exclude};
    use crate::proto::{ExprSerializeProtoExt, deserialize_expr_proto};
    use crate::session::{ExprRegistry, ExprSession};

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
        let serialized_pb = (&expr).serialize_proto()?;
        let deserialized_expr = deserialize_expr_proto(&serialized_pb, registry)?;

        assert_eq!(&expr, &deserialized_expr);

        Ok(())
    }
}
