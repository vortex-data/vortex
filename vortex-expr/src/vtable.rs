// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arcref::ArcRef;
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::v2::Expression;
use crate::{AnalysisVTable, ExprId};

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

    // TODO(ngates): inline this? Not sure we really want to spread this stuff around, although
    //  it does make testing setup cleaner.
    type AnalysisVTable: AnalysisVTable<Self>;

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
    fn child_name(&self, child_idx: usize) -> ChildName;

    /// Compute the return [`DType`] of the expression if evaluated in the given scope.
    fn return_dtype(&self, expr: &ExprInstance<Self>, scope: &DType) -> VortexResult<DType>;

    /// Evaluate the expression in the given scope.
    fn evaluate(&self, expr: &ExprInstance<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef>;
}

/// Factory functions for static vtables.
pub trait VTableExt: VTable {
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

/// A typed view over an instance of a Vortex expression for a specific vtable.
pub struct ExprInstance<'a, V: VTable> {
    instance: &'a V::Instance,
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
}

#[repr(transparent)]
pub struct VTableAdapter<V>(V);

impl<V: VTable> DynExprVTable for VTableAdapter<V> {
    fn id(&self) -> ExprId {
        V::id(&self.0)
    }

    fn validate(&self, instance: &dyn Any, children: &[Expression]) -> VortexResult<()> {
        let view = ExprInstance::from_dyn(instance, children);
        V::validate(&self.0, &view)
    }

    fn return_dtype(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        scope: &DType,
    ) -> VortexResult<DType> {
        let view = ExprInstance::from_dyn(instance, children);
        V::return_dtype(&self.0, &view, scope)
    }

    fn evaluate(
        &self,
        instance: &dyn Any,
        children: &[Expression],
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let view = ExprInstance::from_dyn(instance, children);
        V::evaluate(&self.0, &view, scope)
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
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.id())
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
