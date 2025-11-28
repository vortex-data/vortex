// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_vector::Vector;
use vortex_vector::VectorOps;

use crate::ArrayRef;
use crate::expr::ExprId;
use crate::expr::ExpressionView;
use crate::expr::StatsCatalog;
use crate::expr::expression::Expression;
use crate::expr::stats::Stat;

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
    fn serialize(&self, instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        _ = instance;
        Ok(None)
    }

    /// Deserialize an instance of this expression.
    ///
    /// Returns `Ok(None)` if the expression is not serializable.
    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        _ = metadata;
        Ok(None)
    }

    /// Validate the metadata and children for the expression.
    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()>;

    /// Returns the name of the nth child of the expr.
    fn child_name(&self, instance: &Self::Instance, child_idx: usize) -> ChildName;

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

    /// Execute the expression on the given vector with the given dtype.
    fn execute(&self, data: &Self::Instance, args: ExecutionArgs) -> VortexResult<Vector> {
        _ = data;
        let _args = args;
        // TODO(ngates): remove this once we port to vector execution
        // TODO(ngates): I think we should take/return an enum of Vector/Scalar.
        vortex_bail!("Expression {} does not support execution", self.id());
    }

    /// See [`Expression::stat_falsification`].
    fn stat_falsification(
        &self,
        expr: &ExpressionView<Self>,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        _ = expr;
        _ = catalog;
        None
    }

    /// See [`Expression::stat_expression`].
    fn stat_expression(
        &self,
        expr: &ExpressionView<Self>,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        _ = expr;
        _ = stat;
        _ = catalog;
        None
    }

    /// Returns whether this expression itself is null-sensitive. Conservatively default to *true*.
    ///
    /// An expression is null-sensitive if it directly operates on null values,
    /// such as `is_null`. Most expressions are not null-sensitive.
    ///
    /// The property we are interested in is if the expression (e) distributes over
    /// mask.
    /// Define a `mask(a, m)` expression that applies the boolean array `m` to the validity of the
    /// array `a`.
    /// An unary expression `e` to be null-sensitive iff forall arrays `a` and masks `m`.
    /// `e(mask(a, m)) == mask(e(a), m)`.
    /// This can be extended to an n-ary expression.
    ///
    /// This method only checks the expression itself, not its children. To check
    /// if an expression or any of its descendants are null-sensitive.
    fn is_null_sensitive(&self, instance: &Self::Instance) -> bool {
        _ = instance;
        true
    }

    /// Returns whether this expression itself is fallible. Conservatively default to *true*.
    ///
    /// An expression is runtime fallible is there is an input set that causes the expression to
    /// panic or return an error, for example checked_add is fallible if there is overflow.
    ///
    /// Note: this is only applicable to expressions that pass type-checking
    /// [`VTable::return_dtype`].
    fn is_fallible(&self, instance: &Self::Instance) -> bool {
        _ = instance;
        true
    }
}

/// Arguments for expression execution.
pub struct ExecutionArgs {
    /// The input vectors for the expression, one per child.
    pub vectors: Vec<Vector>,
    /// The input dtypes for the expression, one per child.
    pub dtypes: Vec<DType>,
    /// The row count of the execution scope.
    pub row_count: usize,
    /// The expected return dtype of the expression, as computed by [`Expression::return_dtype`].
    pub return_dtype: DType,
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
    fn execute(&self, data: &dyn Any, args: ExecutionArgs) -> VortexResult<Vector>;

    fn stat_falsification(
        &self,
        expression: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression>;
    fn stat_expression(
        &self,
        expression: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression>;

    /// See [`VTable::is_null_sensitive`].
    fn is_null_sensitive(&self, instance: &dyn Any) -> bool;
    /// See [`VTable::is_fallible`].
    fn is_fallible(&self, instance: &dyn Any) -> bool;

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

    fn execute(&self, data: &dyn Any, args: ExecutionArgs) -> VortexResult<Vector> {
        let data = data
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");

        let expected_row_count = args.row_count;
        #[cfg(debug_assertions)]
        let expected_dtype = args.return_dtype.clone();

        let result = V::execute(&self.0, data, args)?;

        assert_eq!(
            result.len(),
            expected_row_count,
            "Expression execution returned vector of length {}, but expected {}",
            result.len(),
            expected_row_count,
        );

        // In debug mode, validate that the output dtype matches the expected return dtype.
        #[cfg(debug_assertions)]
        {
            use vortex_error::vortex_ensure;
            use vortex_vector::vector_matches_dtype;
            vortex_ensure!(
                vector_matches_dtype(&result, &expected_dtype),
                "Expression execution invalid for dtype {}",
                expected_dtype
            );
        }

        Ok(result)
    }

    fn stat_falsification(
        &self,
        expression: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        let expr = ExpressionView::new(expression);
        V::stat_falsification(&self.0, &expr, catalog)
    }

    fn stat_expression(
        &self,
        expression: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        let expr = ExpressionView::new(expression);
        V::stat_expression(&self.0, &expr, stat, catalog)
    }

    fn is_null_sensitive(&self, instance: &dyn Any) -> bool {
        let instance = instance
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        V::is_null_sensitive(&self.0, instance)
    }

    fn is_fallible(&self, instance: &dyn Any) -> bool {
        let instance = instance
            .downcast_ref::<V::Instance>()
            .vortex_expect("Failed to downcast expression instance to expected type");
        V::is_fallible(&self.0, instance)
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
    use crate::expr::VTable;
    use crate::expr::VTableAdapter;

    pub trait Sealed {}
    impl<V: VTable> Sealed for VTableAdapter<V> {}
}

/// A Vortex expression vtable, used to deserialize or instantiate expressions dynamically.
#[derive(Clone)]
pub struct ExprVTable(ArcRef<dyn DynExprVTable>);

impl ExprVTable {
    /// Only the vortex-array crate can actually invoke the vtable methods.
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
    use rstest::fixture;
    use rstest::rstest;

    use super::*;
    use crate::expr::exprs::between::between;
    use crate::expr::exprs::binary::and;
    use crate::expr::exprs::binary::checked_add;
    use crate::expr::exprs::binary::eq;
    use crate::expr::exprs::binary::gt;
    use crate::expr::exprs::binary::gt_eq;
    use crate::expr::exprs::binary::lt;
    use crate::expr::exprs::binary::lt_eq;
    use crate::expr::exprs::binary::not_eq;
    use crate::expr::exprs::binary::or;
    use crate::expr::exprs::cast::cast;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::is_null::is_null;
    use crate::expr::exprs::list_contains::list_contains;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::merge::merge;
    use crate::expr::exprs::not::not;
    use crate::expr::exprs::pack::pack;
    use crate::expr::exprs::root::root;
    use crate::expr::exprs::select::select;
    use crate::expr::exprs::select::select_exclude;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::proto::deserialize_expr_proto;
    use crate::expr::session::ExprRegistry;
    use crate::expr::session::ExprSession;

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
    #[case(between(col("a"), lit(10), lit(20), crate::compute::BetweenOptions { lower_strict: crate::compute::StrictComparison::NonStrict, upper_strict: crate::compute::StrictComparison::NonStrict }))]
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
    ) -> VortexResult<()> {
        let serialized_pb = (&expr).serialize_proto()?;
        let deserialized_expr = deserialize_expr_proto(&serialized_pb, registry)?;

        assert_eq!(&expr, &deserialized_expr);

        Ok(())
    }
}
