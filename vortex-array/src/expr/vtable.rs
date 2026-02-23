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
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::ExprId;
use crate::expr::StatsCatalog;
use crate::expr::expression::Expression;
use crate::expr::scalar_fn::ScalarFn;
use crate::expr::stats::Stat;

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
    /// Options for this expression.
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// Returns the ID of the expr vtable.
    fn id(&self) -> ExprId;

    /// Serialize the options for this expression.
    ///
    /// Should return `Ok(None)` if the expression is not serializable, and `Ok(vec![])` if it is
    /// serializable but has no metadata.
    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        _ = options;
        Ok(None)
    }

    /// Deserialize the options of this expression.
    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_bail!("Expression {} is not deserializable", self.id());
    }

    /// Returns the arity of this expression.
    fn arity(&self, options: &Self::Options) -> Arity;

    /// Returns the name of the nth child of the expr.
    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName;

    /// Format this expression in nice human-readable SQL-style format
    ///
    /// The implementation should recursively format child expressions by calling
    /// `expr.child(i).fmt_sql(f)`.
    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result;

    /// Compute the return [`DType`] of the expression if evaluated over the given input types.
    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType>;

    /// Execute the expression over the input arguments.
    ///
    /// Implementations are encouraged to check their inputs for constant arrays to perform
    /// more optimized execution.
    ///
    /// If the input arguments cannot be directly used for execution (for example, an expression
    /// may require canonical input arrays), then the implementation should perform a single
    /// child execution and return a new [`crate::arrays::ScalarFnArray`] wrapping up the new child.
    ///
    /// This provides maximum opportunities for array-level optimizations using execute_parent
    /// kernels.
    fn execute(&self, options: &Self::Options, args: ExecutionArgs) -> VortexResult<ArrayRef>;

    /// Implement an abstract reduction rule over a tree of scalar functions.
    ///
    /// The [`ReduceNode`] can be used to traverse children, inspect their types, and
    /// construct the result expression.
    ///
    /// Return `Ok(None)` if no reduction is possible.
    fn reduce(
        &self,
        options: &Self::Options,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        _ = options;
        _ = node;
        _ = ctx;
        Ok(None)
    }

    /// Simplify the expression if possible.
    fn simplify(
        &self,
        options: &Self::Options,
        expr: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        _ = options;
        _ = expr;
        _ = ctx;
        Ok(None)
    }

    /// Simplify the expression if possible, without type information.
    fn simplify_untyped(
        &self,
        options: &Self::Options,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        _ = options;
        _ = expr;
        Ok(None)
    }

    /// See [`Expression::stat_falsification`].
    fn stat_falsification(
        &self,
        options: &Self::Options,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        _ = options;
        _ = expr;
        _ = catalog;
        None
    }

    /// See [`Expression::stat_expression`].
    fn stat_expression(
        &self,
        options: &Self::Options,
        expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        _ = options;
        _ = expr;
        _ = expr;
        _ = stat;
        _ = catalog;
        None
    }

    /// Returns an expression that evaluates to the validity of the result of this expression.
    ///
    /// If a validity expression cannot be constructed, returns `None` and the expression will
    /// be evaluated as normal before extracting the validity mask from the result.
    ///
    /// This is essentially a specialized form of a `reduce_parent`
    fn validity(
        &self,
        options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        _ = (options, expression);
        Ok(None)
    }

    /// Returns whether this expression itself is null-sensitive. Conservatively default to *true*.
    ///
    /// An expression is null-sensitive if it directly operates on null values,
    /// such as `is_null`. Most expressions are not null-sensitive.
    ///
    /// The property we are interested in is if the expression (e) distributes over `mask`.
    /// Define a `mask(a, m)` expression that applies the boolean array `m` to the validity of the
    /// array `a`.
    ///
    /// A unary expression `e` is not null-sensitive iff forall arrays `a` and masks `m`,
    /// `e(mask(a, m)) == mask(e(a), m)`.
    ///
    /// This can be extended to an n-ary expression.
    ///
    /// This method only checks the expression itself, not its children.
    fn is_null_sensitive(&self, options: &Self::Options) -> bool {
        _ = options;
        true
    }

    /// Returns whether this expression itself is fallible. Conservatively default to *true*.
    ///
    /// An expression is runtime fallible is there is an input set that causes the expression to
    /// panic or return an error, for example checked_add is fallible if there is overflow.
    ///
    /// Note: this is only applicable to expressions that pass type-checking
    /// [`VTable::return_dtype`].
    fn is_fallible(&self, options: &Self::Options) -> bool {
        _ = options;
        true
    }
}

/// Arguments for reduction rules.
pub trait ReduceCtx {
    /// Create a new reduction node from the given scalar function and children.
    fn new_node(
        &self,
        scalar_fn: ScalarFn,
        children: &[ReduceNodeRef],
    ) -> VortexResult<ReduceNodeRef>;
}

pub type ReduceNodeRef = Arc<dyn ReduceNode>;

/// A node used for implementing abstract reduction rules.
pub trait ReduceNode {
    /// Downcast to Any.
    fn as_any(&self) -> &dyn Any;

    /// Return the data type of this node.
    fn node_dtype(&self) -> VortexResult<DType>;

    /// Return this node's scalar function if it is indeed a scalar fn.
    fn scalar_fn(&self) -> Option<&ScalarFn>;

    /// Descend to the child of this handle.
    fn child(&self, idx: usize) -> ReduceNodeRef;

    /// Returns the number of children of this node.
    fn child_count(&self) -> usize;
}

/// The arity (number of arguments) of a function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arity {
    Exact(usize),
    Variadic { min: usize, max: Option<usize> },
}

impl Display for Arity {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Arity::Exact(n) => write!(f, "{}", n),
            Arity::Variadic { min, max } => match max {
                Some(max) if min == max => write!(f, "{}", min),
                Some(max) => write!(f, "{}..{}", min, max),
                None => write!(f, "{}+", min),
            },
        }
    }
}

impl Arity {
    /// Whether the given argument count matches this arity.
    pub fn matches(&self, arg_count: usize) -> bool {
        match self {
            Arity::Exact(m) => *m == arg_count,
            Arity::Variadic { min, max } => {
                if arg_count < *min {
                    return false;
                }
                if let Some(max) = max
                    && arg_count > *max
                {
                    return false;
                }
                true
            }
        }
    }
}

/// Context for simplification.
///
/// Used to lazily compute input data types where simplification requires them.
pub trait SimplifyCtx {
    /// Get the data type of the given expression.
    fn return_dtype(&self, expr: &Expression) -> VortexResult<DType>;
}

/// Arguments for expression execution.
pub struct ExecutionArgs<'a> {
    /// The inputs for the expression, one per child.
    pub inputs: Vec<ArrayRef>,
    /// The row count of the execution scope.
    pub row_count: usize,
    /// The execution context.
    pub ctx: &'a mut ExecutionCtx,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EmptyOptions;
impl Display for EmptyOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

/// Factory functions for static vtables.
pub trait VTableExt: VTable {
    /// Bind this vtable with the given options into a [`ScalarFn`].
    fn bind(&'static self, options: Self::Options) -> ScalarFn {
        ScalarFn::new_static(self, options)
    }

    /// Create a new expression with this vtable and the given options and children.
    fn new_expr(
        &'static self,
        options: Self::Options,
        children: impl IntoIterator<Item = Expression>,
    ) -> Expression {
        Self::try_new_expr(self, options, children).vortex_expect("Failed to create expression")
    }

    /// Try to create a new expression with this vtable and the given options and children.
    fn try_new_expr(
        &'static self,
        options: Self::Options,
        children: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Expression> {
        Expression::try_new(self.bind(options), children)
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
    fn fmt_sql(&self, expression: &Expression, f: &mut Formatter<'_>) -> fmt::Result;

    fn options_serialize(&self, options: &dyn Any) -> VortexResult<Option<Vec<u8>>>;
    fn options_deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Box<dyn Any + Send + Sync>>;
    fn options_clone(&self, options: &dyn Any) -> Box<dyn Any + Send + Sync>;
    fn options_eq(&self, a: &dyn Any, b: &dyn Any) -> bool;
    fn options_hash(&self, options: &dyn Any, hasher: &mut dyn Hasher);
    fn options_display(&self, options: &dyn Any, fmt: &mut Formatter<'_>) -> fmt::Result;
    fn options_debug(&self, options: &dyn Any, fmt: &mut Formatter<'_>) -> fmt::Result;

    fn return_dtype(&self, options: &dyn Any, arg_types: &[DType]) -> VortexResult<DType>;
    fn simplify(
        &self,
        expression: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>>;
    fn simplify_untyped(&self, expression: &Expression) -> VortexResult<Option<Expression>>;
    fn validity(&self, expression: &Expression) -> VortexResult<Option<Expression>>;
    fn execute(&self, options: &dyn Any, args: ExecutionArgs) -> VortexResult<ArrayRef>;
    fn reduce(
        &self,
        options: &dyn Any,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>>;

    fn arity(&self, options: &dyn Any) -> Arity;
    fn child_name(&self, options: &dyn Any, child_idx: usize) -> ChildName;
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
    fn is_null_sensitive(&self, options: &dyn Any) -> bool;
    fn is_fallible(&self, options: &dyn Any) -> bool;
}

#[repr(transparent)]
pub struct VTableAdapter<V>(V);

impl<V: VTable> DynExprVTable for VTableAdapter<V> {
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    #[inline(always)]
    fn id(&self) -> ExprId {
        V::id(&self.0)
    }

    fn fmt_sql(&self, expression: &Expression, f: &mut Formatter<'_>) -> fmt::Result {
        V::fmt_sql(
            &self.0,
            downcast::<V>(expression.options().as_any()),
            expression,
            f,
        )
    }

    fn options_serialize(&self, options: &dyn Any) -> VortexResult<Option<Vec<u8>>> {
        V::serialize(&self.0, downcast::<V>(options))
    }

    fn options_deserialize(
        &self,
        bytes: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Box<dyn Any + Send + Sync>> {
        Ok(Box::new(V::deserialize(&self.0, bytes, session)?))
    }

    fn options_clone(&self, options: &dyn Any) -> Box<dyn Any + Send + Sync> {
        let options = options
            .downcast_ref::<V::Options>()
            .vortex_expect("Failed to downcast expression options to expected type");
        Box::new(options.clone())
    }

    fn options_eq(&self, a: &dyn Any, b: &dyn Any) -> bool {
        downcast::<V>(a) == downcast::<V>(b)
    }

    fn options_hash(&self, options: &dyn Any, mut hasher: &mut dyn Hasher) {
        downcast::<V>(options).hash(&mut hasher);
    }

    fn options_display(&self, options: &dyn Any, fmt: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(downcast::<V>(options), fmt)
    }

    fn options_debug(&self, options: &dyn Any, fmt: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(downcast::<V>(options), fmt)
    }

    fn return_dtype(&self, options: &dyn Any, arg_dtypes: &[DType]) -> VortexResult<DType> {
        V::return_dtype(&self.0, downcast::<V>(options), arg_dtypes)
    }

    fn simplify(
        &self,
        expression: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        V::simplify(
            &self.0,
            downcast::<V>(expression.options().as_any()),
            expression,
            ctx,
        )
    }

    fn simplify_untyped(&self, expression: &Expression) -> VortexResult<Option<Expression>> {
        V::simplify_untyped(
            &self.0,
            downcast::<V>(expression.options().as_any()),
            expression,
        )
    }

    fn validity(&self, expression: &Expression) -> VortexResult<Option<Expression>> {
        V::validity(
            &self.0,
            downcast::<V>(expression.options().as_any()),
            expression,
        )
    }

    fn execute(&self, options: &dyn Any, args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let options = downcast::<V>(options);

        let expected_row_count = args.row_count;
        #[cfg(debug_assertions)]
        let expected_dtype = {
            let args_dtypes: Vec<DType> = args
                .inputs
                .iter()
                .map(|array| array.dtype().clone())
                .collect();
            V::return_dtype(&self.0, options, &args_dtypes)
        }?;

        let result = V::execute(&self.0, options, args)?;

        assert_eq!(
            result.len(),
            expected_row_count,
            "Expression execution {} returned vector of length {}, but expected {}",
            self.0.id(),
            result.len(),
            expected_row_count,
        );

        // In debug mode, validate that the output dtype matches the expected return dtype.
        #[cfg(debug_assertions)]
        {
            vortex_error::vortex_ensure!(
                result.dtype() == &expected_dtype,
                "Expression execution {} returned vector of invalid dtype. Expected {}, got {}",
                self.0.id(),
                expected_dtype,
                result.dtype(),
            );
        }

        Ok(result)
    }

    fn reduce(
        &self,
        options: &dyn Any,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        V::reduce(&self.0, downcast::<V>(options), node, ctx)
    }

    fn arity(&self, options: &dyn Any) -> Arity {
        V::arity(&self.0, downcast::<V>(options))
    }

    fn child_name(&self, options: &dyn Any, child_idx: usize) -> ChildName {
        V::child_name(&self.0, downcast::<V>(options), child_idx)
    }

    fn stat_falsification(
        &self,
        expression: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        V::stat_falsification(
            &self.0,
            downcast::<V>(expression.options().as_any()),
            expression,
            catalog,
        )
    }

    fn stat_expression(
        &self,
        expression: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        V::stat_expression(
            &self.0,
            downcast::<V>(expression.options().as_any()),
            expression,
            stat,
            catalog,
        )
    }

    fn is_null_sensitive(&self, options: &dyn Any) -> bool {
        V::is_null_sensitive(&self.0, downcast::<V>(options))
    }

    fn is_fallible(&self, options: &dyn Any) -> bool {
        V::is_fallible(&self.0, downcast::<V>(options))
    }
}

fn downcast<V: VTable>(options: &dyn Any) -> &V::Options {
    options
        .downcast_ref::<V::Options>()
        .vortex_expect("Invalid options type for expression")
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

    /// Return the vtable as an Any reference.
    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }

    /// Creates a new [`ExprVTable`] from a vtable.
    pub fn new<V: VTable>(vtable: V) -> Self {
        Self(ArcRef::new_arc(Arc::new(VTableAdapter(vtable))))
    }

    /// Creates a new [`ExprVTable`] from a static reference to a vtable.
    pub const fn new_static<V: VTable>(vtable: &'static V) -> Self {
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
        self.0.as_any().is::<V>()
    }

    /// Deserialize an options of this expression vtable from metadata.
    pub fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<ScalarFn> {
        Ok(unsafe {
            ScalarFn::new_unchecked(
                self.clone(),
                self.as_dyn().options_deserialize(metadata, session)?,
            )
        })
    }
}

impl PartialEq for ExprVTable {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}
impl Eq for ExprVTable {}

impl Hash for ExprVTable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state);
    }
}

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
    use rstest::rstest;

    use super::*;
    use crate::LEGACY_SESSION;
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
    use crate::expr::exprs::fill_null::fill_null;
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
    // Fill null expressions
    #[case(fill_null(col("a"), lit(0)))]
    // Type casting expressions
    #[case(cast(
        col("a"),
        DType::Primitive(crate::dtype::PType::I64, crate::dtype::Nullability::NonNullable)
    ))]
    // Between expressions
    #[case(between(
        col("a"),
        lit(10),
        lit(20),
        crate::expr::BetweenOptions{ lower_strict: crate::expr::StrictComparison::NonStrict, upper_strict: crate::expr::StrictComparison::NonStrict }
    ))]
    // List contains expressions
    #[case(list_contains(col("list_col"), lit("item")))]
    // Pack expressions - creating struct from fields
    #[case(pack([("field1", col("a")), ("field2", col("b"))], crate::dtype::Nullability::NonNullable
    ))]
    // Merge expressions - merging struct expressions
    #[case(merge([col("struct1"), col("struct2")]))]
    // Complex nested expressions
    #[case(and(gt(col("a"), lit(0)), lt(col("a"), lit(100))))]
    #[case(or(is_null(col("a")), eq(col("a"), lit(0))))]
    #[case(not(and(eq(col("status"), lit("active")), gt(col("age"), lit(18)))))]
    fn text_expr_serde_round_trip(#[case] expr: Expression) -> VortexResult<()> {
        let serialized_pb = expr.serialize_proto()?;
        let deserialized_expr = Expression::from_proto(&serialized_pb, &LEGACY_SESSION)?;

        assert_eq!(&expr, &deserialized_expr);

        Ok(())
    }
}
