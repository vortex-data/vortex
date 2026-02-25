// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use arcref::ArcRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::stats::Stat;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFn;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::SimplifyCtx;

/// An object-safe trait for dynamic dispatch of Vortex expression vtables.
///
/// This trait is automatically implemented via the [`ScalarFnInner`] for any type that
/// implements [`ScalarFnVTable`], and lifts the associated types into dynamic trait objects.
pub trait DynScalarFn: 'static + Send + Sync + private::Sealed {
    fn as_any(&self) -> &dyn Any;

    fn id(&self) -> ScalarFnId;
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
pub struct ScalarFnInner<V>(V);

impl<V: ScalarFnVTable> DynScalarFn for ScalarFnInner<V> {
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    #[inline(always)]
    fn id(&self) -> ScalarFnId {
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

fn downcast<V: ScalarFnVTable>(options: &dyn Any) -> &V::Options {
    options
        .downcast_ref::<V::Options>()
        .vortex_expect("Invalid options type for expression")
}

pub(crate) mod private {
    use crate::scalar_fn::ScalarFnInner;
    use crate::scalar_fn::ScalarFnVTable;

    pub trait Sealed {}
    impl<V: ScalarFnVTable> Sealed for ScalarFnInner<V> {}
}

/// A Vortex expression vtable, used to deserialize or instantiate expressions dynamically.
#[derive(Clone)]
pub struct ScalarFnPlugin(ArcRef<dyn DynScalarFn>);

impl ScalarFnPlugin {
    /// Only the vortex-array crate can actually invoke the vtable methods.
    /// All other users must go via session extensions.
    pub(crate) fn as_dyn(&self) -> &dyn DynScalarFn {
        self.0.as_ref()
    }

    /// Return the vtable as an Any reference.
    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }

    /// Creates a new [`ScalarFnPlugin`] from a vtable.
    pub fn new<V: ScalarFnVTable>(vtable: V) -> Self {
        Self(ArcRef::new_arc(std::sync::Arc::new(ScalarFnInner(vtable))))
    }

    /// Creates a new [`ScalarFnPlugin`] from a static reference to a vtable.
    pub const fn new_static<V: ScalarFnVTable>(vtable: &'static V) -> Self {
        // SAFETY: We can safely cast the vtable to a ScalarFnInner since it has the same layout.
        let adapted: &'static ScalarFnInner<V> =
            unsafe { &*(vtable as *const V as *const ScalarFnInner<V>) };
        Self(ArcRef::new_ref(adapted as &'static dyn DynScalarFn))
    }

    /// Returns the ID of this vtable.
    pub fn id(&self) -> ScalarFnId {
        self.0.id()
    }

    /// Returns whether this vtable is of a given type.
    pub fn is<V: ScalarFnVTable>(&self) -> bool {
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

impl PartialEq for ScalarFnPlugin {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}
impl Eq for ScalarFnPlugin {}

impl Hash for ScalarFnPlugin {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state);
    }
}

impl Display for ScalarFnPlugin {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_dyn().id())
    }
}

impl Debug for ScalarFnPlugin {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_dyn().id())
    }
}
