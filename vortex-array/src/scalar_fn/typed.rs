// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

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
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::SimplifyCtx;

/// An object-safe trait for dynamic dispatch of Vortex scalar function vtables.
///
/// This trait is automatically implemented via the [`ScalarFnInner`] for any type that
/// implements [`ScalarFnVTable`], and lifts the associated types into dynamic trait objects.
pub(crate) trait DynScalarFn: 'static + Send + Sync + super::sealed::Sealed {
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
pub(crate) struct ScalarFnInner<V>(pub(super) V);

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

pub(crate) fn downcast<V: ScalarFnVTable>(options: &dyn Any) -> &V::Options {
    options
        .downcast_ref::<V::Options>()
        .vortex_expect("Invalid options type for expression")
}
