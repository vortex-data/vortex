// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed and inner representations of scalar functions.
//!
//! - [`ScalarFn<V>`]: The public typed wrapper, parameterized by a concrete [`ScalarFnVTable`].
//! - [`ScalarFn<V>`]: The private inner struct that holds the vtable + options.
//! - [`DynScalarFn`]: The private sealed trait for type-erased dispatch (bound, options in self).

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
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
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::SimplifyCtx;

/// A typed scalar function instance, parameterized by a concrete [`ScalarFnVTable`].
///
/// You can construct one via [`new()`], and erase the type with [`erased()`] to obtain a
/// [`ScalarFnRef`].
///
/// [`new()`]: ScalarFn::new
/// [`erased()`]: ScalarFn::erased
pub struct ScalarFn<V: ScalarFnVTable> {
    vtable: V,
    options: V::Options,
}

impl<V: ScalarFnVTable> ScalarFn<V> {
    /// Create a new typed scalar function instance.
    pub fn new(vtable: V, options: V::Options) -> Self {
        Self { vtable, options }
    }

    /// Returns a reference to the vtable.
    pub fn vtable(&self) -> &V {
        &self.vtable
    }

    /// Returns a reference to the options.
    pub fn options(&self) -> &V::Options {
        &self.options
    }

    /// Erase the concrete type information, returning a type-erased [`ScalarFnRef`].
    pub fn erased(self) -> ScalarFnRef {
        ScalarFnRef(Arc::new(self))
    }
}

/// An object-safe, sealed trait for bound scalar function dispatch.
///
/// Options are stored inside the implementing [`ScalarFn<V>`], not passed externally.
/// This is the sole trait behind [`ScalarFnRef`]'s `Arc<dyn DynScalarFn>`.
pub(super) trait DynScalarFn: 'static + Send + Sync + super::sealed::Sealed {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> ScalarFnId;
    fn options_any(&self) -> &dyn Any;

    // Bound methods — options accessed from self
    fn execute(&self, args: &dyn ExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;
    fn return_dtype(&self, arg_types: &[DType]) -> VortexResult<DType>;
    fn reduce(
        &self,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>>;
    fn arity(&self) -> Arity;
    fn child_name(&self, child_idx: usize) -> ChildName;
    fn is_null_sensitive(&self) -> bool;
    fn is_fallible(&self) -> bool;

    // Expression methods — take &Expression for tree traversal
    fn fmt_sql(&self, expression: &Expression, f: &mut Formatter<'_>) -> fmt::Result;
    fn simplify(
        &self,
        expression: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>>;
    fn simplify_untyped(&self, expression: &Expression) -> VortexResult<Option<Expression>>;
    fn validity(&self, expression: &Expression) -> VortexResult<Option<Expression>>;
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

    // Options operations — self-contained
    fn options_serialize(&self) -> VortexResult<Option<Vec<u8>>>;
    fn options_eq(&self, other_options: &dyn Any) -> bool;
    fn options_hash(&self, hasher: &mut dyn Hasher);
    fn options_display(&self, f: &mut Formatter<'_>) -> fmt::Result;
    fn options_debug(&self, f: &mut Formatter<'_>) -> fmt::Result;
}

impl<V: ScalarFnVTable> DynScalarFn for ScalarFn<V> {
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline(always)]
    fn id(&self) -> ScalarFnId {
        V::id(&self.vtable)
    }

    fn options_any(&self) -> &dyn Any {
        &self.options
    }

    fn execute(&self, args: &dyn ExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let expected_row_count = args.row_count();
        #[cfg(debug_assertions)]
        let expected_dtype = {
            let args_dtypes: Vec<DType> = (0..args.num_inputs())
                .map(|i| args.get(i).map(|a| a.dtype().clone()))
                .collect::<VortexResult<_>>()?;
            V::return_dtype(&self.vtable, &self.options, &args_dtypes)
        }?;

        let result = V::execute(&self.vtable, &self.options, args, ctx)?;

        assert_eq!(
            result.len(),
            expected_row_count,
            "Expression execution {} returned vector of length {}, but expected {}",
            self.vtable.id(),
            result.len(),
            expected_row_count,
        );

        #[cfg(debug_assertions)]
        {
            vortex_error::vortex_ensure!(
                result.dtype() == &expected_dtype,
                "Expression execution {} returned vector of invalid dtype. Expected {}, got {}",
                self.vtable.id(),
                expected_dtype,
                result.dtype(),
            );
        }

        Ok(result)
    }

    fn return_dtype(&self, arg_dtypes: &[DType]) -> VortexResult<DType> {
        V::return_dtype(&self.vtable, &self.options, arg_dtypes)
    }

    fn reduce(
        &self,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        V::reduce(&self.vtable, &self.options, node, ctx)
    }

    fn arity(&self) -> Arity {
        V::arity(&self.vtable, &self.options)
    }

    fn child_name(&self, child_idx: usize) -> ChildName {
        V::child_name(&self.vtable, &self.options, child_idx)
    }

    fn is_null_sensitive(&self) -> bool {
        V::is_null_sensitive(&self.vtable, &self.options)
    }

    fn is_fallible(&self) -> bool {
        V::is_fallible(&self.vtable, &self.options)
    }

    fn fmt_sql(&self, expression: &Expression, f: &mut Formatter<'_>) -> fmt::Result {
        V::fmt_sql(&self.vtable, &self.options, expression, f)
    }

    fn simplify(
        &self,
        expression: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        V::simplify(&self.vtable, &self.options, expression, ctx)
    }

    fn simplify_untyped(&self, expression: &Expression) -> VortexResult<Option<Expression>> {
        V::simplify_untyped(&self.vtable, &self.options, expression)
    }

    fn validity(&self, expression: &Expression) -> VortexResult<Option<Expression>> {
        V::validity(&self.vtable, &self.options, expression)
    }

    fn stat_falsification(
        &self,
        expression: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        V::stat_falsification(&self.vtable, &self.options, expression, catalog)
    }

    fn stat_expression(
        &self,
        expression: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        V::stat_expression(&self.vtable, &self.options, expression, stat, catalog)
    }

    fn options_serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        V::serialize(&self.vtable, &self.options)
    }

    fn options_eq(&self, other_options: &dyn Any) -> bool {
        other_options
            .downcast_ref::<V::Options>()
            .is_some_and(|o| self.options == *o)
    }

    fn options_hash(&self, mut hasher: &mut dyn Hasher) {
        self.options.hash(&mut hasher);
    }

    fn options_display(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.options, f)
    }

    fn options_debug(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.options, f)
    }
}
