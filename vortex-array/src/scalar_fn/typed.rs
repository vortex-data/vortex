// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed and inner representations of scalar functions.
//!
//! - [`ScalarFn<V>`]: The public typed wrapper, parameterized by a concrete [`ScalarFnVTable`].
//! - [`TypedScalarFnInstance<V>`]: The inner struct that holds the vtable + options.
//! - [`DynScalarFn`]: The private sealed trait for type-erased dispatch (bound, options in self).
//!
//! [`ScalarFn<V>`]: crate::arrays::scalar_fn::ScalarFn

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ScalarFnArray;
use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::expr::display::ExprDisplay;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ArrayReduceNode;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ExpressionReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnExecutionResult;
use crate::scalar_fn::ScalarFnExecutionStep;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;

/// A typed scalar function instance, parameterized by a concrete [`ScalarFnVTable`].
///
/// You can construct one via [`new()`], and erase the type with [`erased()`] to obtain a
/// [`ScalarFnRef`].
///
/// [`new()`]: TypedScalarFnInstance::new
/// [`erased()`]: TypedScalarFnInstance::erased
pub struct TypedScalarFnInstance<V: ScalarFnVTable> {
    vtable: V,
    options: V::Options,
}

impl<V: ScalarFnVTable> TypedScalarFnInstance<V> {
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

    fn check_result(&self, args: &dyn ExecutionArgs, result: &ArrayRef) -> VortexResult<()> {
        assert_eq!(
            result.len(),
            args.row_count(),
            "Expression execution {} returned vector of length {}, but expected {}",
            self.vtable.id(),
            result.len(),
            args.row_count(),
        );

        if cfg!(debug_assertions) {
            let args_dtypes: Vec<DType> = (0..args.num_inputs())
                .map(|i| args.get(i).map(|a| a.dtype().clone()))
                .collect::<VortexResult<_>>()?;
            let expected_dtype = V::return_dtype(&self.vtable, &self.options, &args_dtypes)?;
            vortex_error::vortex_ensure_eq!(
                result.dtype(),
                &expected_dtype,
                "Expression execution {} returned vector of invalid dtype",
                self.vtable.id(),
            );
        }

        Ok(())
    }
}

/// An object-safe, sealed trait for bound scalar function dispatch.
///
/// Options are stored inside the implementing [`ScalarFn<V>`](crate::arrays::scalar_fn::ScalarFn), not passed externally.
/// This is the sole trait behind [`ScalarFnRef`]'s `Arc<dyn DynScalarFn>`.
pub(super) trait DynScalarFn: 'static + Send + Sync + super::sealed::Sealed {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> ScalarFnId;
    fn options_any(&self) -> &dyn Any;

    // Bound methods — options accessed from self
    fn execute(&self, args: &dyn ExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;
    fn execute_step(
        &self,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnExecutionResult>;
    fn return_dtype(&self, arg_types: &[DType]) -> VortexResult<DType>;
    fn bind_options(&self, arg_types: &[DType]) -> VortexResult<Option<ScalarFnRef>>;
    fn reduce_expression<'a>(
        &self,
        node: &ExpressionReduceNode<'a>,
    ) -> VortexResult<Option<ExpressionReduceNode<'a>>>;
    fn reduce_array<'a>(
        &self,
        node: &ArrayReduceNode<'a>,
    ) -> VortexResult<Option<ArrayReduceNode<'a>>>;
    fn arity(&self) -> Arity;
    fn child_name(&self, child_idx: usize) -> ChildName;
    fn is_strict(&self) -> bool;
    fn is_infallible(&self) -> bool;

    // Expression methods — take expressions for tree traversal
    fn fmt_sql(&self, expression: &dyn ExprDisplay, f: &mut Formatter<'_>) -> fmt::Result;
    fn simplify(&self, expression: &BoundExpression) -> VortexResult<Option<BoundExpression>>;
    fn validity_expression<'a>(
        &self,
        node: &ExpressionReduceNode<'a>,
    ) -> VortexResult<ReduceNodeValidity<ExpressionReduceNode<'a>>>;
    fn validity_array<'a>(
        &self,
        node: &ArrayReduceNode<'a>,
    ) -> VortexResult<ReduceNodeValidity<ArrayReduceNode<'a>>>;

    // Options operations — self-contained
    fn options_serialize(&self) -> VortexResult<Option<Vec<u8>>>;
    fn options_eq(&self, other_options: &dyn Any) -> bool;
    fn options_hash(&self, hasher: &mut dyn Hasher);
    fn options_display(&self, f: &mut Formatter<'_>) -> fmt::Result;
    fn options_debug(&self, f: &mut Formatter<'_>) -> fmt::Result;
}

impl<V: ScalarFnVTable> DynScalarFn for TypedScalarFnInstance<V> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn id(&self) -> ScalarFnId {
        V::id(&self.vtable)
    }

    fn options_any(&self) -> &dyn Any {
        &self.options
    }

    fn execute(&self, args: &dyn ExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        match self.execute_step(args, ctx)?.into_step() {
            ScalarFnExecutionStep::Done(result) => Ok(result),
            // Take one executor step on the requested input and return this function lazily
            // over it, leaving the rest of the work to whoever executes the result.
            ScalarFnExecutionStep::ExecuteInput(idx, _) => {
                let mut inputs = (0..args.num_inputs())
                    .map(|i| args.get(i))
                    .collect::<VortexResult<Vec<_>>>()?;
                inputs[idx] = inputs[idx].clone().execute::<ArrayRef>(ctx)?;
                let scalar_fn =
                    TypedScalarFnInstance::new(self.vtable.clone(), self.options.clone()).erased();
                let array = ScalarFnArray::try_new_with_len(scalar_fn, inputs, args.row_count())?;
                Ok(array.into_array())
            }
        }
    }

    fn execute_step(
        &self,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnExecutionResult> {
        let result = V::execute_step(&self.vtable, &self.options, args, ctx)?;
        match result.step() {
            ScalarFnExecutionStep::Done(array) => self.check_result(args, array)?,
            ScalarFnExecutionStep::ExecuteInput(idx, done) => {
                vortex_ensure!(
                    *idx < args.num_inputs(),
                    "Scalar function {} requested input {idx}, but has {} inputs",
                    self.vtable.id(),
                    args.num_inputs(),
                );
                // Requesting an input that already matches would loop without progress.
                vortex_ensure!(
                    !done(&args.get(*idx)?),
                    "Scalar function {} requested execution of input {idx}, which is already done",
                    self.vtable.id(),
                );
            }
        }
        Ok(result)
    }

    fn return_dtype(&self, arg_dtypes: &[DType]) -> VortexResult<DType> {
        V::return_dtype(&self.vtable, &self.options, arg_dtypes)
    }

    fn bind_options(&self, arg_dtypes: &[DType]) -> VortexResult<Option<ScalarFnRef>> {
        Ok(V::bind_options(&self.vtable, &self.options, arg_dtypes)?
            .map(|options| TypedScalarFnInstance::new(self.vtable.clone(), options).erased()))
    }

    fn reduce_expression<'a>(
        &self,
        node: &ExpressionReduceNode<'a>,
    ) -> VortexResult<Option<ExpressionReduceNode<'a>>> {
        V::reduce(&self.vtable, &self.options, node)
    }

    fn reduce_array<'a>(
        &self,
        node: &ArrayReduceNode<'a>,
    ) -> VortexResult<Option<ArrayReduceNode<'a>>> {
        V::reduce(&self.vtable, &self.options, node)
    }

    fn arity(&self) -> Arity {
        V::arity(&self.vtable, &self.options)
    }

    fn child_name(&self, child_idx: usize) -> ChildName {
        V::child_name(&self.vtable, &self.options, child_idx)
    }

    fn is_strict(&self) -> bool {
        V::is_strict(&self.vtable, &self.options)
    }

    fn is_infallible(&self) -> bool {
        V::is_infallible(&self.vtable, &self.options)
    }

    fn fmt_sql(&self, expression: &dyn ExprDisplay, f: &mut Formatter<'_>) -> fmt::Result {
        V::fmt_sql(&self.vtable, &self.options, expression, f)
    }

    fn simplify(&self, expression: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        V::simplify(&self.vtable, &self.options, expression)
    }

    fn validity_expression<'a>(
        &self,
        node: &ExpressionReduceNode<'a>,
    ) -> VortexResult<ReduceNodeValidity<ExpressionReduceNode<'a>>> {
        V::validity(&self.vtable, &self.options, node)
    }

    fn validity_array<'a>(
        &self,
        node: &ArrayReduceNode<'a>,
    ) -> VortexResult<ReduceNodeValidity<ArrayReduceNode<'a>>> {
        V::validity(&self.vtable, &self.options, node)
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
