// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`ScalarFnVTable`] adapter shared by every [`RowFn`].
//!
//! The [`visitor`](super::visitor) module validates and executes the concrete row signature
//! selected by dispatch. This module connects those visits to batch execution and exposes the
//! resulting scalar function behavior to the rest of the compute stack.

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::row_fn::RowFn;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::union_child_validities;
use crate::scalar_fn::Arity;
use crate::scalar_fn::BorrowedExecutionArgs;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::row::batch::Batch;
use crate::scalar_fn::row::batch::KernelArgs;
use crate::scalar_fn::row::batch::finalize_kernel_output;
use crate::scalar_fn::row::execute::RowExecution;
use crate::scalar_fn::row::visitor::ExecuteRows;
use crate::scalar_fn::row::visitor::ExecuteValidRows;
use crate::scalar_fn::row::visitor::PlanRows;

/// Implement [`ScalarFnVTable`] for every [`RowFn`].
impl<F: RowFn> ScalarFnVTable for F {
    type Options = F::Options;

    fn id(&self) -> ScalarFnId {
        RowFn::id(self)
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        RowFn::serialize(self, options)
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        RowFn::deserialize(self, metadata, session)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(F::ARG_NAMES.len())
    }

    fn child_name(&self, _options: &Self::Options, child_index: usize) -> ChildName {
        ChildName::from(F::ARG_NAMES[child_index])
    }

    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        let plan = self.dispatch(options, args, PlanRows::<F>::new(args))?;

        Ok(plan.result_dtype(args))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Nullary functions have no input validity to propagate, so they skip batch execution.
        if args.num_inputs() == 0 {
            let result_dtype = ScalarFnVTable::return_dtype(self, options, &[])?;
            let nullary_args = KernelArgs {
                arrays: &[],
                row_count: args.row_count(),
                dtypes: &[],
                output_dtype: &result_dtype,
            };

            let execution = execute_rows(self, options, nullary_args, ctx)?;
            let values = VortexResult::from(execution)?;

            return finalize_kernel_output(
                RowFn::id(self),
                &result_dtype,
                args.row_count(),
                values,
            );
        }

        let batch = prepare_batch(self, options, args)?;
        batch.execute(
            |args, ctx| self.reduce_encoded(options, args.arrays, ctx),
            |args, ctx| execute_rows(self, options, args, ctx),
            |args, valid, ctx| try_execute_rows_unfiltered(self, options, args, valid, ctx),
            ctx,
        )
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        F::FALLIBLE
    }
}

/// Run the encoding-aware rewrite when available, or execute the selected row loop.
fn execute_rows<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: KernelArgs<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RowExecution> {
    let execution = BorrowedExecutionArgs::new(args.arrays, args.row_count);

    function.dispatch(
        options,
        args.dtypes,
        ExecuteRows::<F>::new(&execution, args.output_dtype, ctx),
    )
}

/// Try execution against the original inputs, returning `None` when batch execution must filter.
fn try_execute_rows_unfiltered<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: KernelArgs<'_>,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<RowExecution>> {
    let execution = BorrowedExecutionArgs::new(args.arrays, args.row_count);

    function.dispatch(
        options,
        args.dtypes,
        ExecuteValidRows::<F>::new(&execution, args.output_dtype, valid, ctx),
    )
}

/// Prepare the batch inputs and execution plan for `function`.
fn prepare_batch<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: &dyn ExecutionArgs,
) -> VortexResult<Batch> {
    Batch::new(RowFn::id(function), args, |arg_dtypes| {
        function.dispatch(options, arg_dtypes, PlanRows::<F>::new(arg_dtypes))
    })
}
