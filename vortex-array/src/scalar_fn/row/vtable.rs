// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Blanket scalar-function implementation and execution visitors for row functions.

use std::marker::PhantomData;

use vortex_error::VortexResult;
#[cfg(any(test, feature = "_test-harness"))]
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::row_fn::RowFn;
use super::row_fn::RowVisitor;
use super::row_fn::private;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::union_child_validities;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::ExecutionArgs;
#[cfg(any(test, feature = "_test-harness"))]
use crate::scalar_fn::NullStrategy;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::SinkResult;
#[cfg(any(test, feature = "_test-harness"))]
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::row::execute::RowExecution;
use crate::scalar_fn::row::execute::execute_row_sink_branch;
use crate::scalar_fn::row::execute::execute_row_sink_prepared;
use crate::scalar_fn::row::execute::validate_row_sink;
use crate::scalar_fn::row::lift::Batch;
use crate::scalar_fn::row::lift::BatchPlan;
use crate::scalar_fn::row::lift::KernelArgs;
use crate::scalar_fn::row::lift::RowPolicy;
use crate::scalar_fn::row::lift::reconcile_return;

/// Compile-time check that a dispatched `(A, S, R)` agrees with `F`'s public metadata. Evaluated by
/// monomorphizing
/// [`visit_prepared_into`](RowVisitor::visit_prepared_into), so even a dispatch arm that never runs
/// is checked.
const fn assert_dispatch_agrees<F: RowFn, A: ElementTuple, S: OutputSink, R: SinkResult>() {
    assert!(
        A::ARITY == F::ARG_NAMES.len(),
        "dispatch visited a tuple whose arity differs from RowFn::ARG_NAMES",
    );
    // Dictionary pushdown treats an infallible function as safe to evaluate over values no code
    // references, so every dispatch must fit the function-wide declaration.
    assert!(
        !A::DECODE_FALLIBLE || F::FALLIBLE,
        "dispatch decoded fallibly without declaring RowFn::FALLIBLE",
    );
    assert!(
        !R::FALLIBLE || F::FALLIBLE,
        "dispatch returned an error without declaring RowFn::FALLIBLE",
    );
    assert!(
        !R::DEFERRED || F::FALLIBLE,
        "dispatch deferred an error without declaring RowFn::FALLIBLE",
    );
    assert!(
        S::ERRORS_ARE_DEFERRED == R::DEFERRED,
        "a deferred-error sink and row closure must be used together",
    );
}

/// The plan-time visit: validate the dtypes and derive execution from the concrete sink and row
/// closure selected by dispatch.
struct PlanRows<'a, F> {
    args: &'a [DType],

    /// The visited function, carried only so the dispatch check can name its contract.
    row_fn: PhantomData<F>,
}

impl<F> private::Sealed for PlanRows<'_, F> {}

impl<F: RowFn> RowVisitor for PlanRows<'_, F> {
    type Out = BatchPlan;

    fn visit_prepared_into<A: ElementTuple, S: OutputSink, P, R: SinkResult>(
        self,
        _prepare: impl FnOnce(A::ConstElems<'_>) -> P,
        _apply: impl Fn(&P, A::Elems<'_>, S::Row<'_>) -> R,
    ) -> VortexResult<BatchPlan> {
        const { assert_dispatch_agrees::<F, A, S, R>() };

        Ok(BatchPlan {
            sink_dtype: validate_row_sink::<A, S>(self.args)?,
            policy: RowPolicy::for_dispatch::<A, R>(),
        })
    }
}

/// The run-time visit: decode every column once and run the row loop.
struct ExecuteRows<'a, 'b, F> {
    args: &'a dyn ExecutionArgs,

    /// The sink dtype computed by the planning visit.
    sink_dtype: &'a DType,

    ctx: &'b mut ExecutionCtx,

    /// The visited function, carried only so the dispatch check can name its contract.
    row_fn: PhantomData<F>,
}

impl<F> private::Sealed for ExecuteRows<'_, '_, F> {}

impl<F: RowFn> RowVisitor for ExecuteRows<'_, '_, F> {
    type Out = RowExecution;

    fn visit_prepared_into<A: ElementTuple, S: OutputSink, P, R: SinkResult>(
        self,
        prepare: impl FnOnce(A::ConstElems<'_>) -> P,
        apply: impl Fn(&P, A::Elems<'_>, S::Row<'_>) -> R,
    ) -> VortexResult<RowExecution> {
        const { assert_dispatch_agrees::<F, A, S, R>() };
        execute_row_sink_prepared::<A, P, S, R>(
            self.args,
            self.sink_dtype,
            self.ctx,
            prepare,
            apply,
        )
    }
}

/// The run-time visit for the branch-and-skip null strategy: compute only the conjoined-valid
/// rows over unfiltered columns.
///
/// `Ok(None)` means the visit cannot take that strategy because the sink cannot skip rows or an
/// argument has no null-tolerant decode, and the lifting falls back to the filter strategy.
struct ExecuteRowsBranch<'a, 'b, F> {
    args: &'a dyn ExecutionArgs,

    /// The sink dtype computed by the planning visit.
    sink_dtype: &'a DType,

    /// The conjoined validity, materialized by the lifting and guaranteed mixed.
    valid: &'a Mask,

    ctx: &'b mut ExecutionCtx,

    /// The visited function, carried only so the dispatch check can name its contract.
    row_fn: PhantomData<F>,
}

impl<F> private::Sealed for ExecuteRowsBranch<'_, '_, F> {}

impl<F: RowFn> RowVisitor for ExecuteRowsBranch<'_, '_, F> {
    type Out = Option<RowExecution>;

    fn visit_prepared_into<A: ElementTuple, S: OutputSink, P, R: SinkResult>(
        self,
        prepare: impl FnOnce(A::ConstElems<'_>) -> P,
        apply: impl Fn(&P, A::Elems<'_>, S::Row<'_>) -> R,
    ) -> VortexResult<Option<RowExecution>> {
        const { assert_dispatch_agrees::<F, A, S, R>() };
        execute_row_sink_branch::<A, P, S, R>(
            self.args,
            self.sink_dtype,
            self.valid,
            self.ctx,
            prepare,
            apply,
        )
    }
}

/// The kernel the lifting runs: the encoding-aware rewrite if it answers, otherwise the row loop
/// over whichever arguments the lifting hands over.
fn execute_rows<F: RowFn>(
    row_fn: &F,
    options: &F::Options,
    args: KernelArgs<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RowExecution> {
    if let Some(reduced) = row_fn.reduce_encoded(options, args.arrays, ctx)? {
        return Ok(RowExecution::Output(reduced));
    }

    row_fn.dispatch(
        options,
        args.dtypes,
        ExecuteRows::<F> {
            args: args.execution,
            sink_dtype: args.sink_dtype,
            ctx,
            row_fn: PhantomData,
        },
    )
}

/// The branch-and-skip kernel: compute only the rows set in `valid`, over the unfiltered `args`.
///
/// `Ok(None)` sends the batch to the filter strategy instead.
fn execute_rows_branch<F: RowFn>(
    row_fn: &F,
    options: &F::Options,
    args: KernelArgs<'_>,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<RowExecution>> {
    // The encoding-aware rewrite runs before the row loop exactly as in [`execute_rows`]. Here it
    // sees the original (unfiltered) encodings, and its full-length result is masked by the caller
    // like any other branch result.
    if let Some(reduced) = row_fn.reduce_encoded(options, args.arrays, ctx)? {
        return Ok(Some(RowExecution::Output(reduced)));
    }

    row_fn.dispatch(
        options,
        args.dtypes,
        ExecuteRowsBranch::<F> {
            args: args.execution,
            sink_dtype: args.sink_dtype,
            valid,
            ctx,
            row_fn: PhantomData,
        },
    )
}

/// The batch facts for `row_fn` over `args`, derived from its dispatched elements and sink.
fn lift_batch<'a, F: RowFn>(
    row_fn: &F,
    options: &F::Options,
    args: &'a dyn ExecutionArgs,
) -> VortexResult<Batch<'a>> {
    Batch::new(RowFn::id(row_fn), args, |arg_dtypes| {
        let plan = row_fn.dispatch(
            options,
            arg_dtypes,
            PlanRows::<F> {
                args: arg_dtypes,
                row_fn: PhantomData,
            },
        )?;
        Ok(plan)
    })
}

/// The nullable execution policy selected by one concrete dispatch.
#[cfg(test)]
pub(super) fn row_policy<F: RowFn>(
    row_fn: &F,
    options: &F::Options,
    args: &[DType],
) -> VortexResult<RowPolicy> {
    row_fn
        .dispatch(
            options,
            args,
            PlanRows::<F> {
                args,
                row_fn: PhantomData,
            },
        )
        .map(|plan| plan.policy)
}

/// Every [`RowFn`] is a [`ScalarFnVTable`], the row loop lifted by `Batch`.
///
/// This impl is why a [`RowFn`] cannot also implement [`ScalarFnVTable`] itself: coherence forbids
/// the second impl. Nothing in tree needs to, since everything a row function can vary lives on
/// [`RowFn`]; mirror another [`ScalarFnVTable`] method onto it when something actually does.
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

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        ChildName::from(F::ARG_NAMES[child_idx])
    }

    /// The visited output element's dtype, widened to nullable iff any input is nullable, which is
    /// what makes the strictness dtype contract hold by construction.
    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        let plan = self.dispatch(
            options,
            args,
            PlanRows::<F> {
                args,
                row_fn: PhantomData,
            },
        )?;

        let nullability =
            plan.sink_dtype.nullability() | Nullability::from(args.iter().any(DType::is_nullable));
        Ok(plan.sink_dtype.with_nullability(nullability))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Nullary functions have no input values that could be null, so there is nothing to lift.
        if args.num_inputs() == 0 {
            let result_dtype = ScalarFnVTable::return_dtype(self, options, &[])?;
            let values = execute_rows(
                self,
                options,
                KernelArgs {
                    execution: args,
                    arrays: &[],
                    dtypes: &[],
                    sink_dtype: &result_dtype,
                },
                ctx,
            )?
            .into_result()?;
            return reconcile_return(RowFn::id(self), &result_dtype, args.row_count(), values);
        }

        lift_batch(self, options, args)?.execute(
            |args, ctx| execute_rows(self, options, args, ctx),
            |args, valid, ctx| execute_rows_branch(self, options, args, valid, ctx),
            ctx,
        )
    }

    /// Output sinks build an all-valid column, so a row kernel cannot turn a wholly non-null row into
    /// a null and the output validity is exactly the conjunction of the inputs'. Letting a sink
    /// produce nulls would invalidate this.
    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    /// A row kernel maps a null input row to a null output row, and computes non-null outputs from
    /// non-null inputs alone, which is exactly strictness. The lifting is what makes it true.
    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        F::FALLIBLE
    }
}

/// Execute `row_fn` over `inputs` with a forced null strategy, bypassing the per-batch selection.
///
/// A test and benchmark seam only, and the only way to name a strategy from outside: it is how the
/// two are compared and how their agreement is asserted. It skips the null-constant and
/// all-constant folds, so do not pass such inputs. Forcing [`NullStrategy::BranchAndSkip`] on a
/// dispatch with no branch execution is an error rather than a silent fallback to filtering.
#[cfg(any(test, feature = "_test-harness"))]
pub fn execute_row_fn_with_strategy<F: RowFn>(
    row_fn: &F,
    options: &F::Options,
    inputs: Vec<ArrayRef>,
    row_count: usize,
    strategy: NullStrategy,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let args = VecExecutionArgs::new(inputs, row_count);

    lift_batch(row_fn, options, &args)?
        .execute_with_strategy(
            |args, ctx| execute_rows(row_fn, options, args, ctx),
            |args, valid, ctx| execute_rows_branch(row_fn, options, args, valid, ctx),
            strategy,
            ctx,
        )?
        .ok_or_else(|| {
            vortex_err!(
                "{} has no branch-and-skip execution for these inputs",
                RowFn::id(row_fn),
            )
        })
}
