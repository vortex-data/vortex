// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plans and executes a batch in a single dispatch.
//!
//! The row loop of a simple function, such as a comparison, already compiles to a vector loop. So
//! on a short batch, the fixed costs of each call decide the speed. [#11220] showed this for zone
//! map pruning: the row executor added the same time at 16 and at 1,024 zones.
//!
//! A second [`RowFn::dispatch`] was one of those costs. The row closure exists only inside the
//! visit that `dispatch` selects, so the executor used to dispatch once to plan and again to run,
//! and then check that the function's own `dispatch` chose the same plan both times.
//! [`ExecuteFused`] plans and runs in one visit whenever at most one kernel call can finish the
//! batch.
//!
//! A partially valid batch under [`RowPolicy::DenseWithRetry`] or [`RowPolicy::ValidOnly`] can need
//! several kernel calls, which the existing strategies already run through their own visitors. So
//! it comes back as [`Fused::Partial`] and still dispatches twice.
//!
//! [#11220]: https://github.com/vortex-data/vortex/pull/11220

use vortex_error::VortexResult;

use super::BatchPlan;
use super::RowPolicy;
use super::RowVisitor;
use super::check::assert_deferred_visit_contract;
use super::check::assert_owned_visit_contract;
use super::check::assert_sink_visit_contract;
use super::check::validate_owned_visit;
use super::check::validate_sink_visit;
use super::row_visitor::private;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::SinkResult;
use crate::scalar_fn::unstable::row::batch::BatchInputs;
use crate::scalar_fn::unstable::row::batch::BatchProbe;
use crate::scalar_fn::unstable::row::batch::BorrowedRowFnArgs;
use crate::scalar_fn::unstable::row::batch::PartiallyValidBatch;
use crate::scalar_fn::unstable::row::batch::RowInputs;
use crate::scalar_fn::unstable::row::execute::execute_owned;
use crate::scalar_fn::unstable::row::execute::execute_owned_bool;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible_bool;
use crate::scalar_fn::unstable::row::execute::execute_sink;

/// The result of a fused planning and execution visit.
pub(crate) enum Fused<'a> {
    /// The finished output column.
    Done(ArrayRef),

    /// A partially valid batch that needs dense retry or valid-row execution, with its plan.
    Partial(PartiallyValidBatch<'a>),
}

/// The visit that plans a batch, and also executes it when at most one kernel call can finish it.
pub(crate) struct ExecuteFused<'a, 'ctx, F: RowFn, I: ?Sized> {
    /// The borrowed input columns.
    inputs: &'a I,

    /// The classification of `inputs` that [`BatchProbe::new`] computed before dispatch.
    probe: BatchProbe,

    /// The input dtypes passed to the dispatch.
    dtypes: &'a [DType],

    /// The output dtype declared by [`RowVisitor::with_output_dtype`], if any.
    output_dtype: Option<DType>,

    /// The execution context used to decode the input columns.
    ctx: &'ctx mut ExecutionCtx,

    /// The function being executed. Batch execution names it in its errors.
    function: &'a F,
}

impl<'a, 'ctx, F: RowFn, I: RowInputs + ?Sized> ExecuteFused<'a, 'ctx, F, I> {
    pub(crate) fn new(
        function: &'a F,
        inputs: &'a I,
        probe: BatchProbe,
        dtypes: &'a [DType],
        ctx: &'ctx mut ExecutionCtx,
    ) -> Self {
        Self {
            inputs,
            probe,
            dtypes,
            output_dtype: None,
            ctx,
            function,
        }
    }

    /// Plan the batch, then finish it with `kernel` when at most one kernel call can do so.
    fn execute(
        self,
        storage_dtype: DType,
        policy: RowPolicy,
        kernel: impl FnOnce(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
    ) -> VortexResult<Fused<'a>> {
        let plan = BatchPlan::new(storage_dtype, self.output_dtype, policy)?;
        let batch = BatchInputs::new(RowFn::id(self.function), self.probe, self.inputs);

        let fused = match batch.try_execute_without_retry(self.dtypes, plan, kernel, self.ctx)? {
            Ok(values) => Fused::Done(values),
            Err(partial) => Fused::Partial(partial),
        };

        Ok(fused)
    }
}

impl<F: RowFn, I: ?Sized> private::Sealed for ExecuteFused<'_, '_, F, I> {}

impl<'a, F: RowFn, I: RowInputs + ?Sized> RowVisitor for ExecuteFused<'a, '_, F, I> {
    type VisitResult = Fused<'a>;

    fn with_output_dtype(mut self, dtype: DType) -> Self {
        self.output_dtype = Some(dtype);
        self
    }

    fn visit_prepared<Args, Out, Prepared>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>) -> Out,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement,
    {
        const { assert_owned_visit_contract::<F, Args, Out>() };
        let storage_dtype = validate_owned_visit::<Args, Out>(self.dtypes)?;

        self.execute(
            storage_dtype,
            RowPolicy::for_owned_output::<Args>(),
            |args, ctx| execute_owned_infallible::<Args, Out, Prepared>(&args, ctx, prepare, apply),
        )
    }

    fn visit_bool<Args, const MULTIVERSIONED: bool>(
        self,
        apply: impl Fn(Args::Elems<'_>) -> bool,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
    {
        const { assert_owned_visit_contract::<F, Args, bool>() };
        let storage_dtype = validate_owned_visit::<Args, bool>(self.dtypes)?;

        self.execute(
            storage_dtype,
            RowPolicy::for_owned_output::<Args>(),
            |args, ctx| execute_owned_infallible_bool::<Args, MULTIVERSIONED>(&args, ctx, apply),
        )
    }

    fn visit_prepared_into<Args, Sink, Prepared, ApplyResult>(
        self,
        params: Sink::Params,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>, Sink::Row<'_>) -> ApplyResult,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: ElementTuple,
        Sink: OutputSink,
        ApplyResult: SinkResult<WriteToken = Sink::WriteToken>,
    {
        const { assert_sink_visit_contract::<F, Args, ApplyResult>() };
        let storage_dtype = validate_sink_visit::<Args, Sink>(self.dtypes, &params)?;

        self.execute(
            storage_dtype,
            RowPolicy::for_sink::<Args, ApplyResult>(),
            |args, ctx| {
                execute_sink::<Args, Prepared, Sink, ApplyResult>(
                    &args, &params, ctx, prepare, apply,
                )
            },
        )
    }

    fn visit_prepared_deferred<Args, Out, Prepared, Fail>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>) -> (Out, Fail),
        finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement,
        Fail: FailureEvidence,
    {
        const { assert_deferred_visit_contract::<F, Args, Out, Fail>() };
        let storage_dtype = validate_owned_visit::<Args, Out>(self.dtypes)?;

        self.execute(
            storage_dtype,
            RowPolicy::for_deferred_output::<Args>(),
            |args, ctx| {
                execute_owned::<Args, Out, Prepared, Fail>(
                    &args,
                    ctx,
                    prepare,
                    apply,
                    finish_failure,
                )
            },
        )
    }

    fn visit_prepared_deferred_bool<Args, Prepared, Fail, const MULTIVERSIONED: bool>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>) -> (bool, Fail),
        finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Fail: FailureEvidence,
    {
        const { assert_deferred_visit_contract::<F, Args, bool, Fail>() };
        let storage_dtype = validate_owned_visit::<Args, bool>(self.dtypes)?;

        self.execute(
            storage_dtype,
            RowPolicy::for_deferred_output::<Args>(),
            |args, ctx| {
                execute_owned_bool::<Args, Prepared, Fail, MULTIVERSIONED>(
                    &args,
                    ctx,
                    prepare,
                    apply,
                    finish_failure,
                )
            },
        )
    }
}
