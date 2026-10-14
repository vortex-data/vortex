// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plans and executes a batch in a single dispatch.
//!
//! [`ExecuteFused`] borrows the inputs and owns their [`BatchProbe`]. Each visit plans its concrete
//! signature, then runs every batch that its own kernel can finish: an all-null, all-constant, or
//! null-free batch, or a partially valid batch under [`RowPolicy::Dense`]. Such a batch therefore
//! needs no second dispatch and no check that a second dispatch reproduced the plan. A visit hands
//! any other batch back as [`Fused::Partial`], and batch execution dispatches again for dense
//! retry or valid-row execution.
//!
//! A Boolean visit decodes a direct batch straight from the borrowed inputs. Every other visit
//! collects the inputs into [`BatchInputs`].

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
use crate::scalar_fn::unstable::row::batch::RowInputs;
use crate::scalar_fn::unstable::row::execute::execute_owned;
use crate::scalar_fn::unstable::row::execute::execute_owned_bool;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible_bool;
use crate::scalar_fn::unstable::row::execute::execute_rows_infallible_bool;
use crate::scalar_fn::unstable::row::execute::execute_sink;
use crate::validity::Validity;

/// The result of a fused planning and execution visit.
pub(crate) enum Fused {
    /// The finished output column.
    Done(ArrayRef),

    /// A partially valid batch that needs dense retry or valid-row execution, with its plan.
    Partial(BatchInputs, BatchPlan),
}

/// The visit that plans a batch and executes it when no retry can be necessary.
pub(crate) struct ExecuteFused<'a, 'ctx, F: RowFn, I: ?Sized> {
    /// The borrowed input columns.
    inputs: &'a I,

    /// What batch execution derived from the inputs.
    probe: BatchProbe,

    /// The input dtypes passed to the dispatch.
    dtypes: &'a [DType],

    /// The output dtype declared by [`RowVisitor::with_output_dtype`], if any.
    output_dtype: Option<DType>,

    /// The execution context used to decode the input columns.
    ctx: &'ctx mut ExecutionCtx,

    /// The function being executed, named in the errors of a collected batch.
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

    /// Collect the inputs and attach `plan`, then finish the batch with `kernel` when no retry can
    /// be necessary.
    fn execute(
        self,
        storage_dtype: DType,
        policy: RowPolicy,
        kernel: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &mut Option<Validity>,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
    ) -> VortexResult<Fused> {
        let plan = BatchPlan::new(storage_dtype, self.output_dtype, policy)?;
        let batch = BatchInputs::collect(RowFn::id(self.function), self.probe, self.inputs);

        let plan = {
            let planned = batch.with_plan(self.dtypes, plan);
            match planned.try_execute_without_retry(kernel, self.ctx)? {
                Some(values) => return Ok(Fused::Done(values)),
                None => planned.into_plan(),
            }
        };

        Ok(Fused::Partial(batch, plan))
    }
}

impl<F: RowFn, I: ?Sized> private::Sealed for ExecuteFused<'_, '_, F, I> {}

impl<F: RowFn, I: RowInputs + ?Sized> RowVisitor for ExecuteFused<'_, '_, F, I> {
    type VisitResult = Fused;

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
            |args, _validity, ctx| {
                execute_owned_infallible::<Args, Out, Prepared>(&args, ctx, prepare, apply)
            },
        )
    }

    fn visit_bool<Args, const MULTIVERSIONED: bool>(
        mut self,
        apply: impl Fn(Args::Elems<'_>) -> bool,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
    {
        const { assert_owned_visit_contract::<F, Args, bool>() };

        // A direct batch packs straight into the finished column. Its storage dtype is the
        // non-nullable Boolean dtype, so only the argument dtypes need validation.
        if self.output_dtype.is_none() {
            Args::validate(self.dtypes)?;

            let dense = RowPolicy::for_owned_output::<Args>() == RowPolicy::Dense;
            let const_mask = self.probe.const_mask();
            match self
                .probe
                .into_direct_validity(self.inputs.num_inputs(), dense)
            {
                Ok(validity) => {
                    // The collector builds a canonical Boolean array of exactly the batch length
                    // with this validity attached, so the output needs no plan, label, or
                    // validation.
                    return execute_rows_infallible_bool::<Args, I, MULTIVERSIONED>(
                        self.inputs,
                        const_mask,
                        self.ctx,
                        validity,
                        apply,
                    )
                    .map(Fused::Done);
                }
                Err(probe) => self.probe = probe,
            }
        }

        let storage_dtype = validate_owned_visit::<Args, bool>(self.dtypes)?;

        self.execute(
            storage_dtype,
            RowPolicy::for_owned_output::<Args>(),
            |args, validity, ctx| {
                execute_owned_infallible_bool::<Args, MULTIVERSIONED>(
                    &args,
                    ctx,
                    validity.take().unwrap_or(Validity::NonNullable),
                    apply,
                )
            },
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
            |args, _validity, ctx| {
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
            |args, _validity, ctx| {
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
            |args, validity, ctx| {
                execute_owned_bool::<Args, Prepared, Fail, MULTIVERSIONED>(
                    &args,
                    ctx,
                    validity.take().unwrap_or(Validity::NonNullable),
                    prepare,
                    apply,
                    finish_failure,
                )
            },
        )
    }
}
