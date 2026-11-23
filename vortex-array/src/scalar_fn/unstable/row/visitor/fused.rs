// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plans and executes a batch in a single dispatch.
//!
//! Dispatch and batch setup happen once per call, regardless of the number of rows. Repeating that
//! work is especially costly for short batches whose row computation is a simple predicate.
//!
//! The row closure exists only inside the visit that [`RowFn::dispatch`] selects. Returning a plan
//! would discard that closure and require another dispatch to recover it. [`ExecuteFused`] instead
//! plans and runs within the same visit whenever at most one kernel call can finish the batch.
//!
//! A partially valid batch under [`RowPolicy::DenseWithRetry`] or [`RowPolicy::ValidOnly`] can need
//! several kernel calls. It returns as [`Fused::Partial`] with a plan that later execution visits
//! must reproduce before running a kernel.
//!
//! A Boolean visit can also use _direct_ execution, which skips constructing [`BatchInputs`] and a
//! [`BatchPlan`]. It reuses the probe's constant classification over borrowed inputs, avoiding an
//! owned argument list and repeated constant detection. The framework knows that its collector
//! builds a [`BoolArray`] with the requested length and validity, so it can validate those inputs
//! once and return the finished column instead of validating and masking arbitrary kernel output.
//! Generic owned and sink visits retain that validation because their output builders are supplied
//! by the dispatched types.
//!
//! Direct Boolean execution visits every payload, so [`BatchProbe::into_direct_validity`] checks
//! whether that traversal is permitted. Dense execution through the general batch path also
//! visits every payload, but retains planning and output relabeling. A declared output dtype always
//! uses that general path so label validation stays in one place. Its Boolean collector can still
//! attach validity during construction under the [dense contract], avoiding a second
//! array construction for masking.
//!
//! [`BoolArray`]: crate::arrays::BoolArray
//! [dense contract]: crate::scalar_fn::unstable::row::batch::RowFnExecutionArgs::execute_dense

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
use crate::scalar_fn::unstable::row::execute::execute_row_inputs_infallible_bool;
use crate::scalar_fn::unstable::row::execute::execute_sink;
use crate::validity::Validity;

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
    ///
    /// The optional validity argument follows the [dense contract].
    ///
    /// [dense contract]: crate::scalar_fn::unstable::row::batch::RowFnExecutionArgs::execute_dense
    fn execute(
        self,
        storage_dtype: DType,
        policy: RowPolicy,
        kernel: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &mut Option<Validity>,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
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
        let storage_dtype = validate_owned_visit::<Args, bool>(self.dtypes)?;
        let policy = RowPolicy::for_owned_output::<Args>();

        // Keep declared output dtypes on the general path so label validation and relabeling stay
        // in one place, even when the declared dtype would leave the Boolean storage unchanged.
        if self.output_dtype.is_none() {
            let const_mask = self.probe.const_mask();
            match self
                .probe
                .into_direct_validity(self.inputs.num_inputs(), policy)
            {
                Ok(validity) => {
                    // The visit validated the input types, the probe checked input lengths, and
                    // the collector checks decoded lengths and builds exactly one Boolean per
                    // row. These checks replace general plan and output validation on this path.
                    return execute_row_inputs_infallible_bool::<Args, I, MULTIVERSIONED>(
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

        // Pass batch validity straight to the Boolean constructor when dense execution supplies
        // it. Taking the option tells dense execution to skip masking a result that already has
        // that validity. General planning still handles declared output dtypes and final checks.
        self.execute(storage_dtype, policy, |args, validity, ctx| {
            execute_owned_infallible_bool::<Args, MULTIVERSIONED>(
                &args,
                ctx,
                validity.take().unwrap_or(Validity::NonNullable),
                apply,
            )
        })
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
            |args, _validity, ctx| {
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
