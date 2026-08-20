// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executes the dense attempt for deferred kernels that can retry valid rows.
//!
//! [`ExecuteDenseWithRetry`] replays the dispatch selected by the planning visitor. It returns a
//! dense output when the reduced failure evidence is accepted, or `None` when batch execution must
//! retry only the valid rows.

use vortex_error::VortexResult;

use super::RowPolicy;
use super::RowVisitor;
use super::check::assert_deferred_visit_contract;
use super::check::assert_owned_visit_contract;
use super::check::assert_sink_visit_contract;
use super::check::validate_owned_visit;
use super::check::validate_sink_visit;
use super::execute::ensure_plan;
use super::row_visitor::private;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::SinkResult;
use crate::scalar_fn::unstable::row::batch::BorrowedRowFnArgs;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible;
use crate::scalar_fn::unstable::row::execute::execute_owned_with_retry;
use crate::scalar_fn::unstable::row::execute::execute_sink;

/// The runtime visit for a planned [`RowPolicy::DenseWithRetry`] dispatch.
pub(crate) struct ExecuteDenseWithRetry<'visit, 'inputs, 'ctx, F: RowFn> {
    args: &'visit BorrowedRowFnArgs<'inputs>,
    options: &'visit F::Options,
    ctx: &'ctx mut ExecutionCtx,
}

impl<'visit, 'inputs, 'ctx, F: RowFn> ExecuteDenseWithRetry<'visit, 'inputs, 'ctx, F> {
    pub(crate) fn new(
        args: &'visit BorrowedRowFnArgs<'inputs>,
        options: &'visit F::Options,
        ctx: &'ctx mut ExecutionCtx,
    ) -> Self {
        Self { args, options, ctx }
    }
}

impl<F: RowFn> private::Sealed for ExecuteDenseWithRetry<'_, '_, '_, F> {}

impl<F: RowFn> RowVisitor<F::Options> for ExecuteDenseWithRetry<'_, '_, '_, F> {
    type VisitResult = Option<ArrayRef>;

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
        ensure_plan(
            self.args.output_dtype(),
            self.args.policy(),
            validate_owned_visit::<Args, Out>(self.args.dtypes())?,
            RowPolicy::for_owned_output::<Args>(),
        )?;

        execute_owned_infallible::<Args, Out, Prepared>(self.args, self.ctx, prepare, apply)
            .map(Some)
    }

    fn visit_prepared_into<Args, Sink, Prepared, ApplyResult>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(
            &Prepared,
            Args::Elems<'_>,
            <Sink as OutputSink<F::Options>>::Row<'_>,
        ) -> ApplyResult,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: ElementTuple,
        Sink: OutputSink<F::Options>,
        ApplyResult: SinkResult<WriteToken = <Sink as OutputSink<F::Options>>::WriteToken>,
    {
        const { assert_sink_visit_contract::<F, Args, ApplyResult>() };
        ensure_plan(
            self.args.output_dtype(),
            self.args.policy(),
            validate_sink_visit::<Args, Sink, F::Options>(self.options, self.args.dtypes())?,
            RowPolicy::for_sink::<Args, ApplyResult>(),
        )?;

        execute_sink::<Args, Prepared, Sink, ApplyResult, F::Options>(
            self.args, self.ctx, prepare, apply,
        )
        .map(Some)
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
        ensure_plan(
            self.args.output_dtype(),
            self.args.policy(),
            validate_owned_visit::<Args, Out>(self.args.dtypes())?,
            RowPolicy::for_deferred_output::<Args>(),
        )?;

        execute_owned_with_retry::<Args, Out, Prepared, Fail>(
            self.args,
            self.ctx,
            prepare,
            apply,
            finish_failure,
        )
    }
}
