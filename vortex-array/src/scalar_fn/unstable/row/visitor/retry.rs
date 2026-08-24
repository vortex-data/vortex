// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executes the dense attempt for deferred kernels that can retry valid rows.
//!
//! [`ExecuteDenseWithRetry`] replays the dispatch selected by the planning visitor. It returns a
//! dense attempt containing either unvalidated values or an error that batch execution resolves
//! against input validity.

use std::marker::PhantomData;

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
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::SinkResult;
use crate::scalar_fn::unstable::row::batch::BorrowedRowFnArgs;
use crate::scalar_fn::unstable::row::execute::DenseAttempt;
use crate::scalar_fn::unstable::row::execute::execute_owned_dense_attempt;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible;
use crate::scalar_fn::unstable::row::execute::execute_sink;

/// The runtime visit for a planned [`RowPolicy::DenseWithRetry`] dispatch.
pub(in crate::scalar_fn::unstable::row) struct ExecuteDenseWithRetry<
    'visit,
    'inputs,
    'ctx,
    F: RowFn,
> {
    /// The inputs and planning metadata for this dense attempt.
    args: &'visit BorrowedRowFnArgs<'inputs>,

    /// The output dtype declared by [`RowVisitor::with_output_dtype`], if any.
    output_dtype: Option<DType>,

    /// The execution context used to decode the input columns.
    ctx: &'ctx mut ExecutionCtx,

    /// Ties this visit to the function used by its compile-time contract checks.
    function: PhantomData<F>,
}

impl<'visit, 'inputs, 'ctx, F: RowFn> ExecuteDenseWithRetry<'visit, 'inputs, 'ctx, F> {
    pub(in crate::scalar_fn::unstable::row) fn new(
        args: &'visit BorrowedRowFnArgs<'inputs>,
        ctx: &'ctx mut ExecutionCtx,
    ) -> Self {
        Self {
            args,
            output_dtype: None,
            ctx,
            function: PhantomData,
        }
    }
}

impl<F: RowFn> private::Sealed for ExecuteDenseWithRetry<'_, '_, '_, F> {}

impl<F: RowFn> RowVisitor for ExecuteDenseWithRetry<'_, '_, '_, F> {
    type VisitResult = DenseAttempt;

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
        let visited = BatchPlan::new(
            validate_owned_visit::<Args, Out>(self.args.dtypes())?,
            self.output_dtype,
            RowPolicy::for_owned_output::<Args>(),
        )?;
        self.args.plan().ensure_reproduced_by(&visited)?;

        execute_owned_infallible::<Args, Out, Prepared>(self.args, self.ctx, prepare, apply)
            .map(DenseAttempt::Values)
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
        let visited = BatchPlan::new(
            validate_sink_visit::<Args, Sink>(self.args.dtypes(), &params)?,
            self.output_dtype,
            RowPolicy::for_sink::<Args, ApplyResult>(),
        )?;
        self.args.plan().ensure_reproduced_by(&visited)?;

        execute_sink::<Args, Prepared, Sink, ApplyResult>(
            self.args, &params, self.ctx, prepare, apply,
        )
        .map(DenseAttempt::Values)
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
        let visited = BatchPlan::new(
            validate_owned_visit::<Args, Out>(self.args.dtypes())?,
            self.output_dtype,
            RowPolicy::for_deferred_output::<Args>(),
        )?;
        self.args.plan().ensure_reproduced_by(&visited)?;

        execute_owned_dense_attempt::<Args, Out, Prepared, Fail>(
            self.args,
            self.ctx,
            prepare,
            apply,
            finish_failure,
        )
    }
}
