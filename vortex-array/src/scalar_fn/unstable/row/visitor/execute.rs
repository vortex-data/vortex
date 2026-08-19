// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Visitors that execute dense and skip-invalid row loops.
//!
//! Each visit revalidates its concrete signature and checks that its output dtype and execution
//! policy match the plan before entering a row loop. [`ExecuteValidRows`] can decline when the
//! signature cannot execute over the original inputs.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;

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
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::SinkResult;
use crate::scalar_fn::unstable::row::execute::execute_owned;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible;
use crate::scalar_fn::unstable::row::execute::execute_sink;
use crate::scalar_fn::unstable::row::execute::execute_sink_valid_rows;

/// The runtime visit that decodes every column once and runs the selected row loop.
pub(crate) struct ExecuteRows<'args, 'ctx, F: RowFn> {
    /// The inputs for this kernel invocation.
    args: &'args dyn ExecutionArgs,

    /// The input dtypes used by the planning visit.
    dtypes: &'args [DType],

    /// The function options used to derive a sink's runtime dtype.
    options: &'args F::Options,

    /// The output dtype computed by the planning visit.
    output_dtype: &'args DType,

    /// The nullable execution policy selected by the planning visit.
    policy: RowPolicy,

    /// The execution context used to decode the input columns.
    ctx: &'ctx mut ExecutionCtx,
}

impl<'args, 'ctx, F: RowFn> ExecuteRows<'args, 'ctx, F> {
    pub(crate) fn new(
        args: &'args dyn ExecutionArgs,
        dtypes: &'args [DType],
        options: &'args F::Options,
        output_dtype: &'args DType,
        policy: RowPolicy,
        ctx: &'ctx mut ExecutionCtx,
    ) -> Self {
        Self {
            args,
            dtypes,
            options,
            output_dtype,
            policy,
            ctx,
        }
    }
}

impl<F: RowFn> private::Sealed for ExecuteRows<'_, '_, F> {}

impl<F: RowFn> RowVisitor<F::Options> for ExecuteRows<'_, '_, F> {
    type VisitResult = ArrayRef;

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
            self.output_dtype,
            self.policy,
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            RowPolicy::for_owned_output::<Args>(),
        )?;

        execute_owned_infallible::<Args, Out, Prepared>(self.args, self.ctx, prepare, apply)
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
            self.output_dtype,
            self.policy,
            validate_sink_visit::<Args, Sink, F::Options>(self.options, self.dtypes)?,
            RowPolicy::for_sink::<Args, ApplyResult>(),
        )?;

        execute_sink::<Args, Prepared, Sink, ApplyResult, F::Options>(
            self.args, self.ctx, prepare, apply,
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
        ensure_plan(
            self.output_dtype,
            self.policy,
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            RowPolicy::for_deferred_output::<Args>(),
        )?;

        execute_owned::<Args, Out, Prepared, Fail>(
            self.args,
            self.ctx,
            prepare,
            apply,
            finish_failure,
        )
    }
}

/// The runtime visit that executes valid rows over the original input columns.
///
/// Only output sinks have a contract for skipped output positions. Owned visits therefore decline,
/// and batch execution decides how to handle that unsupported signature.
pub(crate) struct ExecuteValidRows<'args, 'ctx, F: RowFn> {
    /// The original inputs for this kernel invocation.
    args: &'args dyn ExecutionArgs,

    /// The input dtypes used by the planning visit.
    dtypes: &'args [DType],

    /// The function options used to derive a sink's runtime dtype.
    options: &'args F::Options,

    /// The output dtype computed by the planning visit.
    output_dtype: &'args DType,

    /// The nullable execution policy selected by the planning visit.
    policy: RowPolicy,

    /// The conjoined validity, containing both valid and invalid rows.
    valid: &'args Mask,

    /// The execution context used to decode the input columns.
    ctx: &'ctx mut ExecutionCtx,
}

impl<'args, 'ctx, F: RowFn> ExecuteValidRows<'args, 'ctx, F> {
    pub(crate) fn new(
        args: &'args dyn ExecutionArgs,
        dtypes: &'args [DType],
        options: &'args F::Options,
        output_dtype: &'args DType,
        policy: RowPolicy,
        valid: &'args Mask,
        ctx: &'ctx mut ExecutionCtx,
    ) -> Self {
        Self {
            args,
            dtypes,
            options,
            output_dtype,
            policy,
            valid,
            ctx,
        }
    }
}

impl<F: RowFn> private::Sealed for ExecuteValidRows<'_, '_, F> {}

impl<F: RowFn> RowVisitor<F::Options> for ExecuteValidRows<'_, '_, F> {
    type VisitResult = Option<ArrayRef>;

    fn visit_prepared<Args, Out, Prepared>(
        self,
        _prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        _apply: impl Fn(&Prepared, Args::Elems<'_>) -> Out,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement,
    {
        const { assert_owned_visit_contract::<F, Args, Out>() };
        ensure_plan(
            self.output_dtype,
            self.policy,
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            RowPolicy::for_owned_output::<Args>(),
        )?;

        Ok(None)
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
            self.output_dtype,
            self.policy,
            validate_sink_visit::<Args, Sink, F::Options>(self.options, self.dtypes)?,
            RowPolicy::for_sink::<Args, ApplyResult>(),
        )?;

        execute_sink_valid_rows::<Args, Prepared, Sink, ApplyResult, F::Options>(
            self.args, self.valid, self.ctx, prepare, apply,
        )
    }

    fn visit_prepared_deferred<Args, Out, Prepared, Fail>(
        self,
        _prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        _apply: impl Fn(&Prepared, Args::Elems<'_>) -> (Out, Fail),
        _finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement,
        Fail: FailureEvidence,
    {
        const { assert_deferred_visit_contract::<F, Args, Out, Fail>() };
        ensure_plan(
            self.output_dtype,
            self.policy,
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            RowPolicy::for_deferred_output::<Args>(),
        )?;

        Ok(None)
    }
}

fn ensure_plan(
    planned_output: &DType,
    planned_policy: RowPolicy,
    actual_output: DType,
    actual_policy: RowPolicy,
) -> VortexResult<()> {
    vortex_ensure_eq!(
        actual_policy,
        planned_policy,
        "row dispatch must select the planned nullable execution policy: planned {planned_policy:?}, got {actual_policy:?}",
    );
    vortex_ensure_eq!(
        actual_output,
        *planned_output,
        "row dispatch must select the planned output dtype: planned {planned_output}, got {actual_output}",
    );

    Ok(())
}
