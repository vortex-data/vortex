// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Visitors that execute dense and skip-invalid row loops.
//!
//! Each method verifies that execution selected the same visit shape as planning before handing
//! its typed closures to the matching loop. Valid-row execution can decline without running a loop;
//! batch execution then filters the inputs and retries the dense loop.

use std::marker::PhantomData;
use std::ops::BitOrAssign;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::RowVisitor;
use super::check::assert_deferred_visit_contract;
use super::check::assert_owned_visit_contract;
use super::check::assert_sink_visit_contract;
use super::private;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::IndexedElementTuple;
use crate::scalar_fn::OutputElement;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::RowFn;
use crate::scalar_fn::SinkResult;
use crate::scalar_fn::row::execute::RowExecution;
use crate::scalar_fn::row::execute::execute_owned;
use crate::scalar_fn::row::execute::execute_owned_infallible;
use crate::scalar_fn::row::execute::execute_sink;
use crate::scalar_fn::row::execute::execute_sink_valid_rows;

/// The run-time visit that decodes every column once and runs the selected row loop.
pub struct ExecuteRows<'args, 'ctx, F> {
    /// The inputs for this kernel invocation.
    args: &'args dyn ExecutionArgs,

    /// The output dtype computed by the planning visit.
    output_dtype: &'args DType,

    /// The execution context used to decode the input columns.
    ctx: &'ctx mut ExecutionCtx,

    /// The visited function, carried only so the dispatch check can name its contract.
    function: PhantomData<F>,
}

impl<'args, 'ctx, F> ExecuteRows<'args, 'ctx, F> {
    pub fn new(
        args: &'args dyn ExecutionArgs,
        output_dtype: &'args DType,
        ctx: &'ctx mut ExecutionCtx,
    ) -> Self {
        Self {
            args,
            output_dtype,
            ctx,
            function: PhantomData,
        }
    }
}

impl<F> private::Sealed for ExecuteRows<'_, '_, F> {}

impl<F: RowFn> RowVisitor for ExecuteRows<'_, '_, F> {
    type VisitResult = RowExecution;

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

        execute_owned_infallible::<Args, Out, Prepared>(self.args, self.ctx, prepare, apply)
    }

    fn visit_prepared_into<Args, Sink, Prepared, ApplyResult>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>, Sink::Row<'_>) -> ApplyResult,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: ElementTuple,
        Sink: OutputSink,
        ApplyResult: SinkResult<WriteToken = Sink::WriteToken>,
    {
        const { assert_sink_visit_contract::<F, Args, ApplyResult>() };

        execute_sink::<Args, Prepared, Sink, ApplyResult>(
            self.args,
            self.output_dtype,
            self.ctx,
            prepare,
            apply,
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
        Fail: Copy + Default + BitOrAssign,
    {
        const { assert_deferred_visit_contract::<F, Args, Out, Fail>() };

        execute_owned::<Args, Out, Prepared, Fail>(
            self.args,
            self.ctx,
            prepare,
            apply,
            finish_failure,
        )
    }
}

/// The run-time visit that tries skip-invalid execution over the original input columns.
///
/// Only output sinks have a contract for skipped output positions. Owned visits therefore decline
/// so batch execution can use its filter-and-scatter fallback.
pub struct ExecuteValidRows<'args, 'ctx, F> {
    /// The original inputs for this kernel invocation.
    args: &'args dyn ExecutionArgs,

    /// The output dtype computed by the planning visit.
    output_dtype: &'args DType,

    /// The conjoined validity, materialized by batch execution and guaranteed mixed.
    valid: &'args Mask,

    /// The execution context used to decode the input columns.
    ctx: &'ctx mut ExecutionCtx,

    /// The visited function, carried only so the dispatch check can name its contract.
    function: PhantomData<F>,
}

impl<'args, 'ctx, F> ExecuteValidRows<'args, 'ctx, F> {
    pub fn new(
        args: &'args dyn ExecutionArgs,
        output_dtype: &'args DType,
        valid: &'args Mask,
        ctx: &'ctx mut ExecutionCtx,
    ) -> Self {
        Self {
            args,
            output_dtype,
            valid,
            ctx,
            function: PhantomData,
        }
    }
}

impl<F> private::Sealed for ExecuteValidRows<'_, '_, F> {}

impl<F: RowFn> RowVisitor for ExecuteValidRows<'_, '_, F> {
    type VisitResult = Option<RowExecution>;

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

        // Owned execution has no sink that can initialize skipped output positions. Decline so
        // batch execution filters the inputs and retries with the dense visitor.
        Ok(None)
    }

    fn visit_prepared_into<Args, Sink, Prepared, ApplyResult>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>, Sink::Row<'_>) -> ApplyResult,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: ElementTuple,
        Sink: OutputSink,
        ApplyResult: SinkResult<WriteToken = Sink::WriteToken>,
    {
        const { assert_sink_visit_contract::<F, Args, ApplyResult>() };

        execute_sink_valid_rows::<Args, Prepared, Sink, ApplyResult>(
            self.args,
            self.output_dtype,
            self.valid,
            self.ctx,
            prepare,
            apply,
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
        Fail: Copy + Default + BitOrAssign,
    {
        const { assert_deferred_visit_contract::<F, Args, Out, Fail>() };

        // Deferred owned execution has the same skipped-output limitation as `visit_prepared`.
        Ok(None)
    }
}
