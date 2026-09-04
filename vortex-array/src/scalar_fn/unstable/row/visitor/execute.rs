// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Visitors that execute dense and skip-invalid row loops.
//!
//! Each visit revalidates its concrete signature and checks that its plan matches the one planning
//! selected before entering a row loop. [`ExecuteValidRows`] can decline when the signature cannot
//! execute over the original inputs; [`ExecuteFilteredRows`] then executes over inputs filtered to
//! the valid rows while writing into the original row domain.

use std::marker::PhantomData;

use vortex_error::VortexResult;
use vortex_mask::MaskValuesRef;

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
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::SinkResult;
use crate::scalar_fn::unstable::row::execute::execute_owned;
use crate::scalar_fn::unstable::row::execute::execute_owned_bool;
use crate::scalar_fn::unstable::row::execute::execute_owned_filtered;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible_bool;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible_filtered;
use crate::scalar_fn::unstable::row::execute::execute_owned_infallible_valid_rows;
use crate::scalar_fn::unstable::row::execute::execute_owned_valid_rows;
use crate::scalar_fn::unstable::row::execute::execute_sink;
use crate::scalar_fn::unstable::row::execute::execute_sink_filtered;
use crate::scalar_fn::unstable::row::execute::execute_sink_valid_rows;

/// The runtime visit that decodes every column once and runs the selected row loop.
pub(crate) struct ExecuteRows<'args, 'ctx, F: RowFn> {
    /// The inputs for this kernel invocation.
    args: &'args dyn ExecutionArgs,

    /// The input dtypes used by the planning visit.
    dtypes: &'args [DType],

    /// The plan selected by the planning visit, which this visit must reproduce.
    plan: &'args BatchPlan,

    /// The output dtype declared by [`RowVisitor::with_output_dtype`], if any.
    output_dtype: Option<DType>,

    /// The execution context used to decode the input columns.
    ctx: &'ctx mut ExecutionCtx,

    /// Ties this visit to the function used by its compile-time contract checks.
    function: PhantomData<F>,
}

impl<'args, 'ctx, F: RowFn> ExecuteRows<'args, 'ctx, F> {
    pub(crate) fn new(
        args: &'args dyn ExecutionArgs,
        dtypes: &'args [DType],
        plan: &'args BatchPlan,
        ctx: &'ctx mut ExecutionCtx,
    ) -> Self {
        Self {
            args,
            dtypes,
            plan,
            output_dtype: None,
            ctx,
            function: PhantomData,
        }
    }
}

impl<F: RowFn> private::Sealed for ExecuteRows<'_, '_, F> {}

impl<F: RowFn> RowVisitor for ExecuteRows<'_, '_, F> {
    type VisitResult = ArrayRef;

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
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_owned_output::<Args>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_owned_infallible::<Args, Out, Prepared>(self.args, self.ctx, prepare, apply)
    }

    fn visit_bool<Args, const MULTIVERSIONED: bool>(
        self,
        apply: impl Fn(Args::Elems<'_>) -> bool,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
    {
        const { assert_owned_visit_contract::<F, Args, bool>() };
        let visited = BatchPlan::new(
            validate_owned_visit::<Args, bool>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_owned_output::<Args>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_owned_infallible_bool::<Args, MULTIVERSIONED>(self.args, self.ctx, apply)
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
            validate_sink_visit::<Args, Sink>(self.dtypes, &params)?,
            self.output_dtype,
            RowPolicy::for_sink::<Args, ApplyResult>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_sink::<Args, Prepared, Sink, ApplyResult>(
            self.args, &params, self.ctx, prepare, apply,
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
        let visited = BatchPlan::new(
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_deferred_output::<Args>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_owned::<Args, Out, Prepared, Fail>(
            self.args,
            self.ctx,
            prepare,
            apply,
            finish_failure,
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
        let visited = BatchPlan::new(
            validate_owned_visit::<Args, bool>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_deferred_output::<Args>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_owned_bool::<Args, Prepared, Fail, MULTIVERSIONED>(
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
/// Owned outputs initialize skipped positions with [`Default::default`]. Output sinks use
/// their own skipped-row initializer.
pub(crate) struct ExecuteValidRows<'args, 'ctx, F: RowFn> {
    /// The original inputs for this kernel invocation.
    args: &'args dyn ExecutionArgs,

    /// The input dtypes used by the planning visit.
    dtypes: &'args [DType],

    /// The plan selected by the planning visit, which this visit must reproduce.
    plan: &'args BatchPlan,

    /// The output dtype declared by [`RowVisitor::with_output_dtype`], if any.
    output_dtype: Option<DType>,

    /// The conjoined validity, containing both valid and invalid rows.
    valid: MaskValuesRef,

    /// The execution context used to decode the input columns.
    ctx: &'ctx mut ExecutionCtx,

    /// Ties this visit to the function used by its compile-time contract checks.
    function: PhantomData<F>,
}

impl<'args, 'ctx, F: RowFn> ExecuteValidRows<'args, 'ctx, F> {
    pub(crate) fn new(
        args: &'args dyn ExecutionArgs,
        dtypes: &'args [DType],
        plan: &'args BatchPlan,
        valid: MaskValuesRef,
        ctx: &'ctx mut ExecutionCtx,
    ) -> Self {
        Self {
            args,
            dtypes,
            plan,
            output_dtype: None,
            valid,
            ctx,
            function: PhantomData,
        }
    }
}

impl<F: RowFn> private::Sealed for ExecuteValidRows<'_, '_, F> {}

impl<F: RowFn> RowVisitor for ExecuteValidRows<'_, '_, F> {
    type VisitResult = Option<ArrayRef>;

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
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_owned_output::<Args>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_owned_infallible_valid_rows::<Args, Out, Prepared>(
            self.args,
            &self.valid,
            self.ctx,
            prepare,
            apply,
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
        let visited = BatchPlan::new(
            validate_sink_visit::<Args, Sink>(self.dtypes, &params)?,
            self.output_dtype,
            RowPolicy::for_sink::<Args, ApplyResult>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_sink_valid_rows::<Args, Prepared, Sink, ApplyResult>(
            self.args,
            &self.valid,
            &params,
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
        Fail: FailureEvidence,
    {
        const { assert_deferred_visit_contract::<F, Args, Out, Fail>() };
        let visited = BatchPlan::new(
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_deferred_output::<Args>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_owned_valid_rows::<Args, Out, Prepared, Fail>(
            self.args,
            &self.valid,
            self.ctx,
            prepare,
            apply,
            finish_failure,
        )
    }
}

/// The runtime visit that executes valid rows over inputs filtered to the valid row domain.
///
/// Batch execution filters the inputs when a representation cannot decode null payloads. This
/// visit reads consecutive filtered rows and writes each output at its original row index, so the
/// result never needs a columnar scatter. Owned outputs initialize skipped positions with
/// [`Default::default`]. Output sinks use their own skipped-row initializer.
pub(crate) struct ExecuteFilteredRows<'args, 'ctx, F: RowFn> {
    /// The inputs filtered to the valid rows of the original batch.
    args: &'args dyn ExecutionArgs,

    /// The input dtypes used by the planning visit.
    dtypes: &'args [DType],

    /// The plan selected by the planning visit, which this visit must reproduce.
    plan: &'args BatchPlan,

    /// The output dtype declared by [`RowVisitor::with_output_dtype`], if any.
    output_dtype: Option<DType>,

    /// The conjoined validity over the original row domain.
    valid: MaskValuesRef,

    /// The execution context used to decode the input columns.
    ctx: &'ctx mut ExecutionCtx,

    /// Ties this visit to the function used by its compile-time contract checks.
    function: PhantomData<F>,
}

impl<'args, 'ctx, F: RowFn> ExecuteFilteredRows<'args, 'ctx, F> {
    pub(crate) fn new(
        args: &'args dyn ExecutionArgs,
        dtypes: &'args [DType],
        plan: &'args BatchPlan,
        valid: MaskValuesRef,
        ctx: &'ctx mut ExecutionCtx,
    ) -> Self {
        Self {
            args,
            dtypes,
            plan,
            output_dtype: None,
            valid,
            ctx,
            function: PhantomData,
        }
    }
}

impl<F: RowFn> private::Sealed for ExecuteFilteredRows<'_, '_, F> {}

impl<F: RowFn> RowVisitor for ExecuteFilteredRows<'_, '_, F> {
    type VisitResult = ArrayRef;

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
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_owned_output::<Args>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_owned_infallible_filtered::<Args, Out, Prepared>(
            self.args,
            &self.valid,
            self.ctx,
            prepare,
            apply,
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
        let visited = BatchPlan::new(
            validate_sink_visit::<Args, Sink>(self.dtypes, &params)?,
            self.output_dtype,
            RowPolicy::for_sink::<Args, ApplyResult>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_sink_filtered::<Args, Prepared, Sink, ApplyResult>(
            self.args,
            &self.valid,
            &params,
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
        Fail: FailureEvidence,
    {
        const { assert_deferred_visit_contract::<F, Args, Out, Fail>() };
        let visited = BatchPlan::new(
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_deferred_output::<Args>(),
        )?;
        self.plan.ensure_reproduced_by(&visited)?;

        execute_owned_filtered::<Args, Out, Prepared, Fail>(
            self.args,
            &self.valid,
            self.ctx,
            prepare,
            apply,
            finish_failure,
        )
    }
}
