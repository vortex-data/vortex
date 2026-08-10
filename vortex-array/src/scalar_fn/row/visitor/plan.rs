// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The visitor that validates a concrete dispatch and plans its nullable execution.

use std::marker::PhantomData;
use std::ops::BitOrAssign;

use vortex_error::VortexResult;

use super::RowVisitor;
use super::check::assert_deferred_visit_contract;
use super::check::assert_owned_visit_contract;
use super::check::assert_sink_visit_contract;
use super::check::validate_owned_visit;
use super::check::validate_sink_visit;
use super::private;
use crate::dtype::DType;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::IndexedElementTuple;
use crate::scalar_fn::OutputElement;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::RowFn;
use crate::scalar_fn::SinkResult;
use crate::scalar_fn::row::batch::BatchPlan;
use crate::scalar_fn::row::batch::RowPolicy;

/// The plan-time visit that validates dtypes and derives the nullable execution policy.
pub struct PlanRows<'a, F> {
    /// The input dtypes for this plan.
    dtypes: &'a [DType],

    /// The visited function, carried only so the dispatch check can name its contract.
    function: PhantomData<F>,
}

impl<'a, F> PlanRows<'a, F> {
    pub fn new(dtypes: &'a [DType]) -> Self {
        Self {
            dtypes,
            function: PhantomData,
        }
    }
}

impl<F> private::Sealed for PlanRows<'_, F> {}

impl<F: RowFn> RowVisitor for PlanRows<'_, F> {
    type VisitResult = BatchPlan;

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

        Ok(BatchPlan {
            output_dtype: validate_owned_visit::<Args, Out>(self.dtypes)?,
            policy: RowPolicy::for_owned_output::<Args>(),
        })
    }

    fn visit_prepared_into<Args, Sink, Prepared, ApplyResult>(
        self,
        _prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        _apply: impl Fn(&Prepared, Args::Elems<'_>, Sink::Row<'_>) -> ApplyResult,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: ElementTuple,
        Sink: OutputSink,
        ApplyResult: SinkResult<WriteToken = Sink::WriteToken>,
    {
        const { assert_sink_visit_contract::<F, Args, ApplyResult>() };

        Ok(BatchPlan {
            output_dtype: validate_sink_visit::<Args, Sink>(self.dtypes)?,
            policy: RowPolicy::for_sink::<Args, ApplyResult>(),
        })
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

        Ok(BatchPlan {
            output_dtype: validate_owned_visit::<Args, Out>(self.dtypes)?,
            policy: RowPolicy::for_deferred_output::<Args>(),
        })
    }
}
