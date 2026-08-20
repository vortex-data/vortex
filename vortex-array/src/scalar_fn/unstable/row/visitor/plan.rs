// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plans the concrete signature selected by [`RowFn::dispatch`].
//!
//! [`BatchPlanner`] validates input and output dtypes, then records the output dtype and
//! null-handling policy that execution must reproduce.

use std::marker::PhantomData;

use vortex_error::VortexResult;

use super::RowVisitor;
use super::check::assert_deferred_visit_contract;
use super::check::assert_owned_visit_contract;
use super::check::assert_sink_visit_contract;
use super::check::validate_owned_visit;
use super::check::validate_sink_visit;
use super::row_visitor::private;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::SinkResult;

/// A planning visitor that validates dtypes and selects the nullable execution policy.
pub(crate) struct BatchPlanner<'a, F: RowFn> {
    dtypes: &'a [DType],

    options: &'a F::Options,

    /// Ties the planner to the function used by its compile-time contract checks.
    function: PhantomData<F>,
}

impl<'a, F: RowFn> BatchPlanner<'a, F> {
    pub(crate) fn new(dtypes: &'a [DType], options: &'a F::Options) -> Self {
        Self {
            dtypes,
            options,
            function: PhantomData,
        }
    }
}

impl<F: RowFn> private::Sealed for BatchPlanner<'_, F> {}

impl<F: RowFn> RowVisitor<F::Options> for BatchPlanner<'_, F> {
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
        _apply: impl Fn(
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
        Ok(BatchPlan {
            output_dtype: validate_sink_visit::<Args, Sink, F::Options>(self.options, self.dtypes)?,
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
        Fail: FailureEvidence,
    {
        const { assert_deferred_visit_contract::<F, Args, Out, Fail>() };
        Ok(BatchPlan {
            output_dtype: validate_owned_visit::<Args, Out>(self.dtypes)?,
            policy: RowPolicy::for_deferred_output::<Args>(),
        })
    }
}

/// The execution policy and output dtype selected by a planning visit.
pub(crate) struct BatchPlan {
    /// The non-nullable dtype built by the selected output capability.
    pub(crate) output_dtype: DType,

    /// How this concrete dispatch executes nullable rows.
    pub(crate) policy: RowPolicy,
}

impl BatchPlan {
    /// Return the output dtype widened with strict input nullability.
    pub(crate) fn result_dtype(&self, args: &[DType]) -> DType {
        let nullability = self.output_dtype.nullability()
            | Nullability::from(args.iter().any(DType::is_nullable));

        self.output_dtype.with_nullability(nullability)
    }
}

/// The nullable execution policy derived from one concrete dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowPolicy {
    /// Evaluate all rows and mask the result.
    Dense,

    /// Evaluate all rows, retrying only valid rows if reduced failure evidence reports an error.
    DenseWithRetry,

    /// Execute only valid rows over the original inputs.
    ValidOnly,
}

impl RowPolicy {
    /// The policy for an infallible owned output.
    pub(crate) const fn for_owned_output<Args: ElementTuple>() -> Self {
        if Args::DENSE_SAFE && Args::DECODE_INFALLIBLE {
            Self::Dense
        } else {
            Self::ValidOnly
        }
    }

    /// The policy for an owned output carrying batch-deferred failure evidence.
    pub(crate) const fn for_deferred_output<Args: ElementTuple>() -> Self {
        if Args::DENSE_SAFE && Args::DECODE_INFALLIBLE {
            Self::DenseWithRetry
        } else {
            Self::ValidOnly
        }
    }

    /// The policy for a sink-writing output.
    pub(crate) const fn for_sink<Args: ElementTuple, ApplyResult: SinkResult>() -> Self {
        if Args::DENSE_SAFE && Args::DECODE_INFALLIBLE && ApplyResult::INFALLIBLE {
            Self::Dense
        } else {
            Self::ValidOnly
        }
    }
}
