// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plans the concrete signature selected by [`RowFn::dispatch`].
//!
//! [`BatchPlanner`] validates input and output dtypes, then records the [`BatchPlan`] that
//! execution must reproduce: the dtype the dispatched capability builds, the dtype the function
//! returns, and the null-handling policy.

use std::marker::PhantomData;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use super::RowVisitor;
use super::check::assert_deferred_visit_contract;
use super::check::assert_owned_visit_contract;
use super::check::assert_sink_visit_contract;
use super::check::validate_owned_visit;
use super::check::validate_sink_visit;
use super::row_visitor::private;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::extension::ExtDTypeRef;
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

    /// The output dtype declared by [`RowVisitor::with_output_dtype`], if any.
    output_dtype: Option<DType>,

    /// Ties the planner to the function used by its compile-time contract checks.
    function: PhantomData<F>,
}

impl<'a, F: RowFn> BatchPlanner<'a, F> {
    pub(crate) fn new(dtypes: &'a [DType]) -> Self {
        Self {
            dtypes,
            output_dtype: None,
            function: PhantomData,
        }
    }
}

impl<F: RowFn> private::Sealed for BatchPlanner<'_, F> {}

impl<F: RowFn> RowVisitor for BatchPlanner<'_, F> {
    type VisitResult = BatchPlan;

    fn with_output_dtype(mut self, dtype: DType) -> Self {
        self.output_dtype = Some(dtype);
        self
    }

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

        BatchPlan::new(
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_owned_output::<Args>(),
        )
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

        BatchPlan::new(
            validate_sink_visit::<Args, Sink>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_sink::<Args, ApplyResult>(),
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

        BatchPlan::new(
            validate_owned_visit::<Args, Out>(self.dtypes)?,
            self.output_dtype,
            RowPolicy::for_deferred_output::<Args>(),
        )
    }
}

/// The output dtypes and execution policy selected by one planning visit.
///
/// The _storage dtype_ is what the dispatched [`OutputElement`] or [`OutputSink`] physically
/// builds. The _output dtype_ is what the function returns. They differ when a dispatch declares a
/// label through [`RowVisitor::with_output_dtype`], which is how a function derives an extension
/// output dtype from its options or argument dtypes.
pub(crate) struct BatchPlan {
    /// The non-nullable dtype the dispatched output capability builds. Kernel output is validated
    /// against this dtype before any label is applied.
    storage_dtype: DType,

    /// The extension dtype labelled onto the finished column, when the declared output dtype
    /// differs from the storage dtype.
    output_label: Option<ExtDTypeRef>,

    /// How this concrete dispatch executes nullable rows.
    policy: RowPolicy,
}

impl BatchPlan {
    /// Plan an output built as `storage_dtype` and returned as `output_dtype`.
    ///
    /// `output_dtype` is the dtype a dispatch declared through
    /// [`RowVisitor::with_output_dtype`], or `None` to return the storage dtype unchanged.
    pub(crate) fn new(
        storage_dtype: DType,
        output_dtype: Option<DType>,
        policy: RowPolicy,
    ) -> VortexResult<Self> {
        let output_label = match output_dtype {
            Some(output_dtype) => validate_output_label(&storage_dtype, output_dtype)?,
            None => None,
        };

        Ok(Self {
            storage_dtype,
            output_label,
            policy,
        })
    }

    /// Return the non-nullable dtype the dispatched output capability builds.
    pub(crate) fn storage_dtype(&self) -> &DType {
        &self.storage_dtype
    }

    /// Return the non-nullable dtype the function returns.
    pub(crate) fn output_dtype(&self) -> DType {
        match &self.output_label {
            Some(output_label) => DType::Extension(output_label.clone()),
            None => self.storage_dtype.clone(),
        }
    }

    /// Return how this concrete dispatch executes nullable rows.
    pub(crate) fn policy(&self) -> RowPolicy {
        self.policy
    }

    /// Return the output dtype widened with strict input nullability.
    pub(crate) fn result_dtype(&self, args: &[DType]) -> DType {
        let output_dtype = self.output_dtype();
        let nullability =
            output_dtype.nullability() | Nullability::from(args.iter().any(DType::is_nullable));

        output_dtype.with_nullability(nullability)
    }

    /// Label `values` with the output dtype, preserving their nullability and every value.
    ///
    /// This is not a cast. [`new`](Self::new) accepted only a label that reinterprets the storage
    /// column, so this wraps rather than converts, and an extension dtype takes its nullability
    /// from the storage column it wraps.
    pub(crate) fn relabel_output(&self, values: ArrayRef) -> VortexResult<ArrayRef> {
        let Some(output_label) = &self.output_label else {
            return Ok(values);
        };

        let output_label = output_label.with_nullability(values.dtype().nullability());

        Ok(ExtensionArray::try_new(output_label, values)?.into_array())
    }

    /// Ensure an executing dispatch reproduced the planned output and policy.
    pub(crate) fn ensure_reproduced_by(&self, actual: &Self) -> VortexResult<()> {
        vortex_ensure_eq!(
            actual.policy,
            self.policy,
            "row dispatch must select the planned nullable execution policy: planned {:?}, got {:?}",
            self.policy,
            actual.policy,
        );
        vortex_ensure_eq!(
            actual.storage_dtype,
            self.storage_dtype,
            "row dispatch must select the planned storage dtype: planned {}, got {}",
            self.storage_dtype,
            actual.storage_dtype,
        );
        vortex_ensure!(
            actual.output_label == self.output_label,
            "row dispatch must declare the planned output dtype: planned {}, got {}",
            self.output_dtype(),
            actual.output_dtype(),
        );

        Ok(())
    }
}

/// Validate a declared output dtype and return the label to apply to the storage column.
///
/// A declared dtype **must** be non-nullable and **must** leave every value unchanged, which
/// restricts it to `storage_dtype` itself or to an extension dtype storing exactly that dtype.
/// Returning `None` for the former keeps [`BatchPlan::relabel_output`] a no-op for it.
///
/// This is the single validation point for that relationship, which
/// [`BatchPlan::relabel_output`] relies on.
fn validate_output_label(
    storage_dtype: &DType,
    output_dtype: DType,
) -> VortexResult<Option<ExtDTypeRef>> {
    vortex_ensure!(
        !output_dtype.is_nullable(),
        "a declared row output dtype must be non-nullable, got {output_dtype}",
    );

    if output_dtype == *storage_dtype {
        return Ok(None);
    }

    let DType::Extension(output_label) = output_dtype else {
        vortex_bail!(
            "a declared row output dtype must label the storage dtype {storage_dtype} without \
             changing any value, got {output_dtype}",
        );
    };
    vortex_ensure_eq!(
        *output_label.storage_dtype(),
        *storage_dtype,
        "a declared row extension output dtype must store {storage_dtype}, got {}",
        output_label.storage_dtype(),
    );

    Ok(Some(output_label))
}

/// The nullable execution policy derived from one concrete dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowPolicy {
    /// Evaluate all rows and mask the result.
    Dense,

    /// Evaluate all rows, then retry a partially valid batch if reduced failure evidence reports an
    /// error.
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
