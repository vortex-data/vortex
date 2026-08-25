// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Contract checks shared by planning and execution visits.
//!
//! Const assertions reject invalid generic visits during compilation. The validators compare a
//! selected visit with the input dtypes during planning and return its output dtype.

use std::mem::needs_drop;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::SinkResult;

/// Assert the no-drop contract that makes partially initialized output safe to abandon on unwind.
pub(crate) const fn assert_owned_output_needs_no_drop<T>() {
    assert!(
        !needs_drop::<T>(),
        "owned row outputs must not require drop glue"
    );
}

const fn assert_input_visit_contract<F: RowFn, Args: ElementTuple>() {
    assert!(
        Args::ARITY == F::ARG_NAMES.len(),
        "the visited argument tuple must have the arity declared by RowFn::ARG_NAMES",
    );
}

pub(super) const fn assert_owned_visit_contract<Function, Args, Out>()
where
    Function: RowFn,
    Args: IndexedElementTuple,
    Out: OutputElement,
{
    assert_input_visit_contract::<Function, Args>();
    assert_owned_output_needs_no_drop::<Out>();
}

pub(super) const fn assert_sink_visit_contract<Function, Args, ApplyResult>()
where
    Function: RowFn,
    Args: ElementTuple,
    ApplyResult: SinkResult,
{
    assert_input_visit_contract::<Function, Args>();
    assert!(
        ApplyResult::INFALLIBLE || !Function::INFALLIBLE,
        "RowFn::INFALLIBLE must be false when a row result can fail",
    );
}

pub(super) const fn assert_deferred_visit_contract<Function, Args, Out, Fail>()
where
    Function: RowFn,
    Args: IndexedElementTuple,
    Out: OutputElement,
    Fail: FailureEvidence,
{
    assert_owned_visit_contract::<Function, Args, Out>();
    assert!(
        !Function::INFALLIBLE,
        "RowFn::INFALLIBLE must be false when a row result defers failure evidence",
    );
    assert!(
        size_of::<Fail>() <= size_of::<Out>(),
        "failure evidence must be no wider than the value, or it bounds the vector width",
    );
}

pub(super) fn validate_owned_visit<Args: ElementTuple, Out: OutputElement>(
    dtypes: &[DType],
) -> VortexResult<DType> {
    Args::validate(dtypes)?;

    let dtype = Out::element_dtype();
    vortex_ensure!(
        !dtype.is_nullable(),
        "row output elements must declare a non-nullable dtype, got {dtype}",
    );

    Ok(dtype)
}

pub(super) fn validate_sink_visit<Args, Sink>(dtypes: &[DType]) -> VortexResult<DType>
where
    Args: ElementTuple,
    Sink: OutputSink,
{
    Args::validate(dtypes)?;

    let dtype = Sink::storage_dtype();
    vortex_ensure!(
        !dtype.is_nullable(),
        "row output sinks must declare a non-nullable dtype, got {dtype}",
    );

    Ok(dtype)
}
