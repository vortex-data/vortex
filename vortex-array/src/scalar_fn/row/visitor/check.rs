// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Contract checks shared by planning and execution visits.
//!
//! Const assertions reject invalid generic visits during compilation. The validators compare a
//! selected visit with the input dtypes during planning and return its output dtype.

use std::mem::needs_drop;
use std::ops::BitOrAssign;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::IndexedElementTuple;
use crate::scalar_fn::OutputElement;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::RowFn;
use crate::scalar_fn::SinkResult;

/// Assert the no-drop contract that makes partially initialized output safe to abandon on unwind.
pub(in crate::scalar_fn::row) const fn assert_owned_output_needs_no_drop<T>() {
    assert!(
        !needs_drop::<T>(),
        "owned row outputs must not require drop glue"
    );
}

/// Assert that the input arity and decode fallibility match the function-wide declarations.
const fn assert_input_visit_contract<F: RowFn, Args: ElementTuple>() {
    assert!(
        Args::ARITY == F::ARG_NAMES.len(),
        "the visited argument tuple must have the arity declared by RowFn::ARG_NAMES",
    );
    // Dictionary pushdown treats an infallible function as safe to evaluate over values no code
    // references, so every dispatch must fit the function-wide declaration.
    assert!(
        !Args::DECODE_FALLIBLE || F::FALLIBLE,
        "RowFn::FALLIBLE must be true when input decoding can fail",
    );
}

/// Assert the input contract and that owned output values do not require drop glue.
pub(super) const fn assert_owned_visit_contract<Function, Args, Out>()
where
    Function: RowFn,
    Args: IndexedElementTuple,
    Out: OutputElement,
{
    assert_input_visit_contract::<Function, Args>();
    assert_owned_output_needs_no_drop::<Out>();
}

/// Assert that a sink visit obeys the input, fallibility, and deferred-error contracts.
pub(super) const fn assert_sink_visit_contract<Function, Args, ApplyResult>()
where
    Function: RowFn,
    Args: ElementTuple,
    ApplyResult: SinkResult,
{
    assert_input_visit_contract::<Function, Args>();
    assert!(
        !ApplyResult::FALLIBLE || Function::FALLIBLE,
        "RowFn::FALLIBLE must be true when a row result can fail",
    );
}

/// Assert the owned-output contract, fallibility declaration, and failure-evidence width bound.
pub(super) const fn assert_deferred_visit_contract<Function, Args, Out, Fail>()
where
    Function: RowFn,
    Args: IndexedElementTuple,
    Out: OutputElement,
    Fail: Copy + Default + BitOrAssign,
{
    assert_owned_visit_contract::<Function, Args, Out>();
    assert!(
        Function::FALLIBLE,
        "RowFn::FALLIBLE must be true when a row result defers failure evidence",
    );
    assert!(
        size_of::<Fail>() <= size_of::<Out>(),
        "failure evidence must be no wider than the value, or it bounds the vector width",
    );
}

/// Validate the input dtypes and return the non-nullable dtype built by `Out`.
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

/// Validate the input dtypes and return the non-nullable dtype built by `Sink`.
pub(super) fn validate_sink_visit<Args: ElementTuple, Sink: OutputSink>(
    dtypes: &[DType],
) -> VortexResult<DType> {
    Args::validate(dtypes)?;

    let dtype = Sink::sink_dtype(dtypes)?;
    vortex_ensure!(
        !dtype.is_nullable(),
        "row output sinks must declare a non-nullable dtype, got {dtype}",
    );

    Ok(dtype)
}
