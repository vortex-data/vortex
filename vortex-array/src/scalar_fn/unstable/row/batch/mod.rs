// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Batch execution for a strict row function.
//!
//! A batch is the set of same-length input columns supplied in one scalar-function call. A row
//! function handles typed values for one logical row. This module adds the columnar concerns around
//! that row function: planning the output and null handling, preserving batch constants,
//! propagating strict validity, selecting an execution strategy, and validating the finished
//! output.
//!
//! [`RowInputs`] borrows the input columns, and [`BatchProbe`] classifies them before dispatch.
//! [`BatchInputs`] keeps the columns with their probe. [`BatchPlan`] carries the nullable execution
//! strategy selected by a concrete dispatch. [`RowFnExecutionArgs`] applies that strategy, and
//! [`BorrowedRowFnArgs`] pairs each kernel invocation with its planning metadata.

use std::borrow::Cow;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFnId;
use crate::validity::Validity;

mod args;
pub(super) use args::BorrowedRowFnArgs;

mod inputs;
pub(crate) use inputs::ArrayInputs;
pub(crate) use inputs::RowInputs;

mod execute;
pub(super) use execute::finalize_kernel_output;

mod planning;

pub(super) use super::visitor::BatchPlan;
pub(super) use super::visitor::RowPolicy;

/// What batch execution derives from its borrowed inputs before a dispatch plans them.
///
/// A probe holds no input column. [`BatchInputs`] pairs it with the columns.
pub(crate) struct BatchProbe {
    /// The number of rows in the original execution scope.
    row_count: usize,

    /// The conjoined input validity. An output row is valid exactly when it is valid in every
    /// input. Conjoining is lazy, and null handling materializes the mask only when required.
    validity: Validity,

    /// Bit `i` is set when input `i` is a batch constant, seen through extension and masked
    /// wrappers.
    const_mask: u32,

    /// Whether any input is a null literal constant.
    has_null_const: bool,
}

/// The same-length input columns of one row-function execution, with their [`BatchProbe`].
pub(crate) struct BatchInputs<'a> {
    /// The function being executed, named in the errors this raises.
    id: ScalarFnId,

    /// The input columns, borrowed when the caller holds them as one slice and cloned otherwise.
    inputs: Cow<'a, [ArrayRef]>,

    /// What batch execution derived from `inputs` before planning.
    probe: BatchProbe,
}

/// The same-length input columns and metadata for one row-function execution.
pub(crate) struct RowFnExecutionArgs<'a> {
    /// The input columns and their conjoined validity.
    batch: &'a BatchInputs<'a>,

    /// The input dtypes, shared by planning and execution.
    arg_dtypes: &'a [DType],

    /// The output dtype, widened to nullable when any input is nullable. The finished column is
    /// reconciled against this dtype.
    result_dtype: DType,

    /// The storage dtype, output label, and null-handling policy selected while planning.
    plan: BatchPlan,
}

#[cfg(test)]
mod tests;
