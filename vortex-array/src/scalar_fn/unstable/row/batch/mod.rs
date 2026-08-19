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
//! [`BatchPlan`] carries the nullable execution strategy selected by a concrete dispatch. [`Batch`]
//! applies that strategy, and [`BorrowedExecutionArgs`] pairs each kernel invocation with its
//! planning metadata.

use smallvec::SmallVec;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFnId;
use crate::validity::Validity;

mod args;
pub(super) use args::BorrowedExecutionArgs;

mod execute;
#[cfg(test)]
pub(super) use execute::finalize_kernel_output;

mod planning;

pub(super) use super::visitor::BatchPlan;
pub(super) use super::visitor::RowPolicy;

/// The same-length input columns and metadata for one row-function execution.
pub(crate) struct Batch {
    /// The function being executed, named in the errors this raises.
    id: ScalarFnId,

    /// The number of rows in the original execution scope.
    row_count: usize,

    /// The input columns, collected once for validity, constant handling, and execution.
    inputs: SmallVec<[ArrayRef; 4]>,

    /// The input dtypes, collected with the columns and reused by both planning and execution.
    arg_dtypes: SmallVec<[DType; 4]>,

    /// The conjoined input validity. An output row is valid exactly when it is valid in every
    /// input. Conjoining is lazy, and null handling materializes the mask only when required.
    validity: Validity,

    /// The declared output dtype, widened to nullable when any input is nullable. Kernel output is
    /// reconciled against this dtype.
    result_dtype: DType,

    /// The non-nullable dtype the dispatched output capability builds, computed while planning.
    output_dtype: DType,

    /// How the concrete dispatch executes nullable rows.
    policy: RowPolicy,
}

#[cfg(test)]
mod tests;
