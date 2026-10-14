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
//! [`BatchInputs`] collects them when a visit cannot finish the batch directly. [`BatchPlan`]
//! carries the nullable execution strategy selected by a concrete dispatch. [`RowFnExecutionArgs`]
//! applies that strategy, and [`BorrowedRowFnArgs`] pairs each kernel invocation with its planning
//! metadata.

use smallvec::SmallVec;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::dtype::Nullability;
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
/// A probe holds no input column. A visit that can finish the batch directly decodes the borrowed
/// inputs. Any other visit collects them into [`BatchInputs`].
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

    /// The strict result nullability: nullable when any input dtype is nullable.
    nullability: Nullability,
}

/// The same-length input columns collected for one row-function execution, with the facts of
/// their [`BatchProbe`].
pub(crate) struct BatchInputs {
    /// The function being executed, named in the errors this raises.
    id: ScalarFnId,

    /// The number of rows in the original execution scope.
    row_count: usize,

    /// The input columns, collected once for validity, constant folding, filtering, and execution.
    inputs: SmallVec<[ArrayRef; 4]>,

    /// The conjoined input validity. An output row is valid exactly when it is valid in every
    /// input. Conjoining is lazy, and null handling materializes the mask only when required.
    validity: Validity,

    /// Bit `i` is set when input `i` is a batch constant, seen through extension and masked
    /// wrappers.
    const_mask: u32,

    /// Whether any input is a null literal constant.
    has_null_const: bool,
}

/// The same-length input columns and metadata for one row-function execution.
pub(crate) struct RowFnExecutionArgs<'a> {
    /// The collected input columns and their conjoined validity.
    batch: &'a BatchInputs,

    /// The input dtypes, collected with the columns and reused by both planning and execution.
    arg_dtypes: &'a [DType],

    /// The output dtype, widened to nullable when any input is nullable. The finished column is
    /// reconciled against this dtype.
    result_dtype: DType,

    /// The storage dtype, output label, and null-handling policy selected while planning.
    plan: BatchPlan,
}

#[cfg(test)]
mod tests;
