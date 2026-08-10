// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Batch execution around a non-null row kernel.
//!
//! A row kernel handles typed values for one row. This module adds the columnar concerns around it:
//! planning the output and null strategy, preserving batch constants and encodings, propagating
//! strict validity, selecting an execution strategy, and validating the finished output.
//!
//! [`policy`] derives the nullable execution strategy from a concrete dispatch. [`execution`]
//! applies that strategy, and [`args`] pairs each kernel invocation with its planning metadata.

mod args;
pub(super) use args::KernelArgs;

mod execution;
pub(super) use execution::Batch;
pub(super) use execution::finalize_kernel_output;

mod policy;
pub(super) use policy::BatchPlan;
pub(super) use policy::RowPolicy;

#[cfg(test)]
mod tests;
