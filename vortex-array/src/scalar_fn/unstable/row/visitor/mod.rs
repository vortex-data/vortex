// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Visits that plan or execute the concrete row signature selected by [`RowFn::dispatch`].
//!
//! [`RowFn::dispatch`]: crate::scalar_fn::unstable::row::RowFn::dispatch

mod check;
pub(super) use check::assert_owned_output_needs_no_drop;

mod execute;
pub(super) use execute::ExecuteRows;
pub(super) use execute::ExecuteValidRows;

mod retry;
pub(super) use retry::ExecuteDenseWithRetry;

mod plan;
pub(super) use plan::BatchPlan;
pub(super) use plan::BatchPlanner;
pub(super) use plan::RowPolicy;

mod row_visitor;
pub use row_visitor::RowVisitor;
