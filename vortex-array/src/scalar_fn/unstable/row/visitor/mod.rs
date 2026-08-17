// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Visits that plan or execute the concrete row signature selected by [`RowFn::dispatch`].
//!
//! [`RowFn::dispatch`]: crate::scalar_fn::unstable::row::RowFn::dispatch

mod check;
pub(super) use check::assert_owned_output_needs_no_drop;

// TODO(connor)[RowFn]: Remove this expectation when #9450 constructs the execution visitors.
#[expect(dead_code)]
mod execute;

mod plan;
pub(super) use plan::BatchPlanner;
pub(super) use plan::RowPolicy;

mod row_visitor;
pub use row_visitor::RowVisitor;
