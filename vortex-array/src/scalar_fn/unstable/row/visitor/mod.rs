// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Visits that plan or execute the concrete row signature selected by [`RowFn::dispatch`].
//!
//! [`RowFn::dispatch`]: crate::scalar_fn::unstable::row::RowFn::dispatch

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::dtype::DType;

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

fn ensure_plan(
    planned_output: &DType,
    planned_policy: RowPolicy,
    actual_output: DType,
    actual_policy: RowPolicy,
) -> VortexResult<()> {
    vortex_ensure_eq!(
        actual_policy,
        planned_policy,
        "row dispatch must select the planned nullable execution policy: planned {planned_policy:?}, got {actual_policy:?}",
    );
    vortex_ensure_eq!(
        actual_output,
        *planned_output,
        "row dispatch must select the planned output dtype: planned {planned_output}, got {actual_output}",
    );

    Ok(())
}
