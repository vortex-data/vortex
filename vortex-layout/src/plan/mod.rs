// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Physical plans for scans.
//!
//! A plan is a tree of physical operators over a row domain. Operator identity and
//! operator-specific state do not depend on the source layout kind, so rewrites can reason about a
//! plan's shape alone. The common child container can initialize individual slots lazily.

mod children;
mod display;
mod lower;
mod optimize;
mod plans;
mod typed;
mod vtable;

pub use children::PlanChildren;
pub use display::PlanIndentedFormatter;
pub use display::PlanSummaryExtractor;
pub use display::PlanTreeContext;
pub use display::PlanTreeDisplay;
pub use display::PlanTreeExtractor;
pub use lower::lower;
pub use optimize::optimize;
pub use plans::Concat;
pub use plans::ConcatData;
pub use plans::ConcatPlan;
pub use plans::Eval;
pub use plans::EvalData;
pub use plans::EvalPlan;
pub use plans::ListPack;
pub use plans::ListPackData;
pub use plans::ListPackPlan;
pub use plans::Pack;
pub use plans::PackData;
pub use plans::PackPlan;
pub use plans::RowIdx;
pub use plans::RowIdxData;
pub use plans::RowIdxPlan;
pub use plans::RowIdxPlanMetadata;
pub use plans::SegmentScan;
pub use plans::SegmentScanData;
pub use plans::SegmentScanPlan;
pub use plans::Take;
pub use plans::TakePlan;
pub use typed::DynPlan;
pub use typed::Plan;
pub use typed::PlanParts;
pub use typed::PlanRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
pub use vtable::PlanId;
pub use vtable::PlanVTable;

/// Returns an error when `children` does not have exactly `expected` entries.
pub(crate) fn check_child_count(
    name: &str,
    children: &PlanChildren,
    expected: usize,
) -> VortexResult<()> {
    if children.len() != expected {
        vortex_bail!(
            "{name} expects {expected} children but got {}",
            children.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
