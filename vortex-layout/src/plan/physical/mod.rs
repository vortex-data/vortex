// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Physical plans for scans.
//!
//! A plan is a tree of physical operators over a row domain. Operator identity and
//! operator-specific state do not depend on the source layout kind, so rewrites can reason about a
//! plan's shape alone. Children live in fixed-position slots, each of which can be built lazily.
//!
//! [`operator`] defines the execution contracts, `operators` implements the concrete sources
//! and stages, and [`pipeline`] compiles and runs them.

mod display;
pub mod operator;
pub(crate) mod operators;
mod optimize;
pub mod optimizer;
pub mod pipeline;
mod plans;
pub(crate) mod selection;
pub use plans::register_pack_partition_rules;
pub use plans::register_row_idx_partition_rules;
mod slots;
mod typed;
mod vtable;

pub use display::PlanIndentedFormatter;
pub use display::PlanSummaryExtractor;
pub use display::PlanTreeContext;
pub use display::PlanTreeDisplay;
pub use display::PlanTreeExtractor;
pub use optimize::optimize;
pub use plans::Concat;
pub use plans::ConcatData;
pub use plans::ConcatPlan;
pub use plans::Eval;
pub use plans::EvalData;
pub use plans::EvalPlan;
pub use plans::Filter;
pub use plans::FilterData;
pub use plans::FilterPlan;
pub use plans::LegacyStats;
pub use plans::LegacyStatsData;
pub use plans::LegacyStatsPlan;
pub use plans::ListPack;
pub use plans::ListPackData;
pub use plans::ListPackPlan;
pub use plans::Pack;
pub use plans::PackData;
pub use plans::PackPlan;
pub use plans::PackSlot;
pub use plans::RowIdx;
pub use plans::RowIdxData;
pub use plans::RowIdxPlan;
pub use plans::SegmentScan;
pub use plans::SegmentScanData;
pub use plans::SegmentScanPlan;
pub use plans::Share;
pub use plans::ShareData;
pub use plans::SharePlan;
pub use plans::Take;
pub use plans::TakePlan;
pub use plans::Zoned;
pub use plans::ZonedData;
pub use plans::ZonedPlan;
pub use plans::plan_row_idx_expression;
pub use plans::row_idx_dtype;
pub use slots::LazyPlanRef;
pub use slots::PlanSlots;
pub use typed::DynPlan;
pub use typed::Plan;
pub(crate) use typed::PlanByPtr;
pub use typed::PlanParts;
pub use typed::PlanRef;
pub use vtable::PlanId;
pub use vtable::PlanVTable;

#[cfg(test)]
mod tests;
