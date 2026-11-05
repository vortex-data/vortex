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
pub use plans::ConcatPlanSlotsExt;
pub use plans::ConcatSlots;
pub use plans::ConcatSlotsView;
pub use plans::Eval;
pub use plans::EvalData;
pub use plans::EvalPlan;
pub use plans::EvalPlanSlotsExt;
pub use plans::EvalSlots;
pub use plans::EvalSlotsView;
pub use plans::Filter;
pub use plans::FilterData;
pub use plans::FilterPlan;
pub use plans::FilterPlanSlotsExt;
pub use plans::FilterSlots;
pub use plans::FilterSlotsView;
pub use plans::LegacyStats;
pub use plans::LegacyStatsData;
pub use plans::LegacyStatsPlan;
pub use plans::LegacyStatsPlanSlotsExt;
pub use plans::LegacyStatsSlots;
pub use plans::LegacyStatsSlotsView;
pub use plans::ListPack;
pub use plans::ListPackData;
pub use plans::ListPackPlan;
pub use plans::ListPackPlanSlotsExt;
pub use plans::ListPackSlots;
pub use plans::ListPackSlotsView;
pub use plans::Pack;
pub use plans::PackData;
pub use plans::PackPlan;
pub use plans::PackPlanSlotsExt;
pub use plans::PackSlots;
pub use plans::PackSlotsView;
pub use plans::RowIdx;
pub use plans::RowIdxData;
pub use plans::RowIdxPlan;
pub use plans::RowIdxPlanSlotsExt;
pub use plans::RowIdxSlots;
pub use plans::RowIdxSlotsView;
pub use plans::SegmentScan;
pub use plans::SegmentScanData;
pub use plans::SegmentScanPlan;
pub use plans::SegmentScanPlanSlotsExt;
pub use plans::SegmentScanSlots;
pub use plans::SegmentScanSlotsView;
pub use plans::Share;
pub use plans::ShareData;
pub use plans::SharePlan;
pub use plans::SharePlanSlotsExt;
pub use plans::ShareSlots;
pub use plans::ShareSlotsView;
pub use plans::Take;
pub use plans::TakePlan;
pub use plans::TakePlanSlotsExt;
pub use plans::TakeSlots;
pub use plans::TakeSlotsView;
pub use plans::Zoned;
pub use plans::ZonedData;
pub use plans::ZonedPlan;
pub use plans::ZonedPlanSlotsExt;
pub use plans::ZonedSlots;
pub use plans::ZonedSlotsView;
pub use plans::plan_row_idx_expression;
pub use plans::row_idx_dtype;
pub use slots::LazyPlanRef;
pub use slots::PlanSlotSlice;
pub use slots::PlanSlots;
pub use typed::DynPlan;
pub use typed::Plan;
pub(crate) use typed::PlanByPtr;
pub use typed::PlanParts;
pub use typed::PlanRef;
/// Generates typed slot constants, a view, and accessors for a plan operator's child slots.
pub use vortex_array_macros::plan_slots;
pub use vtable::PlanId;
pub use vtable::PlanVTable;

#[cfg(test)]
mod tests;
