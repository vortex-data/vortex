// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Physical plans for scans.
//!
//! A plan is a tree of physical operators over a row domain. Operator identity and
//! operator-specific state do not depend on the source layout kind, so rewrites can reason about a
//! plan's shape alone. Children live in fixed-position slots, each of which can be built lazily.
//!
//! [`operator`] defines execution contracts. Built-ins own their sources and stages, while
//! [`pipeline`] compiles and runs them.

mod display;
pub use crate::execution::operator;
pub mod optimize;
pub mod optimizer;
pub use crate::execution::pipeline;
pub(crate) mod builtins;
pub(crate) mod selection;
pub use builtins::register_pack_partition_rules;
pub use builtins::register_row_idx_partition_rules;
pub mod partition;
mod slots;
mod typed;
mod vtable;

pub use builtins::Concat;
pub use builtins::ConcatData;
pub use builtins::ConcatPlan;
pub use builtins::Eval;
pub use builtins::EvalData;
pub use builtins::EvalPlan;
pub use builtins::Filter;
pub use builtins::FilterData;
pub use builtins::FilterPlan;
pub use builtins::ListPack;
pub use builtins::ListPackData;
pub use builtins::ListPackPlan;
pub use builtins::Pack;
pub use builtins::PackData;
pub use builtins::PackPlan;
pub use builtins::PackSlot;
pub use builtins::RowIdx;
pub use builtins::RowIdxData;
pub use builtins::RowIdxPlan;
pub use builtins::Share;
pub use builtins::ShareData;
pub use builtins::SharePlan;
pub use builtins::Take;
pub use builtins::TakePlan;
pub use builtins::Zoned;
pub use builtins::ZonedData;
pub use builtins::ZonedPlan;
pub(crate) use builtins::plan_row_idx_expression;
#[doc(hidden)]
pub use builtins::zoned::is_zone_pruning;
pub use display::PlanIndentedFormatter;
pub use display::PlanSummaryExtractor;
pub use display::PlanTreeContext;
pub use display::PlanTreeDisplay;
pub use display::PlanTreeExtractor;
pub use optimize::optimize;
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
