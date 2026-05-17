// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheduler prototype for LayoutPlan V2.
//!
//! This module contains the current partition-local scheduler experiment:
//! output-frontier grants, a priority queue for CPU/I/O/control work, and the
//! compatibility bridge that drives an existing [`crate::v2::plans::LayoutPlan`]
//! into a scheduler-owned sink.

#![allow(dead_code)]

pub(crate) mod frontier;
mod lowering;
pub(crate) mod queue;

#[cfg(test)]
pub(crate) use frontier::ConjunctFrontierController;
#[cfg(test)]
pub(crate) use frontier::ConjunctFrontierPolicy;
#[cfg(test)]
pub(crate) use frontier::FrontierSource;
pub(crate) use frontier::OutputEstimate;
pub use frontier::OutputFrontier;
#[cfg(test)]
pub(crate) use frontier::OutputGrantReason;
#[cfg(test)]
pub(crate) use frontier::OutputGrantRequest;
#[cfg(test)]
pub(crate) use frontier::OutputGrantor;
pub use lowering::LayoutLoweringCtx;
pub use lowering::LayoutSchedulerRunReport;
pub(crate) use lowering::execute_with_single_scheduler;
#[cfg(test)]
pub(crate) use queue::IoRequestId;
#[cfg(test)]
pub(crate) use queue::MorselEstimate;
#[cfg(test)]
pub(crate) use queue::MorselId;
#[cfg(test)]
pub(crate) use queue::MorselPriority;
#[cfg(test)]
pub(crate) use queue::MorselRole;
#[cfg(test)]
pub(crate) use queue::PartitionScheduler;
#[cfg(test)]
pub(crate) use queue::PartitionSchedulerId;
#[cfg(test)]
pub(crate) use queue::PipelineId;
#[cfg(test)]
pub(crate) use queue::SchedulerBudget;
#[cfg(test)]
pub(crate) use queue::SchedulerControlEvent;
#[cfg(test)]
pub(crate) use queue::SchedulerMorsel;
#[cfg(test)]
pub(crate) use queue::SchedulerSegmentTask;
#[cfg(test)]
pub(crate) use queue::SchedulerStep;
#[cfg(test)]
pub(crate) use queue::SchedulerTask;
#[cfg(test)]
pub(crate) use queue::SchedulerWorkTask;

#[cfg(test)]
mod tests;
