// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheduler prototype for LayoutPlan V2.
//!
//! This module contains the current partition-local scheduler experiment:
//! a priority queue for CPU/I/O/control work plus lowering support for
//! building morsel pipelines from a [`crate::v2::plans::LayoutPlan`].

#![allow(dead_code)]

mod lowering;
pub(crate) mod queue;

pub use lowering::LayoutLoweringCtx;
pub use lowering::LayoutSchedulerRunReport;
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
