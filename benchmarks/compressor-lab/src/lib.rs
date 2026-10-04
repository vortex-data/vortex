// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic planning for compressor-selector data generation.
//!
//! A [`PlanSpec`](spec::PlanSpec) (sources, columns, dtypes, schemes, search and measurement
//! parameters) expands into a [`Plan`](plan::Plan): a DAG of tasks whose keys are content
//! hashes of their inputs plus the code identity (Vortex version and git commit). The same
//! spec, data and commit always produce byte-identical plan files.
//!
//! Task keys do not depend on the plan they appear in, so plans that share work share outputs,
//! and any subset of tasks can run in any order, on any number of machines. Work is split with
//! [`Shard`](shard::Shard), and a task is done when its marker exists in the store's ledger.

pub mod identity;
pub mod key;
pub mod plan;
pub mod shard;
pub mod source;
pub mod spec;

#[cfg(test)]
mod tests;
