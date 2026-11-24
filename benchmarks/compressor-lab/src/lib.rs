// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic, checkpointed data generation for learning compression scheme selection.
//!
//! A [`PlanSpec`](spec::PlanSpec) (sources, columns, dtypes, schemes, search and measurement
//! parameters) expands into a [`Plan`](plan::Plan): tasks whose keys are content hashes of their
//! inputs plus the code identity (Vortex version and git commit). The same spec, data and commit
//! always produce byte-identical plan files.
//!
//! Tasks run against a [`Store`](store::Store) on local disk or any object store. Fact tasks
//! (chunks, features, candidate encodings) write Parquet segments plus a ledger and are skipped
//! once done, so runs resume, split into [`Shard`](shard::Shard)s and spread over machines that
//! all write to one store. No encoded arrays are kept: the time stage replays each plan on any
//! machine and appends samples to that machine's partition.
//! [`materialize`](materialize::materialize) turns a store into training tables.

// Measurement code converts between counts, nanoseconds and floats on purpose; every value it
// casts is far below the narrower type's range.
#![allow(clippy::cast_possible_truncation)]

pub mod blob;
pub mod candidates;
pub mod chunks;
pub mod eval;
pub mod features;
pub mod fetch;
pub mod gc;
pub mod identity;
pub mod key;
pub mod load;
pub mod machine;
pub mod materialize;
pub mod model;
pub mod oracle;
pub mod plan;
pub mod queue;
pub mod runner;
pub mod segment;
pub mod selector;
pub mod shard;
pub mod source;
pub mod spec;
pub mod store;
pub mod synthetic;
pub mod table;
pub mod tpch;
pub mod wrap;

#[cfg(test)]
mod tests;
