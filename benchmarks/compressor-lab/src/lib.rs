// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic, checkpointed data generation for learning compression scheme selection.
//!
//! A [`PlanSpec`](spec::PlanSpec) (sources, columns, dtypes, schemes, search and measurement
//! parameters) expands into a [`Plan`](plan::Plan): tasks whose keys are content hashes of their
//! inputs plus the code identity (Vortex version and git commit). The same spec, data and commit
//! always produce byte-identical plan files.
//!
//! Tasks run against a [`Store`](store::Store). Fact tasks (chunks, features, candidate
//! encodings) are upserted by key and skipped once done, so runs resume, split into
//! [`Shard`](shard::Shard)s and spread over machines. Timing observations are appended per
//! machine and run, so repeated runs add samples and other architectures stay separate.
//! [`materialize`](materialize::materialize) turns a store into training tables.

// Measurement code converts between counts, nanoseconds and floats on purpose; every value it
// casts is far below the narrower type's range.
#![allow(clippy::cast_possible_truncation)]

pub mod blob;
pub mod candidates;
pub mod features;
pub mod identity;
pub mod key;
pub mod load;
pub mod machine;
pub mod materialize;
pub mod model;
pub mod plan;
pub mod runner;
pub mod shard;
pub mod source;
pub mod spec;
pub mod store;
pub mod synthetic;
pub mod tpch;
pub mod wrap;

#[cfg(test)]
mod tests;
