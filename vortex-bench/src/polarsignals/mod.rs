// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! PolarSignals profiling data benchmark.
//!
//! The schema features a sparse struct (10 nullable label fields covering five
//! fill-rate tiers), deeply nested locations
//! (List<Struct<..., List<Struct<...>>>>), and several low-cardinality string
//! columns.

mod benchmark;
mod data;
mod schema;

pub use benchmark::*;
