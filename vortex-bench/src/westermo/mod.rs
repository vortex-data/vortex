// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Westermo test system performance benchmark.
//!
//! Real Prometheus node_exporter metrics from nineteen test servers, sampled every 30 seconds for
//! a month: the Westermo test system performance data set
//! (<https://github.com/westermo/test-system-performance-dataset>, CC BY 4.0, by P E Strandberg
//! and Y Marklund). The CSVs are converted to Prometheus layout: one row per sample, labels in a
//! struct, rows sorted by label set and then by timestamp.

mod benchmark;
mod data;
mod schema;

pub use benchmark::*;
