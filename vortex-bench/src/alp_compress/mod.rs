// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ALP compression benchmark.
//!
//! Exercises ALP-RD encoding on synthetic f64 columns via SQL queries.
//! The data is generated with patterns that use full floating-point precision,
//! ensuring the compressor selects ALP-RD rather than regular ALP.

mod benchmark;
mod data;

pub use benchmark::AlpCompressBenchmark;
