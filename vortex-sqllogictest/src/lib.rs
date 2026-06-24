// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Support library for the `sqllogictests` binary, which replays `.slt` files
//! against both DataFusion and DuckDB configured to read Vortex files.
//!
//! The binary itself lives in `bin/sqllogictests-runner.rs`; this crate exposes
//! the DuckDB [`sqllogictest::AsyncDB`](duckdb::DuckDB) adapter and file
//! discovery [`utils`].

pub mod duckdb;
pub mod normalize;
pub mod utils;
