// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One-shot historical migrator from v2's S3-hosted benchmark dataset
//! to a v3 DuckDB file.
//!
//! The v2 dataset is JSONL of bare benchmark records keyed by name string.
//! v3 uses five typed fact tables with explicit dim columns. This crate
//! ports v2's `getGroup` classifier (in `benchmarks-website/server.js`)
//! bug-for-bug so that historical rows survive the migration with the
//! same group / chart / series structure as the live v2 server.
//!
//! The migrator is throwaway: once v3 cuts over, both the binary and
//! the classifier go away.

pub mod classifier;
pub mod commits;
pub mod migrate;
pub mod source;
pub mod v2;
pub mod verify;
