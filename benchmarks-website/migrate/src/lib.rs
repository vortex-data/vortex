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

/// Routing v2 records into v3 fact tables, ported from v2's `getGroup`.
pub mod classifier;
/// V2 commit -> v3 `commits` row upserts.
pub mod commits;
/// End-to-end migration of v2 dumps into a v3 DuckDB.
pub mod migrate;
/// Streaming readers for the v2 S3 bucket and local dumps.
pub mod source;
/// Wire shapes of the v2 benchmark dataset.
pub mod v2;
/// Structural diff between a migrated v3 DuckDB and v2's `/api/metadata`.
pub mod verify;
