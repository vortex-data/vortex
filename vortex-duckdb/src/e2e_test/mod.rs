// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for DuckDB scans and pushdown.
//!
//! These tests exercise the registered extension through scans, spatial pushdown, and S3 reads.

#[cfg(test)]
mod s3_test;
#[cfg(all(test, duckdb_release))]
mod spatial_pushdown_test;
#[cfg(test)]
mod vortex_scan_test;
