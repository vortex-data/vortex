// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! **Experimental proof of concept.** This crate is not a table format. It is an example of
//! Vortex's scanning API applied to many files, and its API may change or be removed without
//! notice. It is not published, and no other crate in the workspace depends on it.
//!
//! The table is deliberately minimal. A [`Manifest`], itself a Vortex file, lists the data files
//! with where each one's footer starts and, optionally, each file's column statistics as
//! [`TableStats`]. A [`TableScan`] drops the files whose statistics rule out the filter, opens
//! each remaining file through an [`Openable`], several at once and each with one read of its
//! footer, and runs each kept file's ordinary scan, in table order. See `examples/table_scan.rs`.

pub mod manifest;
pub mod open;
pub mod scan;
pub mod stats;

pub use manifest::Manifest;
pub use manifest::ManifestEntry;
pub use open::FileSystemOpenable;
pub use open::Openable;
pub use open::ReadAtOpenable;
pub use scan::PlannedFile;
pub use scan::TableScan;
pub use stats::TableStats;

#[cfg(test)]
mod tests;
