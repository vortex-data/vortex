// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! **Experimental proof of concept.** This crate is not a table format. It is an example of
//! Vortex's scanning API applied to many files, and its API may change or be removed without
//! notice. It is not published, and no other crate in the workspace depends on it.
//!
//! The table it will hold is deliberately minimal: a manifest, itself a Vortex file, lists the
//! data files with their footers and file-level statistics. A scan prunes files on those
//! statistics in one evaluation, opens the files it keeps from their recorded footers, and runs
//! each file's ordinary scan.
//!
//! Planned contents, each added separately:
//!
//! - the manifest: one row per data file with its footer;
//! - pruning every file at once on a columnar table of the files' statistics;
//! - a planner that scans the files the manifest keeps, and an example that runs it.

pub mod manifest;

pub use manifest::Manifest;
pub use manifest::ManifestEntry;

#[cfg(test)]
mod tests;
