// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A minimal table format over Vortex files.
//!
//! A table is a set of Vortex data files and a [`Manifest`] naming them, which is itself a Vortex
//! file. The manifest records each file's size, row count, and footer region, optionally with the
//! region's bytes inlined, so a scan plans every file from the manifest alone: it prunes files on
//! the column statistics their footers carry and opens the rest without searching for footers.
//!
//! A [`TableScan`] is nothing but planners. It composes a table planner, one footer planner per
//! file, and each file's own scan planners from `vortex-file`, and declares each of them
//! independent of the others. Any [`PlanDriver`](vortex_scan::planning::drive::PlanDriver) runs
//! it: [`SequentialDriver`](vortex_scan::planning::drive::SequentialDriver) one file after another
//! on the polling task, [`ParallelDriver`](vortex_scan::planning::drive::ParallelDriver) across a
//! pool of workers. An index narrows a scan by handing it the rows to read per file, with
//! [`TableScan::with_file_selections`].

pub mod footer;
pub mod manifest;
pub mod scan;
pub mod storage;
pub mod writer;

pub use manifest::Manifest;
pub use manifest::ManifestEntry;
pub use scan::TableScan;
pub use storage::LocalStorage;
pub use storage::MemoryStorage;
pub use storage::TableStorage;
pub use writer::ManifestBuilder;

#[cfg(test)]
mod tests;
