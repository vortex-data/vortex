// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// A heuristic for an ideal split size.
///
/// We don't actually know if this is right, but it is probably a good estimate.
const IDEAL_SPLIT_SIZE: u64 = 100_000;

pub mod arrow;
mod filter;
pub mod row_mask;
mod splits;
mod tasks;

mod selection;
pub use selection::Selection;

mod split_by;
pub use split_by::SplitBy;

mod scan_builder;
pub use scan_builder::ScanBuilder;

#[cfg(gpu_unstable)]
pub mod gpu;
mod repeated_scan;

pub use repeated_scan::RepeatedScan;
