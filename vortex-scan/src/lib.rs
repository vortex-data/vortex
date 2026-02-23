// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(ngates): this crate is being re-purposed to expose the Scan API, rather than any specific
//  implementation of it. Much of the current code is about implementing the Scan API over a
//  Vortex Layout tree. This should move to the vortex-layout crate.

/// A heuristic for an ideal split size.
///
/// We don't actually know if this is right, but it is probably a good estimate.
const IDEAL_SPLIT_SIZE: u64 = 100_000;

pub mod api;
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

pub mod layout;
pub mod multi;
mod repeated_scan;
#[cfg(test)]
mod test;

pub use repeated_scan::RepeatedScan;
