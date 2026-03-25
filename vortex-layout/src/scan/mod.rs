// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod arrow;
mod filter;
pub mod layout;
pub mod multi;
pub mod repeated_scan;
pub mod scan_builder;
pub mod split_by;
mod splits;
mod tasks;
#[cfg(test)]
mod test;
pub mod v2_layout;

/// A heuristic for an ideal split size.
///
/// We don't actually know if this is right, but it is probably a good estimate.
const IDEAL_SPLIT_SIZE: u64 = 100_000;
