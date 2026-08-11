// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan-native scanning for Vortex layouts.
//!
//! This crate intentionally owns a separate copy of the scan orchestration. It executes
//! [`vortex_layout::plan::Plan`] trees and never constructs a
//! [`vortex_layout::LayoutReader`].
//!
//! Set `RUST_LOG=vortex_scan_v2=debug` to log source and optimized plan trees and selected scan
//! splits. Use `trace` to also log execution of each split.

mod filter;
mod repeated_scan;
mod scan_builder;
mod splits;
mod tasks;

#[cfg(test)]
mod tests;

pub use filter::FilterMode;
pub use repeated_scan::RepeatedScan;
pub use scan_builder::ScanBuilder;
pub use splits::SplitBy;
