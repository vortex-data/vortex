// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Layout trees, layout readers, scan planning, and segment IO.
//!
//! A [`Layout`] is the serialized, row-counted representation of an array tree. It records logical
//! dtype, child layout relationships, segment ids, and encoding metadata; it does not own the
//! segment bytes. A [`LayoutReader`] pairs a layout with a [`SegmentSource`](segments::SegmentSource)
//! and session so scans can evaluate projections and filters.
//!
//! Most users enter this crate through file APIs, but extension authors implement [`VTable`] and
//! [`LayoutStrategy`] to add new on-disk organizations.
//!
//! Scanning is built with [`scan::scan_builder::ScanBuilder`]. It accepts a bound projection,
//! optional bound filter, optional row range, [`Selection`](vortex_scan::selection::Selection),
//! split strategy, and task concurrency settings, then produces array streams or iterators.
pub mod layouts;
pub mod plan;

pub use flatbuffers::*;
pub use vortex_layout_commons::*;
pub mod aliases;
mod children;
mod flatbuffers;
mod reader {
    pub use vortex_layout_commons::LayoutReader;
    pub use vortex_layout_commons::RowSplits;
    pub use vortex_layout_commons::SplitRange;
}
pub mod scan;
pub mod session;
mod strategy;
pub use strategy::*;
#[cfg(test)]
mod test;

/// Layout tree display helpers.
pub mod display {
    pub use vortex_layout_commons::display::*;
}

/// Segment access contracts and runtime implementations used by layout readers and writers.
pub mod segments;

/// Sequence types used to preserve writer ordering.
pub mod sequence {
    pub use vortex_layout_commons::sequence::*;
}
