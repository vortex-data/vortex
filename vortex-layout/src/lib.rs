// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Layout trees, layout readers, scan planning, and segment IO.
//!
//! A [`Layout`] is the serialized, row-counted representation of an array tree. It records logical
//! dtype, child layout relationships, segment ids, and encoding metadata; it does not own the
//! segment bytes. A [`LayoutReader`] pairs a layout with a [`SegmentSource`](segments::SegmentSource)
//! and session so scans can evaluate projections and filters.
//!
//! Most users enter this crate through file APIs, but extension authors implement [`VTable`],
//! [`LayoutEncoding`], and [`LayoutStrategy`] to add new on-disk organizations.
//!
//! Scanning is built with [`scan::scan_builder::ScanBuilder`]. It accepts a projection expression,
//! optional filter, optional row range, [`Selection`](vortex_scan::selection::Selection), split
//! strategy, and task concurrency settings, then produces array streams or iterators.
pub mod layouts;

pub use children::*;
pub use encoding::*;
pub use flatbuffers::*;
pub use layout::*;
pub use reader::*;
pub use reader_context::*;
pub use strategy::*;
use vortex_session::registry::Context;
pub use vtable::*;
pub mod aliases;
mod children;
pub mod display;
mod encoding;
mod flatbuffers;
mod layout;
mod reader;
mod reader_context;
pub mod scan;
pub mod segments;
pub mod sequence;
pub mod session;
mod strategy;
#[cfg(test)]
mod test;
pub mod vtable;

pub type LayoutContext = Context<LayoutEncodingRef>;
