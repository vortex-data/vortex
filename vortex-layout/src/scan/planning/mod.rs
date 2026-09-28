// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Layout-level morsels for the planning protocol.
//!
//! A [`SplitMorsel`] evaluates a filter and projection over one split through an unchanged
//! [`LayoutReader`](crate::LayoutReader), and routes the reader's segment reads through the
//! protocol with a [`PollingSegmentSource`]. Planners that decide which splits to read, and that
//! know where each segment lives in a file, belong to the crates that own those sources.

mod segments;
mod split_morsel;

pub use segments::PollingSegmentSource;
pub use segments::SegmentLocation;
pub use split_morsel::SplitMorsel;
