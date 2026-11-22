// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Segment access abstractions for layouts.
//!
//! Layouts refer to byte ranges by [`SegmentId`]. A [`SegmentSource`] resolves those ids to buffer
//! handles for readers, while a [`SegmentSink`] assigns ids when writers emit buffers.
//! [`SegmentCache`] implementations can sit in front of sources to reuse segment bytes across
//! scans.

mod cache;
#[cfg(test)]
mod shared;
mod sink;
mod source;
mod usage;

#[cfg(any(test, feature = "_test-harness"))]
mod test;

pub use cache::*;
pub use sink::*;
pub use source::*;
#[cfg(any(test, feature = "_test-harness"))]
pub use test::*;
pub use usage::*;
pub use vortex_io::segments::SharedSegmentSource;
pub use vortex_scan::source::SegmentId;
