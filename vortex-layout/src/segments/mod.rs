// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Segment access contracts and runtime implementations.
//!
//! Layouts refer to byte ranges by [`SegmentId`]. The source and sink contracts are re-exported
//! from [`vortex_layout_commons`], while this module provides the cache and request-sharing
//! policies used by Vortex file readers.

mod cache;
mod shared;

#[cfg(any(test, feature = "_test-harness"))]
mod test;

pub use cache::*;
pub use shared::*;
#[cfg(any(test, feature = "_test-harness"))]
pub use test::*;
pub use vortex_layout_commons::segments::*;
