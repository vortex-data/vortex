// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod coalesced;
mod direct;

use crate::SegmentSpec;
use crate::segments::SegmentCache;
pub use coalesced::*;
pub use direct::*;
use std::sync::Arc;
use vortex_error::VortexResult;
use vortex_io::ReadAt;
use vortex_layout::segments::SegmentSource;
use vortex_metrics::VortexMetrics;

/// A trait for providing an implementation of a [`SegmentSource`].
pub trait FileDriver {
    fn create_segment_source(
        &self,
        read: Arc<dyn ReadAt>,
        segments: Arc<[SegmentSpec]>,
        // TODO(ngates): pass in the initial read buffer instead?
        segment_cache: Option<Arc<dyn SegmentCache>>,
        metrics: &VortexMetrics,
    ) -> VortexResult<Arc<dyn SegmentSource>>;
}
