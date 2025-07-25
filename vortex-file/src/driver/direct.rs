// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use vortex_error::{VortexResult, vortex_err};
use vortex_io::ReadAt;
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};
use vortex_metrics::VortexMetrics;

use crate::SegmentSpec;
use crate::driver::FileDriver;
use crate::segments::SegmentCache;

/// A [`FileDriver`] that directly reads segments from the underlying I/O source, with no
/// coalescing or pre-fetching of segments.
pub struct DirectDriver;

impl FileDriver for DirectDriver {
    fn create_segment_source(
        &self,
        read: Arc<dyn ReadAt>,
        segments: Arc<[SegmentSpec]>,
        _segment_cache: Option<Arc<dyn SegmentCache>>,
        _metrics: &VortexMetrics,
    ) -> VortexResult<Arc<dyn SegmentSource>> {
        Ok(Arc::new(DirectSegmentSource { read, segments }))
    }
}

/// A [`FileDriver`] that directly reads segments from the underlying I/O source.
struct DirectSegmentSource {
    read: Arc<dyn ReadAt>,
    segments: Arc<[SegmentSpec]>,
}

impl SegmentSource for DirectSegmentSource {
    fn request(&self, id: SegmentId, _for_whom: &Arc<str>) -> SegmentFuture {
        let spec = self.segments.get(*id as usize).cloned();
        let read = self.read.clone();

        async move {
            let spec = spec.ok_or_else(|| vortex_err!("Segment ID out of bounds: {id}"))?;
            read.read_range(spec.offset, spec.length as usize, spec.alignment)
                .await
        }
        .boxed()
    }
}
