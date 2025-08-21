// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::SegmentSpec;
use futures::FutureExt;
use std::sync::Arc;
use vortex_error::vortex_err;
use vortex_io::runtime::VortexRead;
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

pub struct FileSegmentSource {
    segment_map: Arc<[SegmentSpec]>,
    read: Arc<dyn VortexRead>,
}

impl FileSegmentSource {
    pub fn new(segment_map: Arc<[SegmentSpec]>, read: Arc<dyn VortexRead>) -> Self {
        Self { segment_map, read }
    }
}

impl SegmentSource for FileSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let segment_map = self.segment_map.clone();
        let read = self.read.clone();
        async move {
            let spec = segment_map
                .get(*id as usize)
                .ok_or_else(|| vortex_err!("Segment {} not found", id))?;
            read.read(spec.offset, spec.length as usize, spec.alignment)
                .await
        }
        .boxed()
    }
}
