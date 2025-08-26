// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::SegmentSpec;
use futures::FutureExt;
use std::sync::Arc;
use vortex_error::vortex_err;
use vortex_io::runtime::{FileIo, Handle};
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

pub struct FileSegmentSource {
    segment_map: Arc<[SegmentSpec]>,
    file: FileIo,
}

impl FileSegmentSource {
    pub fn new(segment_map: Arc<[SegmentSpec]>, file: FileIo) -> Self {
        Self { segment_map, file }
    }
}

impl SegmentSource for FileSegmentSource {
    fn request(&self, id: SegmentId, handle: &Handle) -> SegmentFuture {
        let segment_map = self.segment_map.clone();
        let file = self.file.clone();
        async move {
            let spec = segment_map
                .get(*id as usize)
                .ok_or_else(|| vortex_err!("Segment {} not found", id))?;
            file.read(spec.offset, spec.length as usize, spec.alignment)
                .await
        }
        .boxed()
    }
}
