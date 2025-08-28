// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::file::FileIoSource;
use crate::SegmentSpec;
use futures::FutureExt;
use std::sync::Arc;
use vortex_error::vortex_err;
use vortex_io::runtime::Handle;
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

pub struct FileSegmentSource {
    segment_map: Arc<[SegmentSpec]>,
    file: FileIoSource,
}

impl FileSegmentSource {
    pub fn new(segment_map: Arc<[SegmentSpec]>, file: FileIoSource) -> Self {
        Self { segment_map, file }
    }
}

impl SegmentSource for FileSegmentSource {
    fn request<'handle>(&self, id: SegmentId, handle: &Handle<'handle>) -> SegmentFuture<'handle> {
        let segment_map = self.segment_map.clone();

        // FIXME(ngates): if we want to avoid opening this every time, we need SegmentSource
        //  have a handle lifetime.
        let file = self.file.clone().open(&handle);

        async move {
            let spec = segment_map
                .get(*id as usize)
                .ok_or_else(|| vortex_err!("Segment {} not found", id))?;
            let resp = file
                .read(spec.offset, spec.length as usize, spec.alignment)
                .await;
            resp
        }
        .boxed()
    }
}
