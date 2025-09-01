// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::SegmentSpec;
use futures::FutureExt;
use std::sync::Arc;
use vortex_error::vortex_err;
use vortex_io::runtime::io::FileIo;
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

pub struct FileSegmentSource<'rt> {
    segment_map: Arc<[SegmentSpec]>,
    file: FileIo<'rt>,
}

impl<'rt> FileSegmentSource<'rt> {
    pub fn new(segment_map: Arc<[SegmentSpec]>, file: FileIo<'rt>) -> Self {
        Self { segment_map, file }
    }
}

impl<'rt> SegmentSource<'rt> for FileSegmentSource<'rt> {
    fn request(&self, id: SegmentId) -> SegmentFuture<'rt> {
        let segment_map = self.segment_map.clone();
        let file = self.file.clone();
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
