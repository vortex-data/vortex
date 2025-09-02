// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::SegmentSpec;
use std::sync::Arc;
use vortex_error::{vortex_err, VortexResult};
use vortex_io::runtime::FileIo;
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

/// A segment source that reads segments from a file using the footer's segment map.
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
    fn request(&self, id: SegmentId) -> VortexResult<SegmentFuture<'rt>> {
        let spec = self
            .segment_map
            .get(*id as usize)
            .cloned()
            .ok_or_else(|| vortex_err!("Segment {} not found", id))?;

        let file = self.file.clone();
        Ok(SegmentFuture::new(spec.length as u64, async move {
            file.read(spec.offset, spec.length as usize, spec.alignment)
                .await
        }))

        // // Eagerly submit the I/O request, we if the SegmentFuture is dropped, this read may
        // // be canceled before it's even initiated.
        // let spec = segment_map
        //     .get(*id as usize)
        //     .cloned()
        //     .ok_or_else(|| vortex_err!("Segment {} not found", id))
        //     .map(move |spec| async move {
        //         file.read(spec.offset, spec.length as usize, spec.alignment)
        //             .await
        //     });
        //
        // async move { spec?.await }.boxed()
    }
}
