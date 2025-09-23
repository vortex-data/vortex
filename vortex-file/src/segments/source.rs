// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::{FutureExt, TryFutureExt};
use vortex_error::{VortexError, vortex_err};
use vortex_io::VortexReadAt;
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

use crate::SegmentSpec;

pub struct FileSegmentSource {
    segments: Arc<[SegmentSpec]>,
    read: Arc<dyn VortexReadAt>,
}

impl FileSegmentSource {
    pub fn new(segments: Arc<[SegmentSpec]>, read: Arc<dyn VortexReadAt>) -> Self {
        Self { segments, read }
    }
}

impl SegmentSource for FileSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        // We eagerly create the read future here assuming the behaviour of [`FileRead`], where
        // coalescing becomes effective prior to the future being polled.
        let maybe_fut = self.segments.get(*id as usize).cloned().map(|spec| {
            self.read
                .clone()
                .read_at(spec.offset, spec.length as usize, spec.alignment)
                .map_err(VortexError::from)
        });

        async move {
            maybe_fut
                .ok_or_else(|| vortex_err!("Missing segment: {}", id))?
                .await
        }
        .boxed()
    }
}
