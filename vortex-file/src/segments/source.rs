// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use vortex_error::vortex_err;
use vortex_io::{Dispatch, IoDispatcher, VortexReadAt};
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

use crate::SegmentSpec;

pub struct FileSegmentSource<R> {
    read: Arc<R>,
    segments: Arc<[SegmentSpec]>,
    io_dispatcher: IoDispatcher,
}

impl<R> FileSegmentSource<R> {
    pub fn new(read: Arc<R>, segments: Arc<[SegmentSpec]>, io_dispatcher: IoDispatcher) -> Self {
        Self {
            read,
            segments,
            io_dispatcher,
        }
    }
}

impl<R: VortexReadAt + Send + Sync> SegmentSource for FileSegmentSource<R> {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let segments = self.segments.clone();
        let read = self.read.clone();
        let dispatcher = self.io_dispatcher.clone();
        async move {
            dispatcher
                .dispatch(move || async move {
                    let spec: &SegmentSpec = segments
                        .get(*id as usize)
                        .ok_or_else(|| vortex_err!("Unknown segment"))?;
                    Ok(read
                        .read_byte_range(
                            spec.offset..spec.offset + spec.length as u64,
                            spec.alignment,
                        )
                        .await?)
                })?
                .await?
        }
        .boxed()
    }
}
