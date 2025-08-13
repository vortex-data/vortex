// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::segments::SegmentId;
use futures::future::{BoxFuture, try_join_all};
use std::sync::Arc;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexResult};
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<ByteBuffer>>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;
}

pub trait SegmentSourceExt: SegmentSource {
    fn request_all<'a>(
        self: Arc<Self>,
        segment_ids: &'a HashSet<SegmentId>,
    ) -> impl Future<Output = VortexResult<HashMap<SegmentId, ByteBuffer>>> + Send + 'a {
        let src = self;
        async move {
            Ok(HashMap::from_iter(
                try_join_all(segment_ids.iter().map(move |id| {
                    let src = src.clone();
                    async move {
                        let segment = src.request(*id).await?;
                        Ok::<_, VortexError>((*id, segment))
                    }
                }))
                .await?,
            ))
        }
    }
}

impl<S: SegmentSource + ?Sized> SegmentSourceExt for S {}
