use std::sync::Arc;

use futures::FutureExt;
use vortex_array::aliases::hash_map::HashMap;
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

use crate::segments::SegmentCache;

/// A [`SegmentSource`] that first tries to look up segments in the cache.
pub struct CachedSegmentSource {
    cache: Arc<dyn SegmentCache>,
    delegate: Arc<dyn SegmentSource>,
}

impl CachedSegmentSource {
    pub fn new(cache: Arc<dyn SegmentCache>, delegate: Arc<dyn SegmentSource>) -> Self {
        Self { cache, delegate }
    }
}

impl SegmentSource for CachedSegmentSource {
    fn request(&self, id: SegmentId, for_whom: &Arc<str>) -> SegmentFuture {
        if let Ok(Some(segment)) = self.cache.get(id) {
            return async move { Ok(segment) }.boxed();
        }
        self.delegate.request(id, for_whom)
    }
}
