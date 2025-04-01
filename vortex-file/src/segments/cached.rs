use std::sync::Arc;

use futures::FutureExt;
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
        let cache = self.cache.clone();
        let delegate = self.delegate.request(id, for_whom);
        let for_whom = for_whom.clone();
        async move {
            if let Ok(Some(segment)) = cache.get(id).await {
                log::debug!("Resolved segment {} for {} from cache", id, &for_whom);
                return Ok(segment);
            }
            delegate.await
        }
        .boxed()
    }
}
