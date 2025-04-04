use std::sync::Arc;

use futures::FutureExt;
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

use crate::segments::SegmentCache;

/// A [`SegmentSource`] that first tries to look up segments in the cache.
pub struct CachedSegmentSource {
    cache: Arc<dyn SegmentCache>,
    delegate: Arc<dyn SegmentSource>,
    /// Whether to store segments in the cache on successful retrieval.
    store: bool,
}

impl CachedSegmentSource {
    pub fn new(
        cache: Arc<dyn SegmentCache>,
        delegate: Arc<dyn SegmentSource>,
        store: bool,
    ) -> Self {
        Self {
            cache,
            delegate,
            store,
        }
    }
}

impl SegmentSource for CachedSegmentSource {
    fn request(&self, id: SegmentId, for_whom: &Arc<str>) -> SegmentFuture {
        let cache = self.cache.clone();
        let delegate = self.delegate.request(id, for_whom);
        let store = self.store;
        let for_whom = for_whom.clone();

        async move {
            if let Ok(Some(segment)) = cache.get(id).await {
                log::debug!("Resolved segment {} for {} from cache", id, &for_whom);
                return Ok(segment);
            }
            let result = delegate.await?;
            if store {
                if let Err(e) = cache.put(id, result.clone()).await {
                    log::warn!(
                        "Failed to store segment {} for {} in cache: {}",
                        id,
                        &for_whom,
                        e
                    );
                }
            }
            Ok(result)
        }
        .boxed()
    }
}
