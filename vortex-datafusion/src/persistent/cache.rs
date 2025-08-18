// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use datafusion_common::ScalarValue;
use moka::future::Cache;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexResult, vortex_err};
use vortex::file::segments::SegmentCache;
use vortex::file::{Footer, SegmentSpec, VortexFile, VortexOpenOptions};
use vortex::layout::segments::SegmentId;
use vortex::session::VortexSession;
use vortex::stats::{Precision, Stat};
use vortex::utils::aliases::DefaultHashBuilder;

#[derive(Clone)]
pub(crate) struct VortexFileCache {
    file_cache: Cache<FileKey, VortexFile, DefaultHashBuilder>,
    segment_cache: Cache<SegmentKey, ByteBuffer, DefaultHashBuilder>,
    session: Arc<VortexSession>,
}

/// Cache key for a [`VortexFile`].
#[derive(Hash, Eq, PartialEq, Debug, Clone)]
struct FileKey {
    location: Arc<Path>,
    m_time: DateTime<Utc>,
}

impl From<&ObjectMeta> for FileKey {
    fn from(value: &ObjectMeta) -> Self {
        Self {
            location: Arc::new(value.location.clone()),
            m_time: value.last_modified,
        }
    }
}

/// Global cache key for a segment.
#[derive(Hash, Eq, PartialEq, Debug)]
struct SegmentKey {
    file: FileKey,
    segment_id: SegmentId,
}

impl VortexFileCache {
    pub fn new(size_mb: usize, segment_size_mb: usize, session: Arc<VortexSession>) -> Self {
        let file_cache = Cache::builder()
            .max_capacity(size_mb as u64 * (1 << 20))
            .eviction_listener(|k: Arc<FileKey>, _v: VortexFile, cause| {
                log::trace!("Removed {k:?} due to {cause:?}");
            })
            .weigher(|_k, vxf| {
                u32::try_from(estimate_layout_size(vxf.footer())).unwrap_or(u32::MAX)
            })
            .build_with_hasher(DefaultHashBuilder::default());

        let segment_cache = Cache::builder()
            .max_capacity(segment_size_mb as u64 * (1 << 20))
            .eviction_listener(|k: Arc<SegmentKey>, _v: ByteBuffer, cause| {
                log::trace!("Removed {k:?} due to {cause:?}");
            })
            .weigher(|_k, v| u32::try_from(v.len()).unwrap_or(u32::MAX))
            .build_with_hasher(DefaultHashBuilder::default());

        Self {
            file_cache,
            segment_cache,
            session,
        }
    }

    pub async fn try_get(
        &self,
        object: &ObjectMeta,
        object_store: Arc<dyn ObjectStore>,
    ) -> VortexResult<VortexFile> {
        let file_key = FileKey::from(object);
        self.file_cache
            .try_get_with(
                file_key.clone(),
                VortexOpenOptions::file()
                    // FIXME(ngates): we don't really want to clone on every open...
                    .with_array_registry(Arc::new(self.session.arrays().clone()))
                    .with_layout_registry(Arc::new(self.session.layouts().clone()))
                    .with_metrics(
                        self.session
                            .metrics()
                            .child_with_tags([("filename", object.location.to_string())]),
                    )
                    .with_file_size(object.size)
                    .with_segment_cache(Arc::new(VortexFileSegmentCache {
                        file_key,
                        segment_cache: self.segment_cache.clone(),
                    }))
                    .open_object_store(&object_store, object.location.as_ref()),
            )
            .await
            .map_err(|e: Arc<VortexError>| {
                Arc::try_unwrap(e).unwrap_or_else(|e| vortex_err!("{}", e.to_string()))
            })
    }
}

/// A [`SegmentCache`] implementation that uses the shared global segment cache.
struct VortexFileSegmentCache {
    file_key: FileKey,
    segment_cache: Cache<SegmentKey, ByteBuffer, DefaultHashBuilder>,
}

#[async_trait]
impl SegmentCache for VortexFileSegmentCache {
    async fn get(&self, segment_id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self
            .segment_cache
            .get(&SegmentKey {
                file: self.file_key.clone(),
                segment_id,
            })
            .await)
    }

    async fn put(&self, segment_id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.segment_cache
            .insert(
                SegmentKey {
                    file: self.file_key.clone(),
                    segment_id,
                },
                buffer,
            )
            .await;
        Ok(())
    }
}

/// Approximate the in-memory size of a layout
fn estimate_layout_size(footer: &Footer) -> usize {
    let segments_size = footer.segment_map().len() * size_of::<SegmentSpec>();
    let stats_size = footer
        .statistics()
        .iter()
        .map(|v| {
            v.iter()
                .map(|_| size_of::<Stat>() + size_of::<Precision<ScalarValue>>())
                .sum::<usize>()
        })
        .sum::<usize>();

    let root_layout = footer.layout();
    let layout_size = size_of::<DType>()
        + root_layout.metadata().len()
        + root_layout.segment_ids().len() * size_of::<SegmentId>();

    segments_size + stats_size + layout_size
}
