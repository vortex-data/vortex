use std::sync::Arc;

use chrono::{DateTime, Utc};
use datafusion_common::ScalarValue;
use moka::future::Cache;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use vortex_array::aliases::DefaultHashBuilder;
use vortex_array::stats::{Precision, Stat};
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexError, VortexResult};
use vortex_file::{FileLayout, Segment, VortexOpenOptions};
use vortex_io::ObjectStoreReadAt;
use vortex_layout::segments::SegmentId;

#[derive(Debug, Clone)]
pub(crate) struct FileLayoutCache {
    inner: Cache<Key, FileLayout, DefaultHashBuilder>,
    context: ContextRef,
}

#[derive(Hash, Eq, PartialEq, Debug)]
pub(crate) struct Key {
    location: Path,
    m_time: DateTime<Utc>,
}

impl From<&ObjectMeta> for Key {
    fn from(value: &ObjectMeta) -> Self {
        Self {
            location: value.location.clone(),
            m_time: value.last_modified,
        }
    }
}

/// Approximate the in-memory size of a layout
fn estimate_layout_size(file_layout: &FileLayout) -> usize {
    let segments_size = file_layout.segment_map().len() * size_of::<Segment>();
    let stats_size = file_layout
        .statistics()
        .iter()
        .map(|v| {
            v.iter()
                .map(|_| size_of::<Stat>() + size_of::<Precision<ScalarValue>>())
                .sum::<usize>()
        })
        .sum::<usize>();

    let root_layout = file_layout.root_layout();
    let layout_size = size_of::<DType>()
        + root_layout.metadata().map(|b| b.len()).unwrap_or_default()
        + root_layout.nsegments() * size_of::<SegmentId>();

    segments_size + stats_size + layout_size
}

impl FileLayoutCache {
    pub fn new(size_mb: usize, context: ContextRef) -> Self {
        let inner = Cache::builder()
            .max_capacity(size_mb as u64 * (2 << 20))
            .eviction_listener(|k: Arc<Key>, _v: FileLayout, cause| {
                log::trace!("Removed {} due to {:?}", k.location, cause);
            })
            .weigher(|_k, file_layout| {
                let size = estimate_layout_size(file_layout);
                u32::try_from(size).unwrap_or(u32::MAX)
            })
            .build_with_hasher(DefaultHashBuilder::default());

        Self { inner, context }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(location = object.location.as_ref())))]
    pub async fn try_get(
        &self,
        object: &ObjectMeta,
        object_store: Arc<dyn ObjectStore>,
    ) -> VortexResult<FileLayout> {
        self.inner
            .try_get_with(Key::from(object), async {
                let os_read_at = ObjectStoreReadAt::new(object_store, object.location.clone());
                let vxf = VortexOpenOptions::file(os_read_at)
                    .with_ctx(self.context.clone())
                    .with_file_size(object.size as u64)
                    .open()
                    .await?;
                VortexResult::Ok(vxf.file_layout().clone())
            })
            .await
            .map_err(|e: Arc<VortexError>| {
                Arc::try_unwrap(e).unwrap_or_else(|e| vortex_err!("{}", e.to_string()))
            })
    }
}
