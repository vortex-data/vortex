use std::sync::Arc;

use chrono::{DateTime, Utc};
use datafusion_common::ScalarValue;
use moka::future::Cache;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use vortex_array::ArrayRegistry;
use vortex_array::aliases::DefaultHashBuilder;
use vortex_array::stats::{Precision, Stat};
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_err};
use vortex_file::{Footer, Segment, VortexOpenOptions};
use vortex_io::ObjectStoreReadAt;
use vortex_layout::LayoutRegistry;
use vortex_layout::segments::SegmentId;

#[derive(Debug, Clone)]
pub(crate) struct FooterCache {
    inner: Cache<Key, Footer, DefaultHashBuilder>,
    array_registry: Arc<ArrayRegistry>,
    layout_registry: Arc<LayoutRegistry>,
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
fn estimate_layout_size(footer: &Footer) -> usize {
    let segments_size = footer.segment_map().len() * size_of::<Segment>();
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
        + root_layout.metadata().map(|b| b.len()).unwrap_or_default()
        + root_layout.nsegments() * size_of::<SegmentId>();

    segments_size + stats_size + layout_size
}

impl FooterCache {
    pub fn new(
        size_mb: usize,
        array_registry: Arc<ArrayRegistry>,
        layout_registry: Arc<LayoutRegistry>,
    ) -> Self {
        let inner = Cache::builder()
            .max_capacity(size_mb as u64 * (2 << 20))
            .eviction_listener(|k: Arc<Key>, _v: Footer, cause| {
                log::trace!("Removed {} due to {:?}", k.location, cause);
            })
            .weigher(|_k, footer| {
                let size = estimate_layout_size(footer);
                u32::try_from(size).unwrap_or(u32::MAX)
            })
            .build_with_hasher(DefaultHashBuilder::default());

        Self {
            inner,
            array_registry,
            layout_registry,
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(location = object.location.as_ref())))]
    pub async fn try_get(
        &self,
        object: &ObjectMeta,
        object_store: Arc<dyn ObjectStore>,
    ) -> VortexResult<Footer> {
        self.inner
            .try_get_with(Key::from(object), async {
                let os_read_at =
                    ObjectStoreReadAt::new(object_store, object.location.clone(), None);
                let vxf = VortexOpenOptions::file(os_read_at)
                    .with_array_registry(self.array_registry.clone())
                    .with_layout_registry(self.layout_registry.clone())
                    .with_file_size(object.size as u64)
                    .open()
                    .await?;
                Ok(vxf.footer().clone())
            })
            .await
            .map_err(|e: Arc<VortexError>| {
                Arc::try_unwrap(e).unwrap_or_else(|e| vortex_err!("{}", e.to_string()))
            })
    }
}
