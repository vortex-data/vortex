use std::sync::Arc;

use chrono::{DateTime, Utc};
use moka::future::Cache;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use vortex_array::ContextRef;
use vortex_error::{vortex_err, VortexError, VortexResult};
use vortex_file::v2::{FileLayout, VortexOpenOptions};
use vortex_io::ObjectStoreReadAt;

#[derive(Debug, Clone)]
pub(crate) struct FileLayoutCache {
    inner: Cache<Key, FileLayout>,
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

impl FileLayoutCache {
    pub fn new(size_mb: usize) -> Self {
        let inner = Cache::builder()
            .max_capacity(size_mb as u64 * (2 << 20))
            .eviction_listener(|k: Arc<Key>, _v, cause| {
                log::trace!("Removed {} due to {:?}", k.location, cause);
            })
            .build();

        Self { inner }
    }

    pub async fn try_get(
        &self,
        object: &ObjectMeta,
        store: Arc<dyn ObjectStore>,
    ) -> VortexResult<FileLayout> {
        self.inner
            .try_get_with(Key::from(object), async {
                let os_read_at = ObjectStoreReadAt::new(store.clone(), object.location.clone());
                let vxf = VortexOpenOptions::new(ContextRef::default())
                    .with_file_size(object.size as u64)
                    .open(os_read_at)
                    .await?;
                VortexResult::Ok(vxf.file_layout().clone())
            })
            .await
            .map_err(|e: Arc<VortexError>| match Arc::try_unwrap(e) {
                Ok(e) => e,
                Err(e) => vortex_err!("{}", e.to_string()),
            })
    }
}
