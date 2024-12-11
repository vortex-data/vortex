use std::sync::Arc;

use chrono::{DateTime, Utc};
use moka::future::Cache;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use vortex_error::{vortex_err, VortexError, VortexResult};
use vortex_file::{read_initial_bytes, InitialRead};
use vortex_io::ObjectStoreReadAt;

#[derive(Debug, Clone)]
pub struct InitialReadCache {
    inner: Cache<Key, InitialRead>,
}

impl Default for InitialReadCache {
    fn default() -> Self {
        let inner = Cache::builder()
            .weigher(|k: &Key, v: &InitialRead| {
                (k.location.as_ref().as_bytes().len() + v.buf.len()) as u32
            })
            .max_capacity(256 * (2 << 20))
            .eviction_listener(|k, _v, cause| {
                log::trace!("Removed {} due to {:?}", k.location, cause);
            })
            .build();

        Self { inner }
    }
}

#[derive(Hash, Eq, PartialEq, Debug)]
pub struct Key {
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

impl InitialReadCache {
    pub async fn try_get(
        &self,
        object: &ObjectMeta,
        store: Arc<dyn ObjectStore>,
    ) -> VortexResult<InitialRead> {
        self.inner
            .try_get_with(Key::from(object), async {
                let os_read_at = ObjectStoreReadAt::new(store.clone(), object.location.clone());
                let initial_read = read_initial_bytes(&os_read_at, object.size as u64).await?;
                VortexResult::Ok(initial_read)
            })
            .await
            .map_err(|e: Arc<VortexError>| match Arc::try_unwrap(e) {
                Ok(e) => e,
                Err(e) => vortex_err!("{}", e.to_string()),
            })
    }
}
