// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::{Arc, OnceLock};

use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use vortex::error::{VortexResult, vortex_err};
use vortex_utils::aliases::dash_map::DashMap;

// Global S3 object store cache.
pub fn s3_store(bucket: &str) -> VortexResult<Arc<dyn ObjectStore>> {
    static S3_STORES: OnceLock<DashMap<String, Arc<dyn ObjectStore>>> = OnceLock::new();
    let stores = S3_STORES.get_or_init(|| DashMap::with_hasher(Default::default()));

    fn create_s3_object_store(bucket: &str) -> VortexResult<Arc<dyn ObjectStore>> {
        Ok(Arc::new(
            AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()
                .map_err(|e| vortex_err!("Failed to create S3 store: {}", e))?,
        ) as Arc<dyn ObjectStore>)
    }

    let object_store = match stores.get(bucket) {
        Some(store) => store.clone(),
        None => {
            let store = create_s3_object_store(bucket)?;
            stores.insert(bucket.to_string(), store.clone());
            store
        }
    };

    Ok(object_store)
}
