// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Python-facing wrapper for the OpenDAL-backed Tencent Cloud COS object store.
//!
//! This lets callers build a concrete store object in Python and pass it to
//! `vortex.io.read_url(store=...)` / `vortex.io.write(..., store=...)` exactly like any
//! of the built-in store classes (S3, Azure, ...). The store is constructed from the same
//! configuration the `cos://` URL registry uses, but materialized eagerly so it
//! can be handed around as a first-class value.
//!
//! This module is only compiled when the `opendal` feature is enabled.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vortex_cloud::opendal::CosConfig;

/// A Tencent Cloud COS object store, backed by OpenDAL.
///
/// Construct it with explicit configuration and pass it to
/// ``vortex.io.read_url(url, store=cos_store)`` / ``vortex.io.write(arrays, path, store=cos_store)``.
#[pyclass(name = "CosStore", module = "vortex._lib", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct CosStore {
    store: Arc<dyn object_store::ObjectStore>,
}

impl CosStore {
    /// Clone the underlying object store as an `Arc<dyn ObjectStore>`.
    pub fn to_arc(&self) -> Arc<dyn object_store::ObjectStore> {
        Arc::clone(&self.store)
    }
}

#[pymethods]
impl CosStore {
    #[new]
    #[pyo3(signature = (
        bucket,
        endpoint,
        *,
        secret_id = None,
        secret_key = None,
        root = None,
        disable_config_load = false,
    ))]
    fn new(
        bucket: String,
        endpoint: String,
        secret_id: Option<String>,
        secret_key: Option<String>,
        root: Option<String>,
        disable_config_load: bool,
    ) -> PyResult<Self> {
        let config = CosConfig {
            bucket,
            endpoint,
            secret_id,
            secret_key,
            root,
            disable_config_load,
        };
        let store = vortex_cloud::opendal::make_cos_store(config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { store })
    }
}

/// Register the OpenDAL-backed store classes on the `vortex._lib` module.
pub(crate) fn init(_py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_class::<CosStore>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_cloud::opendal::CosConfig;

    use super::*;

    /// `make_cos_store` (the strongly-typed entry point used by the `CosStore` pyclass) must
    /// accept a fully-specified `CosConfig` and yield a usable `Arc<dyn ObjectStore>`.
    #[test]
    fn cos_config_builds_object_store() {
        let config = CosConfig {
            bucket: "my-bucket".to_string(),
            endpoint: "https://cos.ap-guangzhou.myqcloud.com".to_string(),
            secret_id: Some("AKID".to_string()),
            secret_key: Some("secret".to_string()),
            root: Some("nested/prefix".to_string()),
            disable_config_load: true,
        };
        let store = vortex_cloud::opendal::make_cos_store(config).expect("cos store should build");
        // Sanity-check the result is a non-null `Arc<dyn ObjectStore>`.
        assert!(Arc::strong_count(&store) >= 1);
    }

    /// Constructing a `CosConfig` with an empty `bucket` must surface as a `MissingConfig` error
    /// rather than silently building an invalid store.
    #[test]
    fn cos_config_rejects_empty_bucket() {
        let result = vortex_cloud::opendal::make_cos_store(CosConfig {
            bucket: String::new(),
            endpoint: "https://cos.ap-guangzhou.myqcloud.com".to_string(),
            ..CosConfig::default()
        });
        assert!(matches!(
            result,
            Err(vortex_cloud::opendal::OpenDALStoreError::MissingConfig(
                "bucket"
            ))
        ));
    }
}
