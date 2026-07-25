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
use url::Url;
use vortex_utils::aliases::hash_map::HashMap;

/// Build an `object_store::ObjectStore` for the given `cos://` / `oss://` URL and properties.
fn build_store(
    url: &str,
    properties: HashMap<String, String>,
) -> PyResult<Arc<dyn object_store::ObjectStore>> {
    let url = Url::parse(url)
        .map_err(|e| PyValueError::new_err(format!("invalid store URL {url}: {e}")))?;
    let store: Arc<dyn object_store::ObjectStore> =
        vortex_object_store_opendal::make_opendal_store(&url, &properties)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(store)
}

/// A Tencent Cloud COS object store, backed by OpenDAL.
///
/// Construct it with explicit configuration and pass it to
/// ``vortex.io.read_url(url, store=cos_store)`` / ``vortex.io.write(arrays, path, store=cos_store)``.
#[pyclass(name = "CosStore", module = "vortex._lib", from_py_object)]
#[derive(Clone)]
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
        let mut properties = HashMap::new();
        properties.insert("bucket".to_string(), bucket);
        properties.insert("endpoint".to_string(), endpoint);
        if let Some(v) = secret_id {
            properties.insert("secret_id".to_string(), v);
        }
        if let Some(v) = secret_key {
            properties.insert("secret_key".to_string(), v);
        }
        if let Some(v) = root {
            properties.insert("root".to_string(), v);
        }
        if disable_config_load {
            properties.insert("disable_config_load".to_string(), "true".to_string());
        }
        let store = build_store("cos://bucket", properties)?;
        Ok(Self { store })
    }
}

/// Register the OpenDAL-backed store classes on the `vortex._lib` module.
#[cfg(feature = "opendal")]
pub(crate) fn init(_py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_class::<CosStore>()?;
    Ok(())
}

#[cfg(all(test, feature = "opendal"))]
mod tests {
    use super::*;

    /// A `CosStore` built in Rust must yield a non-null `Arc<dyn ObjectStore>` that can be
    /// handed to `read_url`/`write`.
    #[test]
    fn cos_store_builds_object_store() {
        let store = CosStore::new(
            "my-bucket".to_string(),
            "https://cos.ap-guangzhou.myqcloud.com".to_string(),
            Some("AKID".to_string()),
            Some("secret".to_string()),
            None,
            false,
        )
        .expect("cos store should build");
        // The store must be usable as an `Arc<dyn ObjectStore>`.
        let arc: Arc<dyn object_store::ObjectStore> = store.to_arc();
        assert!(Arc::strong_count(&arc) >= 1);
    }
}
