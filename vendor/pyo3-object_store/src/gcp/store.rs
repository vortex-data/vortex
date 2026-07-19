use std::collections::HashMap;
use std::sync::Arc;

use object_store::gcp::{GoogleCloudStorage, GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::ObjectStoreScheme;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};
use pyo3::{intern, IntoPyObjectExt};
use url::Url;

use crate::client::PyClientOptions;
use crate::config::PyConfigValue;
use crate::error::{GenericError, ParseUrlError, PyObjectStoreError, PyObjectStoreResult};
use crate::gcp::credentials::PyGcpCredentialProvider;
use crate::path::PyPath;
use crate::retry::PyRetryConfig;
use crate::{MaybePrefixedStore, PyUrl};

#[derive(Debug, Clone, PartialEq)]
struct GCSConfig {
    prefix: Option<PyPath>,
    config: PyGoogleConfig,
    client_options: Option<PyClientOptions>,
    retry_config: Option<PyRetryConfig>,
    credential_provider: Option<PyGcpCredentialProvider>,
}

impl GCSConfig {
    fn bucket(&self) -> &str {
        self.config
            .0
            .get(&PyGoogleConfigKey(GoogleConfigKey::Bucket))
            .expect("Bucket should always exist in the config")
            .as_ref()
    }

    fn __getnewargs_ex__<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let args = PyTuple::empty(py).into_bound_py_any(py)?;
        let kwargs = PyDict::new(py);

        if let Some(prefix) = &self.prefix {
            kwargs.set_item(intern!(py, "prefix"), prefix.as_ref().as_ref())?;
        }
        kwargs.set_item(intern!(py, "config"), &self.config)?;
        if let Some(client_options) = &self.client_options {
            kwargs.set_item(intern!(py, "client_options"), client_options)?;
        }
        if let Some(retry_config) = &self.retry_config {
            kwargs.set_item(intern!(py, "retry_config"), retry_config)?;
        }
        if let Some(credential_provider) = &self.credential_provider {
            kwargs.set_item("credential_provider", credential_provider)?;
        }

        PyTuple::new(py, [args, kwargs.into_bound_py_any(py)?])
    }
}

/// A Python-facing wrapper around a [`GoogleCloudStorage`].
#[derive(Debug, Clone)]
#[pyclass(name = "GCSStore", frozen, subclass, from_py_object)]
pub struct PyGCSStore {
    store: Arc<MaybePrefixedStore<GoogleCloudStorage>>,
    /// A config used for pickling. This must stay in sync with the underlying store's config.
    config: GCSConfig,
}

impl AsRef<Arc<MaybePrefixedStore<GoogleCloudStorage>>> for PyGCSStore {
    fn as_ref(&self) -> &Arc<MaybePrefixedStore<GoogleCloudStorage>> {
        &self.store
    }
}

impl PyGCSStore {
    /// Consume self and return the underlying [`GoogleCloudStorage`].
    pub fn into_inner(self) -> Arc<MaybePrefixedStore<GoogleCloudStorage>> {
        self.store
    }
}

#[pymethods]
impl PyGCSStore {
    // Create from parameters
    #[new]
    #[pyo3(signature = (bucket=None, *, prefix=None, config=None, client_options=None, retry_config=None, credential_provider=None, **kwargs))]
    fn new(
        bucket: Option<String>,
        prefix: Option<PyPath>,
        config: Option<PyGoogleConfig>,
        client_options: Option<PyClientOptions>,
        retry_config: Option<PyRetryConfig>,
        credential_provider: Option<PyGcpCredentialProvider>,
        kwargs: Option<PyGoogleConfig>,
    ) -> PyObjectStoreResult<Self> {
        let mut builder = GoogleCloudStorageBuilder::from_env();
        let mut config = config.unwrap_or_default();
        if let Some(bucket) = bucket.clone() {
            // Note: we apply the bucket to the config, not directly to the builder, so they stay
            // in sync.
            config.insert_raising_if_exists(GoogleConfigKey::Bucket, bucket)?;
        }
        let combined_config = combine_config_kwargs(Some(config), kwargs)?;
        builder = combined_config.clone().apply_config(builder);
        if let Some(client_options) = client_options.clone() {
            builder = builder.with_client_options(client_options.into())
        }
        if let Some(retry_config) = retry_config.clone() {
            builder = builder.with_retry(retry_config.into())
        }
        if let Some(credential_provider) = credential_provider.clone() {
            builder = builder.with_credentials(Arc::new(credential_provider));
        }
        Ok(Self {
            store: Arc::new(MaybePrefixedStore::new(builder.build()?, prefix.clone())),
            config: GCSConfig {
                prefix,
                config: combined_config,
                client_options,
                retry_config,
                credential_provider,
            },
        })
    }

    #[classmethod]
    #[pyo3(signature = (url, *, config=None, client_options=None, retry_config=None, credential_provider=None, **kwargs))]
    pub(crate) fn from_url<'py>(
        cls: &Bound<'py, PyType>,
        url: PyUrl,
        config: Option<PyGoogleConfig>,
        client_options: Option<PyClientOptions>,
        retry_config: Option<PyRetryConfig>,
        credential_provider: Option<PyGcpCredentialProvider>,
        kwargs: Option<PyGoogleConfig>,
    ) -> PyObjectStoreResult<Bound<'py, PyAny>> {
        // We manually parse the URL to find the prefix because `parse_url` does not apply the
        // prefix.
        let (_, prefix) =
            ObjectStoreScheme::parse(url.as_ref()).map_err(object_store::Error::from)?;
        let prefix: Option<String> = if prefix.parts().count() != 0 {
            Some(prefix.into())
        } else {
            None
        };
        let config = parse_url(config, url.as_ref())?;

        // Note: we pass **back** through Python so that if cls is a subclass, we instantiate the
        // subclass
        let kwargs = kwargs.unwrap_or_default().into_pyobject(cls.py())?;
        kwargs.set_item("prefix", prefix)?;
        kwargs.set_item("config", config)?;
        kwargs.set_item("client_options", client_options)?;
        kwargs.set_item("retry_config", retry_config)?;
        kwargs.set_item("credential_provider", credential_provider)?;
        Ok(cls.call((), Some(&kwargs))?)
    }

    fn __eq__(&self, other: &Bound<PyAny>) -> bool {
        // Ensure we never error on __eq__ by returning false if the other object is not the same
        // type
        other
            .cast::<PyGCSStore>()
            .map(|other| self.config == other.get().config)
            .unwrap_or(false)
    }

    fn __getnewargs_ex__<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.config.__getnewargs_ex__(py)
    }

    fn __repr__(&self) -> String {
        let bucket = self.config.bucket();
        if let Some(prefix) = &self.config.prefix {
            format!(
                "GCSStore(bucket=\"{}\", prefix=\"{}\")",
                bucket,
                prefix.as_ref()
            )
        } else {
            format!("GCSStore(bucket=\"{bucket}\")")
        }
    }

    #[getter]
    fn prefix(&self) -> Option<&PyPath> {
        self.config.prefix.as_ref()
    }

    #[getter]
    fn config(&self) -> &PyGoogleConfig {
        &self.config.config
    }

    #[getter]
    fn client_options(&self) -> Option<&PyClientOptions> {
        self.config.client_options.as_ref()
    }

    #[getter]
    fn credential_provider(&self) -> Option<&PyGcpCredentialProvider> {
        self.config.credential_provider.as_ref()
    }

    #[getter]
    fn retry_config(&self) -> Option<&PyRetryConfig> {
        self.config.retry_config.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyGoogleConfigKey(GoogleConfigKey);

impl<'py> FromPyObject<'_, 'py> for PyGoogleConfigKey {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let s = obj.extract::<PyBackedStr>()?.to_lowercase();
        // https://github.com/apache/arrow-rs-object-store/pull/467
        if s == "application_credentials" {
            return Ok(Self(GoogleConfigKey::ApplicationCredentials));
        }

        let key = s.parse().map_err(PyObjectStoreError::ObjectStoreError)?;
        Ok(Self(key))
    }
}

impl AsRef<str> for PyGoogleConfigKey {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

impl<'py> IntoPyObject<'py> for PyGoogleConfigKey {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (&self).into_pyobject(py)
    }
}

impl<'py> IntoPyObject<'py> for &PyGoogleConfigKey {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let s = self
            .0
            .as_ref()
            .strip_prefix("google_")
            .expect("Expected config prefix to start with google_");
        Ok(PyString::new(py, s))
    }
}

impl From<GoogleConfigKey> for PyGoogleConfigKey {
    fn from(value: GoogleConfigKey) -> Self {
        Self(value)
    }
}

impl From<PyGoogleConfigKey> for GoogleConfigKey {
    fn from(value: PyGoogleConfigKey) -> Self {
        value.0
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, IntoPyObject, IntoPyObjectRef)]
pub struct PyGoogleConfig(HashMap<PyGoogleConfigKey, PyConfigValue>);

// Note: we manually impl FromPyObject instead of deriving it so that we can raise an
// UnknownConfigurationKeyError instead of a `TypeError` on invalid config keys.
//
// We also manually impl this so that we can raise on duplicate keys.
impl<'py> FromPyObject<'_, 'py> for PyGoogleConfig {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let mut slf = Self::new();
        for (key, val) in obj.extract::<Bound<'py, PyDict>>()?.iter() {
            slf.insert_raising_if_exists(
                key.extract::<PyGoogleConfigKey>()?,
                val.extract::<PyConfigValue>()?,
            )?;
        }
        Ok(slf)
    }
}

impl PyGoogleConfig {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn apply_config(self, mut builder: GoogleCloudStorageBuilder) -> GoogleCloudStorageBuilder {
        for (key, value) in self.0.into_iter() {
            builder = builder.with_config(key.0, value.0);
        }
        builder
    }

    fn merge(mut self, other: PyGoogleConfig) -> PyObjectStoreResult<PyGoogleConfig> {
        for (key, val) in other.0.into_iter() {
            self.insert_raising_if_exists(key, val)?;
        }

        Ok(self)
    }

    fn insert_raising_if_exists(
        &mut self,
        key: impl Into<PyGoogleConfigKey>,
        val: impl Into<String>,
    ) -> PyObjectStoreResult<()> {
        let key = key.into();
        let old_value = self.0.insert(key.clone(), PyConfigValue::new(val.into()));
        if old_value.is_some() {
            return Err(GenericError::new_err(format!(
                "Duplicate key {} provided",
                key.0.as_ref()
            ))
            .into());
        }

        Ok(())
    }

    /// Insert a key only if it does not already exist.
    ///
    /// This is used for URL parsing, where any parts of the URL **do not** override any
    /// configuration keys passed manually.
    fn insert_if_not_exists(&mut self, key: impl Into<PyGoogleConfigKey>, val: impl Into<String>) {
        self.0.entry(key.into()).or_insert(PyConfigValue::new(val));
    }
}

fn combine_config_kwargs(
    config: Option<PyGoogleConfig>,
    kwargs: Option<PyGoogleConfig>,
) -> PyObjectStoreResult<PyGoogleConfig> {
    match (config, kwargs) {
        (None, None) => Ok(Default::default()),
        (Some(x), None) | (None, Some(x)) => Ok(x),
        (Some(config), Some(kwargs)) => Ok(config.merge(kwargs)?),
    }
}

/// Sets properties on this builder based on a URL
///
/// This is vendored from
/// https://github.com/apache/arrow-rs/blob/f7263e253655b2ee613be97f9d00e063444d3df5/object_store/src/gcp/builder.rs#L316-L338
///
/// We do our own URL parsing so that we can keep our own config in sync with what is passed to the
/// underlying ObjectStore builder. Passing the URL on verbatim makes it hard because the URL
/// parsing only happens in `build()`. Then the config parameters we have don't include any config
/// applied from the URL.
fn parse_url(config: Option<PyGoogleConfig>, parsed: &Url) -> object_store::Result<PyGoogleConfig> {
    let host = parsed
        .host_str()
        .ok_or_else(|| ParseUrlError::UrlNotRecognised {
            url: parsed.as_str().to_string(),
        })?;
    let mut config = config.unwrap_or_default();

    match parsed.scheme() {
        "gs" => {
            config.insert_if_not_exists(GoogleConfigKey::Bucket, host);
        }
        scheme => {
            let scheme = scheme.to_string();
            return Err(ParseUrlError::UnknownUrlScheme { scheme }.into());
        }
    }

    Ok(config)
}
