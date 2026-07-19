use std::collections::HashMap;
use std::sync::Arc;

use itertools::Itertools;
use object_store::aws::{AmazonS3, AmazonS3Builder, AmazonS3ConfigKey};
use object_store::ObjectStoreScheme;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};
use pyo3::{intern, IntoPyObjectExt};
use url::Url;

use crate::aws::credentials::PyAWSCredentialProvider;
use crate::client::PyClientOptions;
use crate::config::PyConfigValue;
use crate::error::{GenericError, ParseUrlError, PyObjectStoreError, PyObjectStoreResult};
use crate::path::PyPath;
use crate::prefix::MaybePrefixedStore;
use crate::retry::PyRetryConfig;
use crate::PyUrl;

#[derive(Debug, Clone, PartialEq)]
struct S3Config {
    prefix: Option<PyPath>,
    config: PyAmazonS3Config,
    client_options: Option<PyClientOptions>,
    retry_config: Option<PyRetryConfig>,
    credential_provider: Option<PyAWSCredentialProvider>,
}

impl S3Config {
    fn bucket(&self) -> &str {
        self.config
            .0
            .get(&PyAmazonS3ConfigKey(AmazonS3ConfigKey::Bucket))
            .expect("bucket should always exist in the config")
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

/// A Python-facing wrapper around an [`AmazonS3`].
#[derive(Debug, Clone)]
#[pyclass(name = "S3Store", frozen, subclass, from_py_object)]
pub struct PyS3Store {
    store: Arc<MaybePrefixedStore<AmazonS3>>,
    /// A config used for pickling. This must stay in sync with the underlying store's config.
    config: S3Config,
}

impl AsRef<Arc<MaybePrefixedStore<AmazonS3>>> for PyS3Store {
    fn as_ref(&self) -> &Arc<MaybePrefixedStore<AmazonS3>> {
        &self.store
    }
}

impl PyS3Store {
    /// Consume self and return the underlying [`AmazonS3`].
    pub fn into_inner(self) -> Arc<MaybePrefixedStore<AmazonS3>> {
        self.store
    }
}

#[pymethods]
impl PyS3Store {
    // Create from parameters
    #[new]
    #[pyo3(signature = (bucket=None, *, prefix=None, config=None, client_options=None, retry_config=None, credential_provider=None, **kwargs))]
    fn new(
        bucket: Option<String>,
        prefix: Option<PyPath>,
        config: Option<PyAmazonS3Config>,
        client_options: Option<PyClientOptions>,
        retry_config: Option<PyRetryConfig>,
        credential_provider: Option<PyAWSCredentialProvider>,
        kwargs: Option<PyAmazonS3Config>,
    ) -> PyObjectStoreResult<Self> {
        let mut builder = AmazonS3Builder::from_env();
        let mut config = config.unwrap_or_default();

        if let Some(bucket) = bucket {
            // Note: we apply the bucket to the config, not directly to the builder, so they stay
            // in sync.
            config.insert_raising_if_exists(AmazonS3ConfigKey::Bucket, bucket)?;
        }

        let mut combined_config = combine_config_kwargs(config, kwargs)?;

        if let Some(client_options) = client_options.clone() {
            builder = builder.with_client_options(client_options.into())
        }
        if let Some(retry_config) = retry_config.clone() {
            builder = builder.with_retry(retry_config.into())
        }

        if let Some(credential_provider) = credential_provider.clone() {
            // Apply credential provider config onto main config
            if let Some(credential_config) = credential_provider.config() {
                for (key, val) in credential_config.0.iter() {
                    // Give precedence to passed-in config values
                    combined_config.insert_if_not_exists(key.clone(), val.clone());
                }
            }
            builder = builder.with_credentials(Arc::new(credential_provider));
        }

        builder = combined_config.clone().apply_config(builder);

        Ok(Self {
            store: Arc::new(MaybePrefixedStore::new(builder.build()?, prefix.clone())),
            config: S3Config {
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
        config: Option<PyAmazonS3Config>,
        client_options: Option<PyClientOptions>,
        retry_config: Option<PyRetryConfig>,
        credential_provider: Option<PyAWSCredentialProvider>,
        kwargs: Option<PyAmazonS3Config>,
    ) -> PyObjectStoreResult<Bound<'py, PyAny>> {
        // We manually parse the URL to find the prefix because `with_url` does not apply the
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
        // Ensure we never error on __eq__ by returning false if the other object is not an S3Store
        other
            .cast::<PyS3Store>()
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
                "S3Store(bucket=\"{}\", prefix=\"{}\")",
                bucket,
                prefix.as_ref()
            )
        } else {
            format!("S3Store(bucket=\"{bucket}\")")
        }
    }

    #[getter]
    fn prefix(&self) -> Option<&PyPath> {
        self.config.prefix.as_ref()
    }

    #[getter]
    fn config(&self) -> &PyAmazonS3Config {
        &self.config.config
    }

    #[getter]
    fn client_options(&self) -> Option<&PyClientOptions> {
        self.config.client_options.as_ref()
    }

    #[getter]
    fn credential_provider(&self) -> Option<&PyAWSCredentialProvider> {
        self.config.credential_provider.as_ref()
    }

    #[getter]
    fn retry_config(&self) -> Option<&PyRetryConfig> {
        self.config.retry_config.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyAmazonS3ConfigKey(AmazonS3ConfigKey);

impl<'py> FromPyObject<'_, 'py> for PyAmazonS3ConfigKey {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let s = obj.extract::<PyBackedStr>()?.to_lowercase();
        let key = s.parse().map_err(PyObjectStoreError::ObjectStoreError)?;
        Ok(Self(key))
    }
}

impl AsRef<str> for PyAmazonS3ConfigKey {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

impl<'py> IntoPyObject<'py> for &PyAmazonS3ConfigKey {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let s = self
            .0
            .as_ref()
            .strip_prefix("aws_")
            .expect("Expected config prefix to start with aws_");
        Ok(PyString::new(py, s))
    }
}

impl<'py> IntoPyObject<'py> for PyAmazonS3ConfigKey {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (&self).into_pyobject(py)
    }
}

impl From<AmazonS3ConfigKey> for PyAmazonS3ConfigKey {
    fn from(value: AmazonS3ConfigKey) -> Self {
        Self(value)
    }
}

impl From<PyAmazonS3ConfigKey> for AmazonS3ConfigKey {
    fn from(value: PyAmazonS3ConfigKey) -> Self {
        value.0
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, IntoPyObject, IntoPyObjectRef)]
pub struct PyAmazonS3Config(HashMap<PyAmazonS3ConfigKey, PyConfigValue>);

// Note: we manually impl FromPyObject instead of deriving it so that we can raise an
// UnknownConfigurationKeyError instead of a `TypeError` on invalid config keys.
//
// We also manually impl this so that we can raise on duplicate keys.
impl<'py> FromPyObject<'_, 'py> for PyAmazonS3Config {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let mut slf = Self::new();
        for (key, val) in obj.extract::<Bound<'py, PyDict>>()?.iter() {
            slf.insert_raising_if_exists(
                key.extract::<PyAmazonS3ConfigKey>()?,
                val.extract::<PyConfigValue>()?,
            )?;
        }
        Ok(slf)
    }
}

impl PyAmazonS3Config {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn apply_config(self, mut builder: AmazonS3Builder) -> AmazonS3Builder {
        for (key, value) in self.0.into_iter() {
            builder = builder.with_config(key.0, value.0);
        }
        builder
    }

    fn merge(mut self, other: PyAmazonS3Config) -> PyObjectStoreResult<PyAmazonS3Config> {
        for (key, val) in other.0.into_iter() {
            self.insert_raising_if_exists(key, val)?;
        }

        Ok(self)
    }

    fn insert_raising_if_exists(
        &mut self,
        key: impl Into<PyAmazonS3ConfigKey>,
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
    fn insert_if_not_exists(
        &mut self,
        key: impl Into<PyAmazonS3ConfigKey>,
        val: impl Into<String>,
    ) {
        self.0.entry(key.into()).or_insert(PyConfigValue::new(val));
    }
}

fn combine_config_kwargs(
    config: PyAmazonS3Config,
    kwargs: Option<PyAmazonS3Config>,
) -> PyObjectStoreResult<PyAmazonS3Config> {
    if let Some(kwargs) = kwargs {
        config.merge(kwargs)
    } else {
        Ok(config)
    }
}

/// Sets properties on a configuration based on a URL
///
/// This is vendored from
/// https://github.com/apache/arrow-rs/blob/f7263e253655b2ee613be97f9d00e063444d3df5/object_store/src/aws/builder.rs#L600-L647
///
/// We do our own URL parsing so that we can keep our own config in sync with what is passed to the
/// underlying ObjectStore builder. Passing the URL on verbatim makes it hard because the URL
/// parsing only happens in `build()`. Then the config parameters we have don't include any config
/// applied from the URL.
fn parse_url(
    config: Option<PyAmazonS3Config>,
    parsed: &Url,
) -> object_store::Result<PyAmazonS3Config> {
    let host = parsed
        .host_str()
        .ok_or_else(|| ParseUrlError::UrlNotRecognised {
            url: parsed.as_str().to_string(),
        })?;
    let mut config = config.unwrap_or_default();

    match parsed.scheme() {
        "s3" | "s3a" => {
            config.insert_if_not_exists(AmazonS3ConfigKey::Bucket, host);
        }
        "https" => match host.splitn(4, '.').collect_tuple() {
            Some(("s3", region, "amazonaws", "com")) => {
                config.insert_if_not_exists(AmazonS3ConfigKey::Region, region);
                let bucket = parsed.path_segments().into_iter().flatten().next();
                if let Some(bucket) = bucket {
                    config.insert_if_not_exists(AmazonS3ConfigKey::Bucket, bucket);
                }
            }
            Some((bucket, "s3", "amazonaws", "com")) => {
                config.insert_if_not_exists(AmazonS3ConfigKey::Bucket, bucket);
                config.insert_if_not_exists(AmazonS3ConfigKey::VirtualHostedStyleRequest, "true");
            }
            Some((bucket, "s3", region, "amazonaws.com")) => {
                config.insert_if_not_exists(AmazonS3ConfigKey::Bucket, bucket);
                config.insert_if_not_exists(AmazonS3ConfigKey::Region, region);
                config.insert_if_not_exists(AmazonS3ConfigKey::VirtualHostedStyleRequest, "true");
            }
            Some((account, "r2", "cloudflarestorage", "com")) => {
                config.insert_if_not_exists(AmazonS3ConfigKey::Region, "auto");
                let endpoint = format!("https://{account}.r2.cloudflarestorage.com");
                config.insert_if_not_exists(AmazonS3ConfigKey::Endpoint, endpoint);

                let bucket = parsed.path_segments().into_iter().flatten().next();
                if let Some(bucket) = bucket {
                    config.insert_if_not_exists(AmazonS3ConfigKey::Bucket, bucket);
                }
            }
            _ => {
                return Err(ParseUrlError::UrlNotRecognised {
                    url: parsed.as_str().to_string(),
                }
                .into())
            }
        },
        scheme => {
            let scheme = scheme.into();
            return Err(ParseUrlError::UnknownUrlScheme { scheme }.into());
        }
    };

    Ok(config)
}
