use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use object_store::aws::AwsCredential;
use object_store::CredentialProvider;
use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::aws::store::PyAmazonS3Config;
use crate::credentials::{is_awaitable, TemporaryToken, TokenCache};

/// A wrapper around an [AwsCredential] that includes an optional expiry timestamp.
struct PyAwsCredential {
    credential: AwsCredential,
    expires_at: Option<DateTime<Utc>>,
}

impl<'py> FromPyObject<'_, 'py> for PyAwsCredential {
    type Error = PyErr;

    /// Converts from a Python dictionary of the form
    ///
    /// ```py
    /// class S3Credential(TypedDict):
    ///     access_key_id: str
    ///     secret_access_key: str
    ///     token: str | None
    ///     expires_at: datetime | None
    /// ```
    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let py = obj.py();
        let key_id = obj.get_item(intern!(py, "access_key_id"))?.extract()?;
        let secret_key = obj.get_item(intern!(py, "secret_access_key"))?.extract()?;
        let token = if let Ok(token) = obj.get_item(intern!(py, "token")) {
            token.extract()?
        } else {
            // Allow the dictionary not having a `token` key (so `get_item` will `Err` above)
            None
        };
        let credential = AwsCredential {
            key_id,
            secret_key,
            token,
        };
        let expires_at = obj.get_item(intern!(py, "expires_at"))?.extract()?;
        Ok(Self {
            credential,
            expires_at,
        })
    }
}

// TODO: don't use a cache for static credentials where `expires_at` is `None`
// (so you don't need to access a mutex)
#[derive(Debug)]
pub struct PyAWSCredentialProvider {
    /// The provided user callback to manage credential refresh
    user_callback: Py<PyAny>,
    cache: TokenCache<Arc<AwsCredential>>,
    /// An optional config passed down from the credential provider class
    config: Option<PyAmazonS3Config>,
}

impl PyAWSCredentialProvider {
    /// Access the S3 config passed down from the credential provider
    pub(crate) fn config(&self) -> Option<&PyAmazonS3Config> {
        self.config.as_ref()
    }

    fn equals(&self, py: Python, other: &Self) -> PyResult<bool> {
        self.user_callback
            .call_method1(py, "__eq__", PyTuple::new(py, vec![&other.user_callback])?)?
            .extract(py)
    }
}

impl Clone for PyAWSCredentialProvider {
    fn clone(&self) -> Self {
        let cloned_callback = Python::attach(|py| self.user_callback.clone_ref(py));
        Self {
            user_callback: cloned_callback,
            cache: self.cache.clone(),
            config: self.config.clone(),
        }
    }
}

impl PartialEq for PyAWSCredentialProvider {
    fn eq(&self, other: &Self) -> bool {
        Python::attach(|py| self.equals(py, other)).unwrap_or(false)
    }
}

impl<'py> FromPyObject<'_, 'py> for PyAWSCredentialProvider {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if !obj.hasattr(intern!(obj.py(), "__call__"))? {
            return Err(PyTypeError::new_err(
                "Expected callable object for credential_provider.",
            ));
        }
        let mut cache = TokenCache::default();
        if let Ok(refresh_threshold) = obj.getattr(intern!(obj.py(), "refresh_threshold")) {
            cache = cache.with_min_ttl(refresh_threshold.extract()?);
        }

        let config = if let Ok(config) = obj.getattr(intern!(obj.py(), "config")) {
            config.extract()?
        } else {
            // Allow not having a `config` attribute
            None
        };

        Ok(Self {
            user_callback: obj.as_unbound().clone_ref(obj.py()),
            cache,
            config,
        })
    }
}

impl<'py> IntoPyObject<'py> for PyAWSCredentialProvider {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (&self).into_pyobject(py)
    }
}

impl<'py> IntoPyObject<'py> for &PyAWSCredentialProvider {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.user_callback.bind(py).clone())
    }
}

/// Note: This is copied across providers at the moment
enum PyCredentialProviderResult {
    Async(Py<PyAny>),
    Sync(PyAwsCredential),
}

impl PyCredentialProviderResult {
    async fn resolve(self) -> PyResult<PyAwsCredential> {
        match self {
            Self::Sync(credentials) => Ok(credentials),
            Self::Async(coroutine) => {
                let future = Python::attach(|py| {
                    pyo3_async_runtimes::tokio::into_future(coroutine.bind(py).clone())
                })?;
                let result = future.await?;
                Python::attach(|py| result.extract(py))
            }
        }
    }
}

impl<'py> FromPyObject<'_, 'py> for PyCredentialProviderResult {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if is_awaitable(&obj)? {
            Ok(Self::Async(obj.as_unbound().clone_ref(obj.py())))
        } else {
            Ok(Self::Sync(obj.extract()?))
        }
    }
}

impl PyAWSCredentialProvider {
    /// Call the user-provided callback and extract to a token.
    ///
    /// This is separate from `fetch_token` below so that it can return a `PyResult`.
    async fn call(&self) -> PyResult<PyAwsCredential> {
        let call_result = Python::attach(|py| {
            self.user_callback
                .call0(py)?
                .extract::<PyCredentialProviderResult>(py)
        })?;
        call_result.resolve().await
    }

    /// Call the user-provided callback
    async fn fetch_token(&self) -> object_store::Result<TemporaryToken<Arc<AwsCredential>>> {
        let credential = self
            .call()
            .await
            .map_err(|err| object_store::Error::Unauthenticated {
                path: "External AWS credential provider".to_string(),
                source: Box::new(err),
            })?;

        Ok(TemporaryToken {
            token: Arc::new(credential.credential),
            expiry: credential.expires_at,
        })
    }
}

#[async_trait]
impl CredentialProvider for PyAWSCredentialProvider {
    type Credential = AwsCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        self.cache.get_or_insert_with(|| self.fetch_token()).await
    }
}
