use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use object_store::gcp::GcpCredential;
use object_store::CredentialProvider;
use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::credentials::{is_awaitable, TemporaryToken, TokenCache};

/// Ref https://github.com/apache/arrow-rs/pull/6638
const DEFAULT_GCP_MIN_TTL: TimeDelta = TimeDelta::minutes(4);

/// A wrapper around a [GcpCredential] that includes an optional expiry timestamp.
struct PyGcpCredential {
    credential: GcpCredential,
    expires_at: Option<DateTime<Utc>>,
}

impl<'py> FromPyObject<'_, 'py> for PyGcpCredential {
    type Error = PyErr;

    /// Converts from a Python dictionary of the form
    ///
    /// ```py
    /// class GCSCredential(TypedDict):
    ///     token: str
    ///     expires_at: datetime | None
    /// ```
    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let bearer = obj
            .get_item(intern!(obj.py(), "token"))?
            .extract::<String>()?;
        let credential = GcpCredential { bearer };
        let expires_at = obj.get_item(intern!(obj.py(), "expires_at"))?.extract()?;
        Ok(Self {
            credential,
            expires_at,
        })
    }
}

// TODO: don't use a cache for static credentials where `expires_at` is `None`
// (so you don't need to access a mutex)
#[derive(Debug)]
pub struct PyGcpCredentialProvider {
    /// The provided user callback to manage credential refresh
    user_callback: Py<PyAny>,
    /// The provided user callback to manage credential refresh
    cache: TokenCache<Arc<GcpCredential>>,
}

impl PyGcpCredentialProvider {
    fn equals(&self, py: Python, other: &Self) -> PyResult<bool> {
        self.user_callback
            .call_method1(py, "__eq__", PyTuple::new(py, vec![&other.user_callback])?)?
            .extract(py)
    }
}

impl Clone for PyGcpCredentialProvider {
    fn clone(&self) -> Self {
        let cloned_callback = Python::attach(|py| self.user_callback.clone_ref(py));
        Self {
            user_callback: cloned_callback,
            cache: self.cache.clone(),
        }
    }
}

impl PartialEq for PyGcpCredentialProvider {
    fn eq(&self, other: &Self) -> bool {
        Python::attach(|py| self.equals(py, other)).unwrap_or(false)
    }
}

impl<'py> FromPyObject<'_, 'py> for PyGcpCredentialProvider {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if !obj.hasattr(intern!(obj.py(), "__call__"))? {
            return Err(PyTypeError::new_err(
                "Expected callable object for credential_provider.",
            ));
        }
        let min_ttl =
            if let Ok(refresh_threshold) = obj.getattr(intern!(obj.py(), "refresh_threshold")) {
                refresh_threshold.extract()?
            } else {
                DEFAULT_GCP_MIN_TTL
            };
        let cache = TokenCache::default().with_min_ttl(min_ttl);
        Ok(Self {
            user_callback: obj.as_unbound().clone_ref(obj.py()),
            cache,
        })
    }
}

impl<'py> IntoPyObject<'py> for &PyGcpCredentialProvider {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.user_callback.bind(py).clone())
    }
}

impl<'py> IntoPyObject<'py> for PyGcpCredentialProvider {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (&self).into_pyobject(py)
    }
}

/// Note: This is copied across providers at the moment
enum PyCredentialProviderResult {
    Async(Py<PyAny>),
    Sync(PyGcpCredential),
}

impl PyCredentialProviderResult {
    async fn resolve(self) -> PyResult<PyGcpCredential> {
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

impl PyGcpCredentialProvider {
    /// Call the user-provided callback and extract to a token.
    ///
    /// This is separate from `fetch_token` below so that it can return a `PyResult`.
    async fn call(&self) -> PyResult<PyGcpCredential> {
        let call_result = Python::attach(|py| {
            self.user_callback
                .call0(py)?
                .extract::<PyCredentialProviderResult>(py)
        })?;
        call_result.resolve().await
    }

    /// Call the user-provided callback
    async fn fetch_token(&self) -> object_store::Result<TemporaryToken<Arc<GcpCredential>>> {
        let credential = self
            .call()
            .await
            .map_err(|err| object_store::Error::Unauthenticated {
                path: "External GCP credential provider".to_string(),
                source: Box::new(err),
            })?;

        Ok(TemporaryToken {
            token: Arc::new(credential.credential),
            expiry: credential.expires_at,
        })
    }
}

#[async_trait]
impl CredentialProvider for PyGcpCredentialProvider {
    type Credential = GcpCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        self.cache.get_or_insert_with(|| self.fetch_token()).await
    }
}
