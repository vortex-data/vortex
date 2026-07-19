use std::time::Duration;

use object_store::{BackoffConfig, RetryConfig};
use pyo3::intern;
use pyo3::prelude::*;

#[derive(Clone, Debug, IntoPyObject, IntoPyObjectRef, PartialEq)]
pub struct PyBackoffConfig {
    #[pyo3(item)]
    init_backoff: Duration,
    #[pyo3(item)]
    max_backoff: Duration,
    #[pyo3(item)]
    base: f64,
}

impl<'py> FromPyObject<'_, 'py> for PyBackoffConfig {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let mut backoff_config = BackoffConfig::default();
        let py = obj.py();
        if let Ok(init_backoff) = obj.get_item(intern!(py, "init_backoff")) {
            backoff_config.init_backoff = init_backoff.extract()?;
        }
        if let Ok(max_backoff) = obj.get_item(intern!(py, "max_backoff")) {
            backoff_config.max_backoff = max_backoff.extract()?;
        }
        if let Ok(base) = obj.get_item(intern!(py, "base")) {
            backoff_config.base = base.extract()?;
        }
        Ok(backoff_config.into())
    }
}

impl From<PyBackoffConfig> for BackoffConfig {
    fn from(value: PyBackoffConfig) -> Self {
        BackoffConfig {
            init_backoff: value.init_backoff,
            max_backoff: value.max_backoff,
            base: value.base,
        }
    }
}

impl From<BackoffConfig> for PyBackoffConfig {
    fn from(value: BackoffConfig) -> Self {
        PyBackoffConfig {
            init_backoff: value.init_backoff,
            max_backoff: value.max_backoff,
            base: value.base,
        }
    }
}

#[derive(Clone, Debug, IntoPyObject, IntoPyObjectRef, PartialEq)]
pub struct PyRetryConfig {
    #[pyo3(item)]
    backoff: PyBackoffConfig,
    #[pyo3(item)]
    max_retries: usize,
    #[pyo3(item)]
    retry_timeout: Duration,
}

impl<'py> FromPyObject<'_, 'py> for PyRetryConfig {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let mut retry_config = RetryConfig::default();
        let py = obj.py();
        if let Ok(backoff) = obj.get_item(intern!(py, "backoff")) {
            retry_config.backoff = backoff.extract::<PyBackoffConfig>()?.into();
        }
        if let Ok(max_retries) = obj.get_item(intern!(py, "max_retries")) {
            retry_config.max_retries = max_retries.extract()?;
        }
        if let Ok(retry_timeout) = obj.get_item(intern!(py, "retry_timeout")) {
            retry_config.retry_timeout = retry_timeout.extract()?;
        }
        Ok(retry_config.into())
    }
}

impl From<PyRetryConfig> for RetryConfig {
    fn from(value: PyRetryConfig) -> Self {
        RetryConfig {
            backoff: value.backoff.into(),
            max_retries: value.max_retries,
            retry_timeout: value.retry_timeout,
        }
    }
}

impl From<RetryConfig> for PyRetryConfig {
    fn from(value: RetryConfig) -> Self {
        PyRetryConfig {
            backoff: value.backoff.into(),
            max_retries: value.max_retries,
            retry_timeout: value.retry_timeout,
        }
    }
}
